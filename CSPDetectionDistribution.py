import torch
import pyro.distributions as dist
import torch.nn.functional as F
class CSPDetectionDistribution(torch.distributions.Distribution):
    def __init__(self, distances: torch.tensor, csp_prob_parameters: torch.tensor, csp_distribution_parameters: torch.tensor, non_matching_distribution_parameters: torch.tensor):
        super().__init__()

        assert(distances.shape + (2,) == csp_prob_parameters.shape)
        assert(csp_distribution_parameters.shape == (2,)) #Gaussian parameters mean, std
        assert(non_matching_distribution_parameters.shape == (2,)) #Weibull distribution parameters scale, concentration

        self._distances = distances
        self._csp_probability_parameters = csp_prob_parameters # logits
        self._csp_distribution_parameters = csp_distribution_parameters
        self._csp_distribution = dist.Normal(self._csp_distribution_parameters[0],self._csp_distribution_parameters[1])
        self._non_matching_distribution_parameters = non_matching_distribution_parameters
        self._non_matching_distribution = dist.Weibull(concentration=self._non_matching_distribution_parameters[0],
                                                       scale=self._non_matching_distribution_parameters[1]) #location scale
        self._chi2_distribution = dist.torch.Chi2(1) #chi2 distribution for

        self.eps_float32 = torch.finfo(torch.float32).eps
        self.max_float32 = torch.finfo(torch.float32).max
        self.min_float32 = torch.finfo(torch.float32).min
        self._event_shape = torch.Size([self._distances.shape[0],3])

        self._loglikelihoodMatrix = torch.stack((self._chi2_distribution.log_prob(self._distances).clamp(min=self.min_float32),
                                                self._csp_distribution.log_prob(self._distances).clamp(min=self.min_float32),
                                                self._non_matching_distribution.log_prob(self._distances).clamp(min=self.min_float32)),dim=2)

        self._calculateBaseMatchingLikelihood()
        self._non_matching_totalloglikelihood = self._loglikelihoodMatrix[:,:,2].sum()

        self._event_shape = (self._distances.shape[0],)
        assert not self._loglikelihoodMatrix.isnan().any()
    #
    def _calculateBaseMatchingLikelihood(self):
       # nonCSPlikelihoods = self._loglikelihoodMatrix[:,:, 0]
       # CSPLikelihoods = self._loglikelihoodMatrix[:,:, 1]
        self._rowNoMatchesLikelihood = self._loglikelihoodMatrix[:,:,2].sum(0)
        self._columnNoMatchesLikelihood = self._loglikelihoodMatrix[:,:,2].sum(1)
        cspProbLogits = self._csp_probability_parameters
        #log_weights = torch.zeros(10,10,2)
        log_weights = cspProbLogits - cspProbLogits.logsumexp(dim=2, keepdim=True)
        self._baseMatchLikelihood = torch.logsumexp(self._loglikelihoodMatrix[:,:,:2] + log_weights, dim=2)

    def _makeDecision(self, logits: torch.tensor) -> torch.tensor:
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return idx, torch.exp(logits[idx] - logits.logsumexp(dim=0))


    def __getNextInSequence(self, sample: torch.tensor, availableParticles: torch.tensor, sample_weights: torch.tensor, sample_indicies: torch.tensor) -> torch.tensor:


        total_available = availableParticles.sum(-1)
        availableRows = availableParticles[...,:self._distances.shape[0]]
        availableColumns = availableParticles[...,self._distances.shape[0]:]

        availableRowsCount = availableRows.sum(-1)
        availableColumnsCount = availableColumns.sum(-1)

        completed_sample_mask = total_available == 0

        row_or_col_indexes = torch.zeros(sample.shape[0:-1],dtype=torch.int32)

        sampled_particles = torch.multinomial(availableParticles[~completed_sample_mask,:].type(torch.float32),1).type(torch.int32).squeeze(-1)

        row_or_col_indexes[~completed_sample_mask] = sampled_particles
        row_or_col_indexes[completed_sample_mask] = -1

        row_indexes_mask = (row_or_col_indexes < self._distances.shape[0]) & ~completed_sample_mask #Sampled indicies correspond to free rows
        col_indexes_mask = row_or_col_indexes >= self._distances.shape[0] #Sampled indicies correspond to free columns

        row_or_col_indexes[col_indexes_mask] -= self._distances.shape[0] #convert from total index to a column index

        noMatchMatrix = self._loglikelihoodMatrix[:, :, 2]

        #Handle Row Based decisions


        if row_indexes_mask.any():
            row_indexes = row_or_col_indexes[row_indexes_mask]
            row_matching_likelihoods = self._baseMatchLikelihood[row_indexes,:]*availableColumns[row_indexes_mask,:]
            noMatchDecisionLikelihoodMatrix = (noMatchMatrix[row_indexes,:]*availableColumns[row_indexes_mask,:])
            row_matching_nonMatching_likelihoods = (row_matching_likelihoods +
                                                noMatchDecisionLikelihoodMatrix.sum(-1,keepdim=True) -
                                                noMatchDecisionLikelihoodMatrix)
            decision_likelihoods = torch.cat((row_matching_nonMatching_likelihoods,noMatchDecisionLikelihoodMatrix.sum(-1,keepdim=True)),dim=1)
            matched_column_indexes = torch.multinomial((decision_likelihoods-decision_likelihoods.logsumexp(-1,keepdim=True)).exp(),1).type(torch.int32).squeeze(-1)

            proposal_probability = decision_likelihoods[torch.arange(decision_likelihoods.shape[0]),matched_column_indexes] - decision_likelihoods.logsumexp(dim=1) + torch.log(1.0/total_available[row_indexes_mask])
            sample_weights[row_indexes_mask] = (sample_weights[row_indexes_mask] + decision_likelihoods[torch.arange(decision_likelihoods.shape[0]),matched_column_indexes] +
                                            torch.log(1.0/(availableRowsCount[row_indexes_mask] * availableColumnsCount[row_indexes_mask] + availableColumnsCount[row_indexes_mask] + availableRowsCount[row_indexes_mask])) - proposal_probability)

            nomatched_column_mask = matched_column_indexes >= self._distances.shape[1]


            sample[row_indexes_mask, row_indexes] = matched_column_indexes
            sample[row_indexes_mask,row_indexes][nomatched_column_mask] = -1

            availableRows[row_indexes_mask, row_indexes] = False
            availableColumns[sample_indicies[row_indexes_mask][~nomatched_column_mask],matched_column_indexes[~nomatched_column_mask]] = False
        #
            #Handle Column Based Decision

        if col_indexes_mask.any():
            column_indexes = row_or_col_indexes[col_indexes_mask]
            column_matching_likelihoods = self._baseMatchLikelihood[:,column_indexes].transpose(0,1)*availableRows[col_indexes_mask,:]
            noMatchDecisionLikelihoodMatrix = (noMatchMatrix[:,column_indexes].transpose(0,1)*availableRows[col_indexes_mask,:])

            column_matching_nonMatching_likelihoods = (column_matching_likelihoods +
                                                   noMatchDecisionLikelihoodMatrix.sum(-1,keepdim=True) -
                                                   noMatchDecisionLikelihoodMatrix)
            decision_likelihoods = torch.cat((column_matching_nonMatching_likelihoods, noMatchDecisionLikelihoodMatrix.sum(-1,keepdim=True)), dim=1)
            matched_row_indexes = torch.multinomial((decision_likelihoods-decision_likelihoods.logsumexp(-1,keepdim=True)).exp(),1).type(torch.int32).squeeze(-1)

            proposal_probability = decision_likelihoods[torch.arange(decision_likelihoods.shape[0]),matched_row_indexes] - decision_likelihoods.logsumexp(dim=1) + torch.log(1.0/total_available[col_indexes_mask])

            sample_weights[col_indexes_mask] = (sample_weights[col_indexes_mask] + decision_likelihoods[torch.arange(decision_likelihoods.shape[0]),matched_row_indexes] +
                                            torch.log(1.0/(availableRowsCount[col_indexes_mask] *availableColumnsCount[col_indexes_mask] + availableRowsCount[col_indexes_mask] + availableColumnsCount[col_indexes_mask])) - proposal_probability)

            nomatched_row_mask = matched_row_indexes >= self._distances.shape[0]

            matched_row_indexes[nomatched_row_mask] = -1
            sample[sample_indicies[col_indexes_mask][~nomatched_row_mask], matched_row_indexes[~nomatched_row_mask]] = column_indexes[~nomatched_row_mask]

            availableColumns[col_indexes_mask,column_indexes] = False
            availableRows[sample_indicies[col_indexes_mask][~nomatched_row_mask],matched_row_indexes[~nomatched_row_mask]] = False
        #

    #
    def _sample(self, sample_shape=torch.Size()) -> torch.tensor:
        availableParticles = torch.ones(sample_shape+(self._distances.shape[0]+self._distances.shape[1],), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -1, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float32)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:,:-1])

        while availableParticles.any():
            self.__getNextInSequence(sample, availableParticles, sample_weights, sample_indexes)

        return sample

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        return self._sample(sample_shape)

    def log_prob(self, sample):
        matched_mask = sample != -1
        match_rows = torch.nonzero(matched_mask, as_tuple=False).squeeze()
        match_columns = sample[matched_mask]
        return (self._non_matching_totalloglikelihood -
                self._loglikelihoodMatrix[match_rows,match_columns,2].sum() +
                self._baseMatchLikelihood[match_rows,match_columns].sum())

    def csp_distribution_parameters(self):
        return self._csp_distribution_parameters

    def csp_probability_parameters(self):
        return self._csp_probability_parameters



#
