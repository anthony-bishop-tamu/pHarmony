import torch
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
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



        self._event_shape = (self._distances.shape[0],)
        assert not self._loglikelihoodMatrix.isnan().any()
    #

    def _calculateDecisionLogLikelihood(self):
        #parameter corrected loglikelihoods
        self._match_non_matching_loglikelihoods = self._loglikelihoodMatrix[:, :, 2]
        self._matching_likelihood = (self._loglikelihoodMatrix[:, :,0:2]+self._csp_probability_parameters).logsumexp(dim=2)

        self._base_row_decision_likelihoods= torch.zeros((self._distances.shape[0],self._distances.shape[1]+1),dtype=torch.float32)
        self._base_col_decision_likelihoods = torch.zeros((self._distances.shape[1], self._distances.shape[0] + 1),
                                                          dtype=torch.float32)
        self._base_row_decision_likelihoods[:,:] = self._match_non_matching_loglikelihoods.sum(dim=-1).unsqueeze(1)
        self._base_col_decision_likelihoods[:,:] = self._match_non_matching_loglikelihoods.transpose(0,1).sum(dim=-1).unsqueeze(1)
        self._base_row_decision_likelihoods[:,:-1] += self._matching_likelihood - self._match_non_matching_loglikelihoods
        self._base_col_decision_likelihoods[:,:-1] += self._matching_likelihood.transpose(0,1) - self._match_non_matching_loglikelihoods.transpose(0,1)


    def _makeDecision(self, logits: torch.tensor) -> torch.tensor:
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return idx, torch.exp(logits[idx] - logits.logsumexp(dim=0))

    def _calculateNextParticleNegativeEntropies(self, decision_likelihood_matrix: torch.tensor) -> torch.tensor:
        negInfMask = decision_likelihood_matrix == -1*torch.inf
        norm_likelihood_matrix = decision_likelihood_matrix - decision_likelihood_matrix.logsumexp(dim=-1,keepdim=True)
        norm_likelihood_matrix = norm_likelihood_matrix.exp()
        neg_entropies = norm_likelihood_matrix*decision_likelihood_matrix
        neg_entropies[norm_likelihood_matrix==0] = 0
        neg_entropies = neg_entropies.sum(dim=-1)/0.1
        neg_inf_rows = torch.all(negInfMask,dim=-1)
        neg_entropies[neg_inf_rows] = -1*torch.inf

        return neg_entropies
    #
    def __getNextInSequence(self, sample: torch.tensor, availableParticles: torch.tensor,
                            sample_weights: torch.tensor, sample_indicies: torch.tensor,
                            row_decision_matrix: torch.tensor, col_decision_matrix: torch.tensor,
                            neg_entropy_tensor: torch.tensor,partial_log_likelihood: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):


        total_available = availableParticles.sum(-1)
        availableRows = availableParticles[...,:self._distances.shape[0]]
        availableColumns = availableParticles[...,self._distances.shape[0]:]

        availableRowsCount = availableRows.sum(-1)
        availableColumnsCount = availableColumns.sum(-1)

        completed_sample_mask = total_available == 0

        row_or_col_indexes = torch.zeros(sample.shape[0:-1],dtype=torch.int32)

        neg_entropy_tensor[...,:self._distances.shape[0]] = self._calculateNextParticleNegativeEntropies(row_decision_matrix)
        neg_entropy_tensor[...,self._distances.shape[0]:] = self._calculateNextParticleNegativeEntropies(col_decision_matrix)

        prob_tensor = (neg_entropy_tensor - neg_entropy_tensor.logsumexp(dim=-1,keepdim=True)).exp()
        #prob_tensor = availableParticles
        sampled_particles = torch.multinomial(prob_tensor[~completed_sample_mask].type(torch.float32),1).type(torch.int32).squeeze(-1)

        row_or_col_indexes[~completed_sample_mask] = sampled_particles
        row_or_col_indexes[completed_sample_mask] = -1

        row_indexes_mask = (row_or_col_indexes < self._distances.shape[0]) & ~completed_sample_mask #Sampled indicies correspond to free rows
        col_indexes_mask = row_or_col_indexes >= self._distances.shape[0] #Sampled indicies correspond to free columns

        row_or_col_indexes[col_indexes_mask] -= self._distances.shape[0] #convert from total index to a column index


        #Handle Row Based decisions


        if row_indexes_mask.any():
            row_indexes = row_or_col_indexes[row_indexes_mask]

            matched_column_indexes = torch.multinomial((row_decision_matrix[row_indexes_mask,row_indexes,:]-row_decision_matrix[row_indexes_mask,row_indexes,:].logsumexp(dim=-1,keepdim=True)).exp(),1).type(torch.int32).squeeze(-1)
            proposal_probability = row_decision_matrix[row_indexes_mask,row_indexes,matched_column_indexes]- row_decision_matrix[row_indexes_mask,row_indexes].logsumexp(dim=-1)
            #assert (matched_column_indexes < self._distances.shape[1]).all()

            nomatched_column_mask = matched_column_indexes >= self._distances.shape[1]

            decision_log[sample_indicies[row_indexes_mask][~nomatched_column_mask],decision_counter,2] = (row_decision_matrix[sample_indicies[row_indexes_mask][~nomatched_column_mask], row_indexes[~nomatched_column_mask], matched_column_indexes[~nomatched_column_mask]] +
                col_decision_matrix[sample_indicies[row_indexes_mask][~nomatched_column_mask], matched_column_indexes[~nomatched_column_mask],row_indexes[~nomatched_column_mask]] -
                self._matching_likelihood[row_indexes[~nomatched_column_mask],matched_column_indexes[~nomatched_column_mask]])
            decision_log[sample_indicies[row_indexes_mask][nomatched_column_mask], decision_counter, 2] = row_decision_matrix[sample_indicies[row_indexes_mask][nomatched_column_mask], row_indexes[nomatched_column_mask], -1]

            partial_log_likelihood[sample_indicies[row_indexes_mask]] += decision_log[sample_indicies[row_indexes_mask],decision_counter,2]

            decision_log[sample_indicies[row_indexes_mask],decision_counter,3] = decision_log[sample_indicies[row_indexes_mask],decision_counter,2]
            decision_log[sample_indicies[row_indexes_mask],decision_counter,3] -= proposal_probability  + torch.log(prob_tensor[row_indexes_mask,row_indexes])

            sample_weights[row_indexes_mask] += decision_log[sample_indicies[row_indexes_mask],decision_counter,3]
            assert sample_weights.isfinite().all()


            row_decision_matrix[sample_indicies[row_indexes_mask][~nomatched_column_mask], :, :] \
                -= self._match_non_matching_loglikelihoods[:,matched_column_indexes[~nomatched_column_mask]].transpose(-1,-2).unsqueeze(-1)
            col_decision_matrix[row_indexes_mask, : , :] -= (
                self._match_non_matching_loglikelihoods[row_indexes,:].unsqueeze(-1)
            )
            row_decision_matrix[row_indexes_mask, row_indexes, :] = -1 * torch.inf
            row_decision_matrix[sample_indicies[row_indexes_mask][~nomatched_column_mask], :, matched_column_indexes[~nomatched_column_mask]] = -1 * torch.inf
            col_decision_matrix[sample_indicies[row_indexes_mask], : , row_indexes] = -1*torch.inf
            col_decision_matrix[sample_indicies[row_indexes_mask][~nomatched_column_mask], matched_column_indexes[~nomatched_column_mask] , :] = -1 * torch.inf

            matched_column_indexes[nomatched_column_mask] = -1
            sample[row_indexes_mask, row_indexes] = matched_column_indexes

            decision_log[sample_indicies[row_indexes_mask], decision_counter, 0] = row_indexes.type(torch.float32)
            decision_log[sample_indicies[row_indexes_mask],decision_counter, 1] = -1
            decision_log[sample_indicies[row_indexes_mask][~nomatched_column_mask], decision_counter, 1] = (matched_column_indexes[~nomatched_column_mask]+self._distances.shape[0]).type(torch.float32)

            #sample[sample_indicies[row_indexes_mask][nomatched_column_mask],row_indexes[nomatched_column_mask]] = -1

            assert(sample.max() < self._distances.shape[1])

            availableRows[row_indexes_mask, row_indexes] = False
            availableColumns[sample_indicies[row_indexes_mask][~nomatched_column_mask],matched_column_indexes[~nomatched_column_mask]] = False
            #assert (row_decision_matrix[:,:,:-1] >= row_decision_matrix[:,:,-1].unsqueeze(-1)).any(dim=-1).all()
            #assert (col_decision_matrix[:, :, :-1] >= col_decision_matrix[:, :, -1].unsqueeze(-1)).any(dim=-1).all()
        #
            #Handle Column Based Decision

        if col_indexes_mask.any():
            column_indexes = row_or_col_indexes[col_indexes_mask]

            matched_row_indexes = torch.multinomial((col_decision_matrix[col_indexes_mask,column_indexes,:]-col_decision_matrix[col_indexes_mask,column_indexes,:].logsumexp(dim=-1,keepdim=True)).exp(),1).type(torch.int32).squeeze(-1)
            #assert (matched_row_indexes < self._distances.shape[0]).all()
            proposal_probability = col_decision_matrix[col_indexes_mask,column_indexes,matched_row_indexes]-col_decision_matrix[col_indexes_mask,column_indexes,:].logsumexp(dim=-1)

            nomatched_row_mask = matched_row_indexes >= self._distances.shape[0]

            decision_log[sample_indicies[col_indexes_mask][~nomatched_row_mask], decision_counter, 2] =  (
                row_decision_matrix[sample_indicies[col_indexes_mask][~nomatched_row_mask], matched_row_indexes[~nomatched_row_mask], column_indexes[~nomatched_row_mask] ] +
                col_decision_matrix[sample_indicies[col_indexes_mask][~nomatched_row_mask], column_indexes[~nomatched_row_mask],matched_row_indexes[~nomatched_row_mask]] -
                self._matching_likelihood[matched_row_indexes[~nomatched_row_mask],column_indexes[~nomatched_row_mask]]
            )
            decision_log[sample_indicies[col_indexes_mask][nomatched_row_mask], decision_counter, 2] = col_decision_matrix[
                    sample_indicies[col_indexes_mask][nomatched_row_mask], column_indexes[nomatched_row_mask], -1]

            partial_log_likelihood[sample_indicies[col_indexes_mask]] += decision_log[sample_indicies[col_indexes_mask], decision_counter, 2]
            decision_log[sample_indicies[col_indexes_mask], decision_counter, 3] = decision_log[
                sample_indicies[col_indexes_mask], decision_counter, 2]

            decision_log[sample_indicies[col_indexes_mask], decision_counter, 3] -= proposal_probability  + torch.log(prob_tensor[col_indexes_mask,column_indexes+self._distances.shape[0]])

            sample_weights[col_indexes_mask] += decision_log[sample_indicies[col_indexes_mask], decision_counter, 3]

            assert sample_weights.isfinite().all()

            row_decision_matrix[sample_indicies[col_indexes_mask], :, :] -= (
                self._match_non_matching_loglikelihoods[:,column_indexes].transpose(-1,-2).unsqueeze(-1)
            )
            col_decision_matrix[sample_indicies[col_indexes_mask][~nomatched_row_mask], :, :] -= (
                self._match_non_matching_loglikelihoods[matched_row_indexes[~nomatched_row_mask],:].unsqueeze(-1)
            )
            col_decision_matrix[col_indexes_mask, column_indexes, :] = -1 * torch.inf
            col_decision_matrix[sample_indicies[col_indexes_mask][~nomatched_row_mask],:,matched_row_indexes[~nomatched_row_mask]] = -1 * torch.inf
            row_decision_matrix[sample_indicies[col_indexes_mask], :,column_indexes] = -1 * torch.inf
            row_decision_matrix[sample_indicies[col_indexes_mask][~nomatched_row_mask], matched_row_indexes[~nomatched_row_mask],:] = -1 * torch.inf

            matched_row_indexes[nomatched_row_mask] = -1
            sample[sample_indicies[col_indexes_mask][~nomatched_row_mask], matched_row_indexes[~nomatched_row_mask]] = column_indexes[~nomatched_row_mask]

            decision_log[sample_indicies[col_indexes_mask],decision_counter,0] = (column_indexes + self._distances.shape[0]).type(torch.float32)
            decision_log[sample_indicies[col_indexes_mask],decision_counter,1] = matched_row_indexes.type(torch.float32)

            assert (sample.max() < self._distances.shape[1])
            availableColumns[col_indexes_mask,column_indexes] = False
            availableRows[sample_indicies[col_indexes_mask][~nomatched_row_mask],matched_row_indexes[~nomatched_row_mask]] = False
            #assert (col_decision_matrix[:,:,:-1] >= col_decision_matrix[:,:,-1].unsqueeze(-1)).any(dim=-1).all()
            #assert (row_decision_matrix[:, :, :-1] >= row_decision_matrix[:, :, -1].unsqueeze(-1)).any(dim=-1).all()
        #
        return
    #
    def _resample(self, sample: torch.Tensor, sample_weights: torch.Tensor,
                  partial_log_likelihoods: torch.Tensor,
                  row_decision_matrix: torch.Tensor,
                  col_decision_matrix: torch.Tensor,
                  availableParticles: torch.Tensor,
                  neg_entropy_tensor: torch.Tensor,
                  decision_log):
        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        if ess < nsamples*0.5:
            # Step 1: Create systematic positions
            positions = (torch.arange(nsamples, dtype=sample_weights.dtype, device=sample_weights.device) +
                         torch.rand(1,dtype=sample_weights.dtype,device=sample_weights.device)) / nsamples

            # Step 2: Compute the cumulative sum of weights
            cumulative_sum = torch.cumsum(normalized_weights, dim=0)

            # Step 3: Use searchsorted to find where the systematic positions fall in the cumulative sum
            indices = torch.searchsorted(cumulative_sum, positions).clamp(min=0,max=nsamples-1)
            sample[...,:] = sample[indices,:]
            sample_weights[...] = 1.0/nsamples
            partial_log_likelihoods[...] =partial_log_likelihoods[indices]
            row_decision_matrix[...,:,:] =row_decision_matrix[indices,:,:]
            col_decision_matrix[...,:,:] =col_decision_matrix[indices,:,:]
            availableParticles[...,:] =availableParticles[indices,:]
            neg_entropy_tensor[...,:] =neg_entropy_tensor[indices,:]
            decision_log[...,...] =decision_log[indices,...]
        else:
            return
    #
    def _sample(self, sample_shape=torch.Size()) -> torch.tensor:
        availableParticles = torch.ones(sample_shape+(self._distances.shape[0]+self._distances.shape[1],), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float32)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:,:-1])
        self._calculateDecisionLogLikelihood()

        row_decision_matrix = torch.zeros(sample_shape+(self._distances.shape[0],self._distances.shape[1]+1), dtype=torch.float32)
        col_decision_matrix = torch.zeros(sample_shape + (self._distances.shape[1], self._distances.shape[0]+1),
                                          dtype=torch.float32)
        row_decision_matrix[...,:,:] = self._base_row_decision_likelihoods[:,:]
        col_decision_matrix[...,:,:] = self._base_col_decision_likelihoods[:,:]
        neg_entropy_tensor = torch.zeros_like(availableParticles,dtype=torch.float32)
        partial_log_likelihood = torch.zeros(sample_shape,dtype=torch.float32)
        decision_log = torch.full(sample_shape+(self._distances.shape[0]+self._distances.shape[1],4),-2, dtype=torch.float32)
        decision_counter = 0
        while availableParticles.any():
            self.__getNextInSequence(sample, availableParticles, sample_weights,
                                     sample_indexes, row_decision_matrix,col_decision_matrix,neg_entropy_tensor,partial_log_likelihood,decision_log,decision_counter)
            self._resample(sample, sample_weights,partial_log_likelihood,row_decision_matrix,col_decision_matrix,availableParticles,neg_entropy_tensor,decision_log)
            decision_counter += 1

        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample, sample_weights

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        sample, sample_weights = self._sample(sample_shape)



        return sample

    def log_prob(self, sample):
        self._calculateDecisionLogLikelihood()
        non_matching_totalloglikelihood = self._match_non_matching_loglikelihoods.sum()*np.prod(sample.shape[0:-1])
        matched_mask = sample != -1
        match_rows = torch.nonzero(matched_mask)
        match_columns = sample[matched_mask]
        return (non_matching_totalloglikelihood -
                self._loglikelihoodMatrix[match_rows[...,1],match_columns,2].sum()  +
                self._matching_likelihood[match_rows[...,1],match_columns].sum())

    def csp_distribution_parameters(self):
        return self._csp_distribution_parameters

    def csp_probability_parameters(self):
        return self._csp_probability_parameters



#
