import torch
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
class CSPDetectionDistribution(torch.distributions.Distribution):
    def __init__(self, distances: torch.tensor, csp_prob_parameters: torch.tensor, csp_distribution_parameters: torch.tensor, non_matching_distribution_parameters: torch.tensor):
        super().__init__()
        assert(distances.shape[0] >= distances.shape[1])
        assert(distances.shape + (2,) == csp_prob_parameters.shape)
        assert(csp_distribution_parameters.shape == (2,)) #Gaussian parameters mean, std
        assert(non_matching_distribution_parameters.shape == (2,)) #Weibull distribution parameters scale, concentration

        self._distances = distances
        self._csp_probability_parameters = csp_prob_parameters # logits
        self._csp_distribution_parameters = csp_distribution_parameters
        self._csp_distribution = dist.Weibull(concentration=self._csp_distribution_parameters[0],scale=self._csp_distribution_parameters[1])
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
        self._base_row_decision_likelihoods[:,:] = self._match_non_matching_loglikelihoods.detach().sum(dim=-1).unsqueeze(1)
        self._base_row_decision_likelihoods[:,:-1] += self._matching_likelihood.detach() - self._match_non_matching_loglikelihoods.detach()


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
        neg_entropies = neg_entropies.sum(dim=-1)
        neg_inf_rows = torch.all(negInfMask,dim=-1)
        neg_entropies[neg_inf_rows] = -1*torch.inf

        return neg_entropies
    #
    def __getNextInSequence(self, sample: torch.tensor, sample_indicies: torch.tensor,availableRows: torch.tensor,
                            sample_weights: torch.tensor,
                            row_decision_matrix: torch.tensor,
                            neg_entropy_tensor: torch.tensor,
                            partial_log_likelihood: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):


        #neg_entropy_tensor[...,:] = self._calculateNextParticleNegativeEntropies(row_decision_matrix)

        #prob_tensor = (neg_entropy_tensor - neg_entropy_tensor.logsumexp(dim=-1,keepdim=True)).exp()
        prob_tensor = availableRows.type(torch.float32)
        sampled_rows = torch.multinomial(prob_tensor.type(torch.float32),1).type(torch.int32).squeeze(-1)
        probabilities = row_decision_matrix[sample_indicies,sampled_rows,:] - row_decision_matrix[sample_indicies,sampled_rows,:].logsumexp(dim=-1,keepdim=True)
        matched_columns = torch.multinomial(probabilities.exp(),1).type(torch.int32).squeeze()

        no_matched_columns = matched_columns >= self._distances.shape[1]

        availableRows[sample_indicies,sampled_rows] = False

        decision_log[sample_indicies,decision_counter,0] = sampled_rows.type(torch.float32)
        decision_log[sample_indicies, decision_counter, 2] = row_decision_matrix[sample_indicies, sampled_rows, matched_columns].type(torch.float32)
        partial_log_likelihood += row_decision_matrix[sample_indicies, sampled_rows, matched_columns]
        decision_log[sample_indicies,decision_counter, 3] = -prob_tensor[sample_indicies, sampled_rows]

        sample_weights += decision_log[:, decision_counter, 3]

        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies,decision_counter,1] = matched_columns.type(torch.float32)

        sample[sample_indicies,sampled_rows]=matched_columns

        row_decision_matrix[sample_indicies,sampled_rows,:] = -1*torch.inf
        row_decision_matrix[~no_matched_columns,:,matched_columns[~no_matched_columns]] = -1*torch.inf
        row_decision_matrix[~no_matched_columns,:,:] -= self._match_non_matching_loglikelihoods[:,matched_columns[~no_matched_columns]].transpose(0,1).unsqueeze(-1).detach()

    #
    def _resample(self, sample: torch.Tensor, sample_weights: torch.Tensor,
                  row_decision_matrix: torch.tensor,
                  partial_log_likelihoods: torch.Tensor,
                  availableRows: torch.Tensor,
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
            availableRows[...,:] =availableRows[indices,:]
            neg_entropy_tensor[...,:] =neg_entropy_tensor[indices,:]
            decision_log[...,...] =decision_log[indices,...]
        else:
            return
    #
    def _sample(self, sample_shape=torch.Size()) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._distances.shape[0],), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float32)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionLogLikelihood()

        row_decision_matrix = torch.zeros(sample_shape+(self._distances.shape[0],self._distances.shape[1]+1), dtype=torch.float32)
        row_decision_matrix[...,:,:] = self._base_row_decision_likelihoods[:,:]
        neg_entropy_tensor = torch.zeros_like(availableRows,dtype=torch.float32)
        partial_log_likelihood = torch.zeros(sample_shape,dtype=torch.float32)
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float32)
        decision_counter = 0
        while availableRows.any():
            self.__getNextInSequence(sample, sample_indexes, availableRows, sample_weights, row_decision_matrix,neg_entropy_tensor,partial_log_likelihood,decision_log,decision_counter)
            self._resample(sample, sample_weights,row_decision_matrix,partial_log_likelihood,availableRows,neg_entropy_tensor,decision_log)
            decision_counter += 1

        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample, sample_weights

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        sample, sample_weights = self._sample(sample_shape)



        return sample, sample_weights

    def log_prob(self, sample, sample_weights):
        #sample_weights should be logits
        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        self._calculateDecisionLogLikelihood()
        non_matching_totalloglikelihood = self._match_non_matching_loglikelihoods.sum()
        matched_mask = sample != -1
        match_rows = torch.nonzero(matched_mask)
        match_columns = sample[matched_mask]
        log_prob = (non_matching_totalloglikelihood -
                self._loglikelihoodMatrix[match_rows[...,1],match_columns,2].sum()*normalized_weights +
                self._matching_likelihood[match_rows[...,1],match_columns].sum()*normalized_weights)


        return log_prob

    def csp_distribution_parameters(self):
        return self._csp_distribution_parameters

    def csp_probability_parameters(self):
        return self._csp_probability_parameters



#
