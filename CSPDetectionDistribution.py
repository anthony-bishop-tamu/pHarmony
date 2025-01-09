import torch
import pyro.distributions as dist
import torch.distributions as torchdist
import torch.nn.functional as F
from Frechet import Frechet, KDEDensity, LogTransformedKDEDensity
import numpy as np
def logisticDistribution(loc,scale):
    base_distribution = torchdist.Uniform(0, 1)
    transforms = [torchdist.transforms.SigmoidTransform().inv, torchdist.transforms.AffineTransform(loc=loc, scale=scale)]
    logistic = torchdist.transformed_distribution.TransformedDistribution(base_distribution, transforms)
    return logistic
#
class CSPDetectionDistribution(torch.distributions.Distribution):
    def __init__(self, distances: torch.tensor,
                 csp_mixture_weights: torch.tensor,
                 matching_mixture_weights: torch.tensor,
                 csp_distribution_parameters: torch.tensor,
                 non_matching_parameters: torch.tensor):
        super().__init__()
        assert(distances.shape[0] >= distances.shape[1])
        assert(csp_distribution_parameters.shape == (2,)) #Gaussian parameters mean, std
        assert((2,) == csp_mixture_weights.shape)
        assert ((2,) == matching_mixture_weights.shape)
        assert((2,) == non_matching_parameters.shape)

        self._distances = distances

        self._csp_mixture_weights = csp_mixture_weights - csp_mixture_weights.logsumexp(dim=0,keepdim=True)
        self._matching_mixture_weights = matching_mixture_weights - matching_mixture_weights.logsumexp(dim=0, keepdim=True)
        self._csp_distribution_parameters = csp_distribution_parameters
        self._csp_distribution = Frechet(alpha=self._csp_distribution_parameters[0],
                                         scale=self._csp_distribution_parameters[1])
        self._non_matching_parameters = non_matching_parameters
       # self._non_matching_parameters =
        self._non_matching_distribution = dist.Uniform(non_matching_parameters[0],non_matching_parameters[1])



        self._no_csp_distribution = torch.distributions.Chi2(torch.tensor([2.0],dtype=torch.float64)) #chi2 distribution for

        self.eps_float64 = torch.finfo(torch.float64).eps
        self.max_float64 = torch.finfo(torch.float64).max
        self.min_float64 = torch.finfo(torch.float64).min
        self._event_shape = torch.Size([self._distances.shape[0],3])

        self._loglikelihoodMatrix = torch.stack((self._no_csp_distribution.log_prob(self._distances).clamp(min=self.min_float64),
                                                self._csp_distribution.log_prob(self._distances).clamp(min=self.min_float64),
                                                self._non_matching_distribution.log_prob(self._distances).clamp(min=self.min_float64)),dim=2)
        self._event_shape = (self._distances.shape[0],)

        assert not self._loglikelihoodMatrix.isnan().any()
    #

    def _calculateDecisionLogLikelihood(self):
        unnormalized_csp_posterior_probabilities = self._loglikelihoodMatrix[:,:,0:2].detach() + self._csp_mixture_weights.detach()
        self._csp_posterior_probabilities = unnormalized_csp_posterior_probabilities - unnormalized_csp_posterior_probabilities.logsumexp(dim=-1,keepdim=True)

        unweighted_matching_loglikelihoods = (self._loglikelihoodMatrix[:,:,0:2] + self._csp_mixture_weights.detach()).logsumexp(dim=-1)
        #parameter corrected loglikelihoods

        final_matching_likelihoods = (torch.stack([unweighted_matching_loglikelihoods,self._loglikelihoodMatrix[:,:,2]],dim=2)
                                                  + self._matching_mixture_weights.detach())

        self._matching_likelihood = final_matching_likelihoods[:,:,0]
        self._match_non_matching_loglikelihoods = final_matching_likelihoods[:,:,1]

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
        prob_tensor = availableRows.type(torch.float64)
        sampled_rows = torch.multinomial(prob_tensor.type(torch.float64),1).type(torch.int32).squeeze(-1)
        probabilities = row_decision_matrix[sample_indicies,sampled_rows,:] - row_decision_matrix[sample_indicies,sampled_rows,:].logsumexp(dim=-1,keepdim=True)
        matched_columns = torch.multinomial(probabilities.exp(),1).type(torch.int32).squeeze()

        no_matched_columns = matched_columns >= self._distances.shape[1]

        availableRows[sample_indicies, sampled_rows] = False

        row_probabilities = torch.log(prob_tensor[sample_indicies, sampled_rows]/prob_tensor[sample_indicies].sum(dim=-1))
        decision_log[sample_indicies,decision_counter,0] = sampled_rows.type(torch.float64)
        decision_log[sample_indicies, decision_counter, 2] = row_decision_matrix[sample_indicies, sampled_rows, matched_columns].type(torch.float64) #+row_probabilities


        decision_log[sample_indicies,decision_counter, 3] += row_probabilities


        #sample_weights += decision_log[sample_indicies,decision_counter,2]-decision_log[:, decision_counter, 3]
        sample_weights += row_probabilities



        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies,decision_counter,1] = matched_columns.type(torch.float64)

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
                  decision_log,
                  force_resample=False):
        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        if ess < nsamples*0.5 or force_resample:
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
        sample_weights = torch.zeros(sample_shape, dtype=torch.float64)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionLogLikelihood()

        row_decision_matrix = torch.zeros(sample_shape+(self._distances.shape[0],self._distances.shape[1]+1), dtype=torch.float64)
        row_decision_matrix[...,:,:] = self._base_row_decision_likelihoods[:,:]
        neg_entropy_tensor = torch.zeros_like(availableRows,dtype=torch.float64)
        partial_log_likelihood = torch.zeros(sample_shape,dtype=torch.float64)
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float64)
        decision_counter = 0
        while availableRows.any():
            self.__getNextInSequence(sample, sample_indexes, availableRows, sample_weights, row_decision_matrix,neg_entropy_tensor,partial_log_likelihood,decision_log,decision_counter)
            self._resample(sample, sample_weights,row_decision_matrix,partial_log_likelihood,availableRows,neg_entropy_tensor,decision_log)
            decision_counter += 1
        self._resample(sample,sample_weights,row_decision_matrix,partial_log_likelihood,availableRows,neg_entropy_tensor,decision_log,True)
        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        sample = self._sample(sample_shape)



        return sample

    def log_prob(self, sample):
        #sample_weights should be logits
        self._calculateDecisionLogLikelihood()
        matched_mask = sample != -1
        match_rows = torch.nonzero(matched_mask)
        match_columns = sample[matched_mask]
        nomatch_rows = torch.nonzero(~matched_mask)
        nomatch_columns = sample[~matched_mask]
        log_prob = self._match_non_matching_loglikelihoods[nomatch_rows[...,1],nomatch_columns].sum() + self._matching_likelihood[match_rows[...,1],match_columns].sum()


        return log_prob.sum()/sample.numel()

    @property
    def csp_distribution_parameters(self) -> torch.Tensor:
        return self._csp_distribution_parameters

    @property
    def matching_posterior_probabilities(self) -> torch.Tensor:
        return self._matching_posterior_probabilities
    @property
    def csp_posterior_probabilities(self) -> torch.Tensor:
        return self._csp_posterior_probabilities
    @property
    def non_matching_distribution_parameters(self) -> torch.Tensor:
        return self._non_matching_distribution_parameters
    @property
    def csp_distribution(self) -> torch.distributions.Distribution:
        return self._csp_distribution

    @property
    def no_csp_distribution(self) -> torch.distributions.Distribution:
        return self._no_csp_distribution

    @property
    def non_matching_distribution(self) -> torch.distributions.Distribution:
        return self._non_matching_distribution

    @property
    def distances(self) -> torch.Tensor:
        return self._distances
    @property
    def mixture_weights(self) -> torch.Tensor:
        return self._mixture_weights




#
