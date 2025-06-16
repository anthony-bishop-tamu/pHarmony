import torch
import pyro.distributions as dist
import torch.distributions as torchdist
import torch.nn.functional as F
from Frechet import Frechet, KDEDensity, LogTransformedKDEDensity, RadialChi2
import numpy as np
from torch.profiler import record_function
def logisticDistribution(loc,scale):
    base_distribution = torchdist.Uniform(0, 1)
    transforms = [torchdist.transforms.SigmoidTransform().inv, torchdist.transforms.AffineTransform(loc=loc, scale=scale)]
    logistic = torchdist.transformed_distribution.TransformedDistribution(base_distribution, transforms)
    return logistic
#
class CSPDetectionDistribution(torch.distributions.Distribution):
    arg_constraints = {}
    def __init__(self, distances: torch.tensor,
                 csp_mixture_weights: torch.tensor,
                 matching_mixture_weights: torch.tensor,
                 missing_mixture_weights: torch.tensor,
                 csp_distribution: torch.distributions.Distribution,
                 non_matching_distribution: torch.distributions.Distribution):
        super().__init__()
        #assert(distances.shape[0] >= distances.shape[1])
        assert((2,) == csp_mixture_weights.shape)
        assert ((2,) == matching_mixture_weights.shape)

        self._distances = distances

        self._csp_mixture_weights = csp_mixture_weights - csp_mixture_weights.logsumexp(dim=0,keepdim=True)
        self._matching_mixture_weights = matching_mixture_weights - matching_mixture_weights.logsumexp(dim=0, keepdim=True)
        self._missing_mixture_weights = missing_mixture_weights - missing_mixture_weights.logsumexp(dim=0, keepdim=True)
        self._csp_distribution = csp_distribution
       # self._non_matching_parameters =
        self._non_matching_distribution = non_matching_distribution

        self._no_csp_distribution = torch.distributions.Chi2(torch.tensor([2.0],dtype=torch.float64)) #chi2 distribution for

        self.eps_float64 = torch.finfo(torch.float64).eps
        self.max_float64 = torch.finfo(torch.float64).max
        self.min_float64 = torch.finfo(torch.float64).min
        self._event_shape = torch.Size([self._distances.shape[0],3])

        self._loglikelihoodMatrix = torch.stack((self._no_csp_distribution.log_prob(self._distances).clamp(min=self.min_float64),
                                                self._csp_distribution.log_prob(self._distances).clamp(min=self.min_float64),
                                                self._non_matching_distribution.log_prob(self._distances).clamp(min=self.min_float64)),dim=2)
        self._event_shape = (self._distances.shape[0],)
        if self._loglikelihoodMatrix[:,:,1].isnan().any():
            print(f"Error with CSP dist evaluation: parameters are alpha,scale {csp_distribution.alpha} {csp_distribution.scale}")

        assert not self._loglikelihoodMatrix[:,:,0].isnan().any()
        assert not self._loglikelihoodMatrix[:, :,  1].isnan().any()
        assert not self._loglikelihoodMatrix[:, :, 2].isnan().any()
        self._calculateDecisionLogLikelihood()
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

        distributed_missing_mixture_weights = torch.zeros((self._distances.shape[1]+1,))
        distributed_missing_mixture_weights[:-1] = self._missing_mixture_weights[0].unsqueeze(-1).detach()
        distributed_missing_mixture_weights[-1] = self._missing_mixture_weights.detach()[1]

        self._base_row_decision_likelihoods= torch.zeros((self._distances.shape[0],self._distances.shape[1]+1),dtype=torch.float64)
        self._base_row_decision_likelihoods[:,:] = self._match_non_matching_loglikelihoods.detach().sum(dim=-1).unsqueeze(1)
        self._base_row_decision_likelihoods[:,:-1] += self._matching_likelihood.detach() - self._match_non_matching_loglikelihoods.detach()
        self._base_row_decision_likelihoods += distributed_missing_mixture_weights.unsqueeze(0).detach()
        self._base_row_decision_likelihoods_unnormalized = self._base_row_decision_likelihoods.clone()
        self._base_row_decision_likelihoods -= self._base_row_decision_likelihoods.logsumexp(dim=-1,keepdim=True)

        with record_function("decision_exponentiation"):
            self._base_row_decision_probabilities = self._base_row_decision_likelihoods.exp()




    def __getNextInSequence(self, sample: torch.tensor, sample_indicies: torch.tensor,
                            availableRows: torch.tensor,
                            availableCols: torch.tensor,
                            row_log_evidence: torch.tensor,
                            sample_weights: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):


        row_decision_matrix = self._base_row_decision_probabilities.detach()



        row_probs = (row_log_evidence-row_log_evidence.logsumexp(dim=-1,keepdim=True)).exp()

        #prob_tensor = availableRows.type(torch.float64)
        #row_index_list = torch.nonzero(availableRows.type(torch.float64), as_tuple=True)[1].reshape(sample.shape[0],-1)
        with record_function("Row_Sampling"):
            sampled_rows = torch.multinomial(row_probs*availableRows,num_samples=1,replacement=True).type(torch.int32).squeeze(1)
            #print(sampled_rows.unique().shape)
            #sampled_rows = row_index_list[torch.arange(sample.shape[0]),sampled_rows]

        with record_function("Column_Sampling"):
            probabilities = row_decision_matrix[sampled_rows,:]*availableCols
            matched_columns = torch.multinomial(probabilities,1,replacement=True).type(torch.int32).squeeze()

        no_matched_columns = matched_columns >= self._distances.shape[1]

        #availableRows[sample_indicies, sampled_rows] = False

        decision_log[sample_indicies,decision_counter,0] = sampled_rows.type(torch.float64)
        decision_log[sample_indicies, decision_counter, 2] = row_decision_matrix[sampled_rows, matched_columns].type(torch.float64) #+row_probabilities
        decision_log[sample_indicies,decision_counter, 3] += probabilities[sample_indicies, matched_columns].type(torch.float64)

        sample_weights += self._base_row_decision_likelihoods_unnormalized[sampled_rows,matched_columns] - (decision_log[:, decision_counter, 3] + row_probs[sample_indicies,sampled_rows])
        assert sample_weights.isfinite().all()


        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies,decision_counter,1] = matched_columns.type(torch.float64)

        sample[sample_indicies,sampled_rows]=matched_columns
        availableRows[sample_indicies, sampled_rows] = False
        availableCols[sample_indicies[~no_matched_columns], matched_columns[~no_matched_columns]] = False

        logits_to_remove = self._base_row_decision_likelihoods_unnormalized[:,matched_columns].transpose(-1,-2)
        delta = row_log_evidence-logits_to_remove
        delta [ delta < 0 ] = 0
        new = logits_to_remove + (torch.expm1(delta)).log()
        new[sample_indicies,sampled_rows] = -torch.inf
        assert not new.isnan().any()
        row_log_evidence[sample_indicies[~no_matched_columns],...] = new[sample_indicies[~no_matched_columns],...]


#
    def _stratified_resample(self,
            sample: torch.Tensor,
            sample_weights: torch.Tensor,
            availableRows: torch.Tensor,
            availableCols: torch.Tensor,
            decision_log,
            force_resample=False):
        # Step 0: Convert log-weights to normalized linear weights
        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0 / torch.pow(normalized_weights, 2).sum()
        nsamples = sample.shape[0]

        if ess < nsamples * 0.1 or force_resample:
            # Step 1: Stratified uniform samples
            # Shape: (nsamples,)
            u = (torch.arange(nsamples, dtype=sample_weights.dtype, device=sample_weights.device) +
                 torch.rand(nsamples, dtype=sample_weights.dtype, device=sample_weights.device)) / nsamples

            # Step 2: CDF of normalized weights
            cumulative_sum = torch.cumsum(normalized_weights, dim=0)

            # Step 3: Map uniform samples to particle indices
            indices = torch.searchsorted(cumulative_sum, u).clamp(min=0, max=nsamples - 1)

            # Step 4: Resample all state tensors (ensure shapes are compatible)
            sample.copy_(sample[indices])
            availableRows.copy_(availableRows[indices])
            availableCols.copy_(availableCols[indices])
            decision_log.copy_(decision_log[indices])

            # Step 5: Reset weights (any equal log-weight is fine)
            sample_weights.fill_(0.0)  # log(1.0)
    def _resample(self, sample: torch.Tensor,
                  sample_weights: torch.Tensor,
                  row_log_evidence: torch.Tensor,
                  availableRows: torch.Tensor,
                  availableCols: torch.Tensor,
                  decision_log,
                  force_resample=False):

        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        if ess < nsamples*0.1 or force_resample:
            # Step 1: Create systematic positions
            positions = (torch.arange(nsamples, dtype=sample_weights.dtype, device=sample_weights.device) +
                         torch.rand(1,dtype=sample_weights.dtype,device=sample_weights.device)) / nsamples

            # Step 2: Compute the cumulative sum of weights
            cumulative_sum = torch.cumsum(normalized_weights, dim=0)

            # Step 3: Use searchsorted to find where the systematic positions fall in the cumulative sum
            indices = torch.searchsorted(cumulative_sum, positions).clamp(min=0,max=nsamples-1)
            sample[...,:] = sample[indices,:]
            sample_weights[...] = 0
            availableRows[...,:] =availableRows[indices,:]
            availableCols[...,:] =availableCols[indices,:]
            row_log_evidence[...,:] =row_log_evidence[indices,:]
            decision_log[...,...] =decision_log[indices,...]
        else:
            return
    #
    def _sample(self, sample_shape=torch.Size()) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._distances.shape[0],), dtype=torch.bool)
        availableCols = torch.ones(sample_shape+(self._distances.shape[1]+1,), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float64)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionLogLikelihood()
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float64)
        decision_counter = 0
        row_log_evidence = self._base_row_decision_likelihoods_unnormalized.logsumexp(dim=-1)
        row_log_evidence = torch.where(availableRows, row_log_evidence.unsqueeze(0), -torch.inf)
        while availableRows.any():
            self.__getNextInSequence(sample, sample_indexes, availableRows, availableCols, row_log_evidence, sample_weights,decision_log,decision_counter)
            self._resample(sample, sample_weights, row_log_evidence, availableRows,availableCols,decision_log)
            decision_counter += 1
        self._resample(sample, sample_weights, row_log_evidence, availableRows,availableCols,decision_log,True)
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
    def csp_posterior_probabilities(self) -> torch.Tensor:
        return self._csp_posterior_probabilities
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
    def csp_mixture_weights(self) -> torch.Tensor:
        return self._csp_mixture_weights

    @property
    def matching_mixture_weights(self) -> torch.Tensor:
        return self._matching_mixture_weights

    @property
    def missing_mixture_weights(self) -> torch.Tensor:
        return self._missing_mixture_weights




#
