import torch
import torch.distributions as torchdist
import numpy as np
from torch.profiler import record_function

class SamplingError(Exception):
    pass
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

    def beam_step(self,
                  beam_remaining_decision_mask: torch.tensor,
                  beam_decision_mask: torch.tensor,
                  beam_score: torch.tensor,
                  likelihood_matrix: torch.tensor,
                  beam_width: int):

            beam_indexes = torch.arange(beam_remaining_decision_mask.shape[0])
            beam_score_p_likelihood = beam_score.unsqueeze(-1).unsqueeze(-1) + likelihood_matrix.unsqueeze(0)
            beam_score_p_likelihood[~beam_remaining_decision_mask] = -torch.inf

            top_val,top_index = torch.topk(beam_score_p_likelihood.flatten(), beam_width)
            top_index = torch.unravel_index(top_index, beam_remaining_decision_mask.shape)

            beam_decision_mask_expanded = torch.zeros((beam_width,
                                              beam_decision_mask.shape[1]
                                              ,beam_decision_mask.shape[2]),dtype=torch.bool)
            beam_indexes_expanded = torch.arange(beam_decision_mask_expanded.shape[0])

            #Handle Decision mask
            beam_decision_mask_expanded[beam_indexes_expanded,...] = beam_decision_mask[top_index[0],...]
            beam_decision_mask_expanded[beam_indexes_expanded,top_index[1],top_index[2]] = True
            beam_decision_mask_expanded, inv, counts = torch.unique(beam_decision_mask_expanded, dim=0,
                                                                      return_inverse=True,return_counts=True)
            first_idx = torch.full((beam_decision_mask_expanded.size(0),),beam_width,dtype=torch.int64)
            first_idx.scatter_reduce_(0,inv,beam_indexes_expanded,reduce='amin')
            unique_scores = top_val[first_idx]

            #Handle Remaining Decision Mask
            beam_remaining_decision_mask_expanded = torch.zeros((beam_width,
                                                                 beam_remaining_decision_mask.shape[1],
                                                                 beam_remaining_decision_mask.shape[2]), dtype=torch.bool)


            beam_remaining_decision_mask_expanded[beam_indexes_expanded,...] = beam_remaining_decision_mask[top_index[0],...]
            beam_remaining_decision_mask_expanded[beam_indexes_expanded,top_index[1], :] = False
            valid_col_mask = top_index[2] < beam_remaining_decision_mask.shape[-1] - 1
            beam_remaining_decision_mask_expanded[beam_indexes_expanded[valid_col_mask],:,top_index[2][beam_indexes_expanded[valid_col_mask]]] = False

            beam_remaining_decision_mask_expanded = beam_remaining_decision_mask_expanded[first_idx,...]



            return beam_remaining_decision_mask_expanded,beam_decision_mask_expanded,unique_scores
    #

    def beam_search_decision(self, row: int, col: int,
                             decision_mask: torch.tensor,
                             likelihood_matrix: torch.tensor,
                             beam_width:int, beam_depth: int):
        assert decision_mask.dim() == 2
        assert likelihood_matrix.dim() == 2

        decision_mask=decision_mask.clone()
        decision_mask[row,:] = False
        if col < decision_mask.shape[1]-1:
            decision_mask[:,col] = False


        beam_remaining_decision_mask = decision_mask.clone().unsqueeze(0)
        beam_decision_mask = torch.zeros_like(beam_remaining_decision_mask)
        beam_score = torch.zeros((1,),dtype=torch.float64)
        for d in range(beam_depth):
            if ~beam_remaining_decision_mask.any():
                break
            beam_remaining_decision_mask,beam_decision_mask, beam_score = self.beam_step(beam_remaining_decision_mask,
                                                                     beam_decision_mask,
                                                                     beam_score,
                                                                     likelihood_matrix,
                                                                     beam_width)
        return beam_score.logsumexp(dim=-1)
    #
    def calculate_adjusted_likelihoods(self,
                             likelihood_matrix: torch.tensor,
                             available_rows: torch.tensor,
                             available_columns: torch.tensor,
                             k: int,
                             d: int,
                             b: int,
                             ):

        assert likelihood_matrix.dim() == 2
        assert available_rows.dim() == 1
        assert available_columns.dim() == 1
        adjustment = torch.zeros((k,),dtype=torch.float32)

        decision_mask = available_rows.unsqueeze(-1) & available_columns.unsqueeze(0)
        likelihood_matrix = torch.where(decision_mask,likelihood_matrix,-torch.inf)
        shape = likelihood_matrix.shape
        top_val, top_index = torch.topk(likelihood_matrix.flatten(), k)
        top_index_unraveled = torch.unravel_index(top_index,shape)
        for i in range(k):
            adjustment[i] = self.beam_search_decision(top_index_unraveled[0][i],top_index_unraveled[1][i],decision_mask,likelihood_matrix,b,d)

        indexes =  torch.stack([top_index_unraveled[0],top_index_unraveled[1]],dim=-1)
        adjusted_likelihoods = (adjustment + likelihood_matrix.flatten()[top_index])
        adjusted_likelihoods -= adjusted_likelihoods.logsumexp(dim=-1,keepdim=True)
        return adjusted_likelihoods, indexes



    def __getNextInSequence(self, sample: torch.tensor, sample_indicies: torch.tensor,
                            availableRows: torch.tensor,
                            availableCols: torch.tensor,
                            row_log_evidence: torch.tensor,
                            sample_weights: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):


        k=10
        d=1
        b=1
        adjusted_likelihoods = torch.zeros((sample.shape[0],k),dtype=torch.float32)
        decision_indexes = torch.zeros((sample_indicies.shape[0],k,2),dtype=torch.int32)
        for i in range(sample_indicies.shape[0]):
            adjustment, index = self.calculate_adjusted_likelihoods(
                self._base_row_decision_likelihoods_unnormalized,
                availableRows[i,:],
                availableCols[i,:],
                k,d,b)
            adjusted_likelihoods[i,:] = adjustment
            decision_indexes[i,...] = index

        decision_probabilities = (adjusted_likelihoods - adjusted_likelihoods.logsumexp(dim=-1,keepdim=True)).exp()
        sampled_decision_indexes = torch.multinomial(decision_probabilities,num_samples=1,replacement=True).type(torch.int32).squeeze(-1)
        sampled_rows = decision_indexes[sample_indicies,sampled_decision_indexes,0]
        matched_columns = decision_indexes[sample_indicies, sampled_decision_indexes, 1]


        no_matched_columns = matched_columns >= self._distances.shape[1]

        #availableRows[sample_indicies, sampled_rows] = False

        decision_log[sample_indicies,decision_counter,0] = sampled_rows.type(torch.float64)
        decision_log[sample_indicies, decision_counter, 2] = self._base_row_decision_likelihoods_unnormalized[sampled_rows,matched_columns] #+row_probabilities
        decision_log[sample_indicies,decision_counter, 3] = decision_probabilities[sample_indicies, sampled_decision_indexes].type(torch.float64).log()

        sample_weights += decision_log[sample_indicies,decision_counter,2] - decision_log[sample_indicies,decision_counter,3]
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
        #print(f"ESS ratio:, {ess/sample_weights.shape[0]:0.3f}")
        if ess < nsamples*0.5 or force_resample:
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
            try:
                self.__getNextInSequence(sample, sample_indexes, availableRows, availableCols, row_log_evidence, sample_weights,decision_log,decision_counter)
                self._resample(sample, sample_weights, row_log_evidence, availableRows,availableCols,decision_log)
            except Exception as e:
                raise SamplingError(f"Error during sampling of matching matrices \n"
                                    f"Step: {decision_counter} of {availableRows.shape[0]} \n"
                                    f"CSP Distribution Parameters: {self.csp_distribution.param} \n"
                                    f"CSP_weight logits: {self._csp_mixture_weights} probits: {(self._csp_mixture_weights-self._csp_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"matching_weight_logits: {self._matching_mixture_weights} probits: {(self._matching_mixture_weights - self._matching_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"missing_weight_logits: {self._missing_mixture_weights} probits: {(self._missing_mixture_weights - self._missing_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e

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
