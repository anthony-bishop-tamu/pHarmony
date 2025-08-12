import logging

import torch
import torch.distributions as torchdist
import numpy as np
from torch.profiler import record_function
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
import math
class SamplingError(Exception):
    pass
class EnumerationError(Exception):
    pass
from tqdm import tqdm

def sample_gumbel(shape, device=None, dtype=None, generator=None):
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return -torch.log(-torch.log(u))
def gumbel_max(logits, mask=None, generator=None):
    # logits: [..., C]
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator)
    return (logits + g).argmax(dim=-1)  # indices with categorical(softmax(logits)) law

def gumbel_topk(logits, k, mask=None, generator=None):
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator)
    return (logits + g).topk(k, dim=-1).indices

def validateSample(sample: torch.tensor, availableCols: torch.tensor):
    for i in range(sample.shape[0]):
        unique, counts = torch.unique(sample[i], return_counts=True)
        assert (sample[i] >= -1).all()
        cols = sample[i][sample[i] > -1]
        assert not availableCols[i,cols].any()
        not_used = torch.nonzero(availableCols[i,:])
        assert not_used.tolist() not in sample[i].tolist()
        assert ~((unique[counts > 1] > -1).any())

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
        self._autoregression = None
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
        self._base_row_decision_probabilities = (self._base_row_decision_likelihoods_unnormalized - self._base_row_decision_likelihoods_unnormalized.logsumexp(dim=-1,keepdim=True)).exp()

    def __getNextInSequence(self,
                                             sample: torch.tensor,
                                             sample_indicies: torch.tensor,
                                             row_order: torch.tensor,
                                             availableCols: torch.tensor,
                                             decision_log: torch.tensor,
                                             decision_counter: int):

        current_row_index = row_order[decision_counter]

        logit_corrections = self.row_beam_search(availableCols,
                                                      self._base_row_decision_likelihoods_unnormalized,
                                                      row_order,
                                                      decision_counter,
                                                      beam_width=100,
                                                      max_depth=20)
        log_probabilities = self._base_row_decision_likelihoods_unnormalized[current_row_index,:].unsqueeze(0) + logit_corrections
        log_probabilities.masked_fill_(~availableCols,-torch.inf)
        log_probabilities -= log_probabilities.logsumexp(dim=-1,keepdim=True)


        with record_function("Column_Sampling"):
            matched_columns = torch.multinomial(log_probabilities.exp(), 1, replacement=True).type(torch.int32).squeeze()
            #matched_columns = topk_cols[unmapped_matched_columns].type(torch.int32)
        no_matched_columns = matched_columns >= self._distances.shape[1]

        # availableRows[sample_indicies, sampled_rows] = False

        decision_log[sample_indicies, decision_counter, 0] = row_order[decision_counter].type(torch.float64)
        decision_log[sample_indicies, decision_counter, 2] = log_probabilities[sample_indicies, matched_columns].type(torch.float64)  # +row_probabilities
        decision_log[sample_indicies, decision_counter, 3] = self._base_row_decision_likelihoods_unnormalized[current_row_index,matched_columns].type(torch.float64)
        sample_weights = decision_log[sample_indicies, decision_counter, 3] - decision_log[sample_indicies, decision_counter, 2]
        assert sample_weights.isfinite().all()

        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies, decision_counter, 1] = matched_columns.type(torch.float64)

        sample[sample_indicies, current_row_index] = matched_columns
        availableCols[sample_indicies[~no_matched_columns], matched_columns[~no_matched_columns]] = False

        return sample_weights

    def _resample(self, sample: torch.Tensor,
                  sample_weights: torch.Tensor,
                  availableRows: torch.Tensor,
                  availableCols: torch.Tensor,
                  decision_log,
                  decision_counter: int,
                  force_resample=False):

        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        ess_ratio = ess/sample_weights.shape[0]
        assert ess.isfinite().all()
        if ess_ratio < 0.5 or force_resample:
            print(f"{decision_counter}: ESS ratio:, {ess_ratio:0.3f}; Resample")
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
            decision_log[...,...] =decision_log[indices,...]
            return ess/nsamples < 0.25
        else:
            return False
    #
    def deduplicate_masks(self,availableCols: torch.Tensor):
        unique_masks, inverse = torch.unique(availableCols, dim=0, return_inverse=True)
        return unique_masks, inverse

    @torch.no_grad()
    def row_beam_search(self,
                        availableCols: torch.Tensor,
                        log_likelihoods: torch.Tensor,
                        ordered_rows: torch.Tensor,
                        current_row: int,
                        beam_width: int = 10,
                        max_depth: int = 1000):

        unique_availableCols, reverse_index = torch.unique(availableCols, dim=0, return_inverse=True)
        n_cols = unique_availableCols.shape[1]
        n_unique_samples = unique_availableCols.shape[0]
        future_availableCols = unique_availableCols.clone().unsqueeze(1).expand(-1,n_cols,-1).clone()
        future_logsumexps = torch.zeros((n_unique_samples,n_cols,beam_width),dtype=torch.float32)

        #Build indexes
        unique_sample_indexes = torch.arange(n_unique_samples).unsqueeze(-1).unsqueeze(-1)
        column_indexes = torch.arange(n_cols).unsqueeze(-1).unsqueeze(0)
        beam_indexes = torch.arange(beam_width).unsqueeze(0).unsqueeze(0)

        #Mask out each current decision
        future_availableCols.diagonal(dim1=-2, dim2=-1).fill_(False)
        future_availableCols[:, :, n_cols - 1] = True
        future_availableCols = future_availableCols.unsqueeze(-2).expand(-1,-1,beam_width,-1).clone()
        final_row = min(current_row + max_depth, len(ordered_rows))
        for i in range(current_row+1,final_row):
            top_cols = future_logsumexps.unsqueeze(-1) + log_likelihoods[ordered_rows[i],:].unsqueeze(0).unsqueeze(0)
            top_cols.masked_fill_(~future_availableCols | ~top_cols.isfinite(),-torch.inf)

            top_cols = top_cols.reshape(future_availableCols.shape[0],future_availableCols.shape[1],-1)
            top_cols,indexes = torch.topk(top_cols,beam_width,dim=-1)
            beam_idx, col_idx = torch.unravel_index(indexes,(beam_width,n_cols))
            future_logsumexps = top_cols
            future_availableCols[unique_sample_indexes,column_indexes,beam_indexes,:] = future_availableCols[unique_sample_indexes,column_indexes,beam_idx,:]
            future_availableCols[unique_sample_indexes,column_indexes,beam_indexes,col_idx] = False
            future_availableCols[...,-1] = True
        #
        future_logsumexps = future_logsumexps.logsumexp(dim=-1)

        return future_logsumexps[reverse_index,...]
    #
    def row_beam_search_fast(self,
            availableCols: torch.Tensor,
            log_likelihoods: torch.Tensor,
            ordered_rows: torch.Tensor,
            current_row: int,
            beam_width: int = 50,
            max_depth: int = 5
    ):
        """
        Beam search over columns for each unique mask row.

        Inputs
          availableCols:   (N, C)  bool
          log_likelihoods: (R, C)  float (log-prob per row/column)
          ordered_rows:    (R,)    long indices
          current_row:     int     start row in ordered_rows (exclusive)
          beam_width:      int     beams per (unique_mask, choice)

        Returns
          (N, C) tensor: -logsumexp over beams per initial column, mapped back to N.
        """


        # 1) Deduplicate masks
        unique_masks, reverse_index = torch.unique(
            availableCols, dim=0, return_inverse=True
        )
        U, C = unique_masks.shape
        B = beam_width

        device = 'mps'
        dtype = torch.float32

        unique_masks = unique_masks.to(device)
        log_likelihoods = log_likelihoods.type(dtype).to(device)


        # 2) Build per-choice masks once and keep 4D shape throughout: (U, C, B, C)
        base = unique_masks.unsqueeze(1).expand(U, C, C).clone()  # (U, C, C)
        base.diagonal(dim1=-2, dim2=-1).fill_(False)  # zero the diagonal choices
        base[:, :, C - 1] = True  # keep last column available
        per_choice_masks = base.unsqueeze(2).expand(U, C, B, C).clone()  # (U, C, B, C)

        # 3) Beam scores
        beam_scores = torch.zeros((U, C, B), dtype=dtype, device=device)

        # 4) Iterate over future rows
        final_row = min(current_row + max_depth, len(ordered_rows))
        for i in range(current_row + 1, final_row):
            row_idx = ordered_rows[i]
            row_ll = log_likelihoods[row_idx].view(1, 1, 1, C)  # (1,1,1,C)

            # (U,C,B,1) + (1,1,1,C) -> (U,C,B,C)
            scores = beam_scores.unsqueeze(-1) + row_ll

            # Mask out unavailable / non-finite
            scores.masked_fill_(~per_choice_masks | ~scores.isfinite(), float('-inf'))

            # Top-B over combined (beam × column)
            scores2d = scores.view(U, C, B * C)
            top_vals, flat_idx = torch.topk(scores2d, B, dim=-1)
            beam_idx = flat_idx // C  # (U, C, B)
            col_idx = flat_idx % C  # (U, C, B)

            beam_scores = top_vals  # (U, C, B)

            # Propagate masks: pick parent beams, then disable the chosen column
            gather_idx = beam_idx.unsqueeze(-1).expand(U, C, B, C)  # (U,C,B,C)
            next_masks = per_choice_masks.gather(2, gather_idx)  # (U,C,B,C)
            next_masks.scatter_(3, col_idx.unsqueeze(-1), False)  # turn off chosen col
            next_masks[..., -1] = True  # keep last col on
            per_choice_masks = next_masks.contiguous()

        # 5) Collapse beams and map back to original N
        out = beam_scores.logsumexp(dim=-1)  # (U, C)
        return out[reverse_index].to(availableCols.device)  # (N, C)
    @torch.no_grad()
    def compute_row_entropies(self, log_likelihood_matrix: torch.tensor) -> torch.Tensor:
        entropies = log_likelihood_matrix - log_likelihood_matrix.logsumexp(dim=-1, keepdim=True)
        entropies = entropies.exp()*entropies
        entropies.masked_fill_(~entropies.isfinite(),0.0)
        return -entropies.sum(dim=-1)

    #
    def _sample(self, sample_shape=torch.Size(), allow_resample=True) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._distances.shape[0],), dtype=torch.bool)
        availableCols = torch.ones(sample_shape+(self._distances.shape[1]+1,), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float64)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionLogLikelihood()
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float64)
        decision_counter = 0
        gibbs_sample = False
        row_entropies = self.compute_row_entropies(self._base_row_decision_likelihoods_unnormalized)
        row_order = torch.argsort(row_entropies,descending=False)
        for _ in tqdm(enumerate(row_order),desc="Matching Rows"):
            try:
                step_weights = self.__getNextInSequence(sample, sample_indexes, row_order, availableCols, decision_log, decision_counter)
                sample_weights = sample_weights+step_weights

                if allow_resample:
                    self._resample(sample, sample_weights, availableRows,availableCols,decision_log,decision_counter)
            except Exception as e:
                raise SamplingError(f"Error during sampling of matching matrices \n"
                                    f"Step: {decision_counter} of {availableRows.shape[0]} \n"
                                    f"CSP Distribution Parameters: {self.csp_distribution.param} \n"
                                    f"CSP_weight logits: {self._csp_mixture_weights} probits: {(self._csp_mixture_weights-self._csp_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"matching_weight_logits: {self._matching_mixture_weights} probits: {(self._matching_mixture_weights - self._matching_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"missing_weight_logits: {self._missing_mixture_weights} probits: {(self._missing_mixture_weights - self._missing_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e

            decision_counter += 1

        if allow_resample:
            self._resample(sample, sample_weights, availableRows,availableCols,decision_log,decision_counter)
        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample, sample_indexes, availableRows,availableCols, decision_log, gibbs_sample



    def sample(self,sample_shape=torch.Size()) -> torch.tensor:

        sample, sample_indexes, avaliableRows,availableCols,decision_log,gibbs_sample = self._sample(sample_shape)
        validateSample(sample,availableCols)
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
