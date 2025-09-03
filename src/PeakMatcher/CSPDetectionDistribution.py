import logging

import scipy.spatial.distance
import sklearn.cluster
import torch
import torch.distributions as torchdist
import numpy as np
from torch.profiler import record_function
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering, leaves_list
import math
class SamplingError(Exception):
    pass
class EnumerationError(Exception):
    pass

class LowESSError(Exception):
    pass

class ExcessiveBeamSearchError(Exception):
    pass
from tqdm import tqdm

def to_linkage(model):
    children = model.children_
    n = model.labels_.size
    # counts: number of original samples under each merge
    counts = np.zeros(children.shape[0], dtype=float)
    for i, (a, b) in enumerate(children):
        ca = 1 if a < n else counts[a - n]
        cb = 1 if b < n else counts[b - n]
        counts[i] = ca + cb
    Z = np.column_stack([children, model.distances_, counts]).astype(float)
    return Z


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
                 max_predicted_dnm: float,
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
        self._max_predicted_dnm = max_predicted_dnm

        self._csp_mixture_weights = (csp_mixture_weights - csp_mixture_weights.logsumexp(dim=0,keepdim=True)).detach().clone()
        self._matching_mixture_weights = (matching_mixture_weights - matching_mixture_weights.logsumexp(dim=0, keepdim=True)).detach().clone()
        #self._missing_mixture_weights = (missing_mixture_weights - missing_mixture_weights.logsumexp(dim=0, keepdim=True)).detach().clone()
        self._csp_distribution = csp_distribution.clone()
       # self._non_matching_parameters =
        self._non_matching_distribution = non_matching_distribution

        self._no_csp_distribution = torch.distributions.Chi2(torch.tensor([2.0],dtype=torch.float64)) #chi2 distribution for

        self.eps_float64 = torch.finfo(torch.float64).eps
        self.max_float64 = torch.finfo(torch.float64).max
        self.min_float64 = torch.finfo(torch.float64).min

        self._calculateDecisionLogLikelihood()


    #
    def clone(self):
        return CSPDetectionDistribution(self._distances.detach().clone(),
                                        self._max_predicted_dnm,
                                        self._csp_mixture_weights.detach().clone(),
                                        self._matching_mixture_weights.detach().clone(),
                                        None,
                                        self._csp_distribution.clone(),
                                        self._non_matching_distribution.clone())

    def _calculateDecisionLogLikelihood(self):
        self._loglikelihoodMatrix = torch.stack((self._no_csp_distribution.log_prob(self._distances).clamp(min=self.min_float64),
                                                self._csp_distribution.log_prob(self._distances).clamp(min=self.min_float64),
                                                self._non_matching_distribution.log_prob(self._distances).clamp(min=self.min_float64)),dim=2)
        self._event_shape = (self._distances.shape[0],)
        if self._loglikelihoodMatrix[:,:,1].isnan().any():
            print(f"Error with CSP dist evaluation: parameters are alpha,scale {self.csp_distribution.alpha} {self.csp_distribution.scale}")

        assert not self._loglikelihoodMatrix[:,:,0].isnan().any()
        assert not self._loglikelihoodMatrix[:, :,  1].isnan().any()
        assert not self._loglikelihoodMatrix[:, :, 2].isnan().any()

        max_csp = torch.tensor([self._max_predicted_dnm])
        differential = torch.cat([self._no_csp_distribution.log_prob(max_csp), self._csp_distribution.log_prob(max_csp)]) + self._csp_mixture_weights
        differential = differential.logsumexp(dim=0).detach()
        differential = torch.cat([torch.atleast_1d(differential),self._non_matching_distribution.log_prob(max_csp).detach()]) #+ self._matching_mixture_weights
        differential = differential[0]-differential[1]
        unnormalized_csp_posterior_probabilities = self._loglikelihoodMatrix[:,:,0:2].detach() + self._csp_mixture_weights.detach()

        self._csp_posterior_probabilities = unnormalized_csp_posterior_probabilities - unnormalized_csp_posterior_probabilities.logsumexp(dim=-1,keepdim=True)

        #unweighted_matching_loglikelihoods = (self._loglikelihoodMatrix[:,:,0:2] + self._csp_mixture_weights.detach()).logsumexp(dim=-1)
        unweighted_matching_loglikelihoods = (
                    self._loglikelihoodMatrix[:, :, 0:2]).logsumexp(dim=-1)
        #parameter corrected loglikelihoods

        final_matching_likelihoods = (torch.stack([unweighted_matching_loglikelihoods,self._loglikelihoodMatrix[:,:,2]],dim=2))
                                      #+ self._matching_mixture_weights.detach())

        self._matching_likelihood = final_matching_likelihoods[:,:,0]
        self._match_non_matching_loglikelihoods = final_matching_likelihoods[:,:,1] + differential

        distributed_missing_mixture_weights = torch.zeros((self._distances.shape[1]+1,))
        #distributed_missing_mixture_weights[:-1] = self._missing_mixture_weights[0].unsqueeze(-1).detach()
        #distributed_missing_mixture_weights[:-1] = (self._missing_mixture_weights[0].unsqueeze(-1).exp().detach() / len(
        #    distributed_missing_mixture_weights[:-1])).log()
        #distributed_missing_mixture_weights[-1] = self._missing_mixture_weights.detach()[1]

        self._base_row_decision_likelihoods= torch.zeros((self._distances.shape[0],self._distances.shape[1]+1),dtype=torch.float64)
        self._base_row_decision_likelihoods[:,:] = self._match_non_matching_loglikelihoods.detach().sum(dim=-1).unsqueeze(1)
        self._base_row_decision_likelihoods[:,:-1] += self._matching_likelihood.detach() - self._match_non_matching_loglikelihoods.detach()
        self._base_row_decision_likelihoods += distributed_missing_mixture_weights.unsqueeze(0).detach()
        self._base_row_decision_likelihoods_unnormalized = self._base_row_decision_likelihoods.clone()


    def _detach(self):
        self._distances.detach_()
        self._loglikelihoodMatrix.detach_()
        self._csp_posterior_probabilities.detach_()
        self._matching_likelihood = None
        self._match_non_matching_loglikelihoods = None
        self._base_row_decision_likelihoods = None
        self._base_row_decision_likelihoods_unnormalized = None

    def shortest_paths_floyd(self, D: torch.Tensor, undirected: bool = True) -> torch.Tensor:
        """
        All-pairs shortest paths (Floyd–Warshall) on a weighted graph.

        Args:
            D: (N, N) tensor of edge weights where D[i, j] is the distance of the
               direct edge i->j. Use +inf where there is no direct edge.
            undirected: If True, treat the graph as undirected by taking
               min(D, D.T) before running the algorithm.

        Returns:
            (N, N) tensor of shortest-path distances. +inf means no route exists.
        """
        assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be square"
        N = D.shape[0]
        dtype = D.dtype if D.is_floating_point() else torch.float32

        # Work on a copy, ensure floating dtype
        dist = D.to(dtype=dtype).clone()

        # If undirected, keep the shorter of i->j and j->i
        if undirected:
            dist = torch.minimum(dist, dist.t())

        # Zero on the diagonal (distance from a node to itself)
        dist.fill_diagonal_(0.0)

        # Floyd–Warshall relaxation: dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        # Use a temporary candidate matrix to avoid aliasing issues during in-place minimum.
        for k in range(N):
            cand = dist[:, k:k + 1] + dist[k:k + 1, :]
            torch.minimum(dist, cand, out=dist)

        return dist

    def top_mass_mask_from_logits(self,logits: torch.Tensor, mass: float = 0.99) -> torch.Tensor:
        """
        logits: (m, n) tensor of unnormalized log-probs
        mass:   target cumulative probability mass (e.g., 0.99)

        returns: (m, n) boolean mask picking the minimal set of top entries per row
                 whose probabilities sum to at least `mass`.
        """
        # 1) Convert logits -> probs row-wise
        probs = torch.softmax(logits, dim=1)  # (m, n)

        # 2) Sort probs descending per row, keep indices
        probs_sorted, idx = probs.sort(dim=1, descending=True)  # (m, n), (m, n)

        # 3) Row-wise cumulative sums
        cdf = probs_sorted.cumsum(dim=1)  # (m, n)

        # 4) First index where cum. mass crosses `mass`
        #    (guaranteed to exist since last elem sums to ~1)
        cutoff = (cdf >= mass).to(probs.dtype).argmax(dim=1)  # (m,)

        # 5) Build a mask in sorted space: True for positions <= cutoff
        n = logits.size(1)
        arange = torch.arange(n, device=logits.device).unsqueeze(0).expand_as(probs_sorted)
        mask_sorted = arange <= cutoff.unsqueeze(1)  # (m, n) bool

        # 6) Scatter mask back to original column order
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter(1, idx, mask_sorted)

        return mask
    @torch.no_grad()
    def compute_linkage_matrix(self, log_likelihood_matrix: torch.tensor) -> torch.Tensor:
        self._candidate_indicies = []
        candidate_cols = log_likelihood_matrix[:,:-1] > (log_likelihood_matrix[:,-1].unsqueeze(-1))
        #candidate_cols = candidate_cols & self.top_mass_mask_from_logits(log_likelihood_matrix, mass=0.99)[:,:-1]
        mat = log_likelihood_matrix[:,:-1].masked_fill(~candidate_cols, float('-inf'))
        vals, indexes = torch.topk(mat,k=4,dim=-1)
        c = torch.zeros_like(candidate_cols)
        c[torch.arange(log_likelihood_matrix.shape[0]).unsqueeze(-1),indexes] = True
        candidate_cols = candidate_cols & c

        mat = log_likelihood_matrix.clone()
        mat[:,:-1].masked_fill_(~candidate_cols, float('-inf'))
        matches_only = mat.softmax(dim=-1)[:,:-1].log()

        collisions = candidate_cols.unsqueeze(1) & candidate_cols.unsqueeze(0)

        collisions_in_range = (torch.abs(matches_only.unsqueeze(-2) - matches_only.unsqueeze(0)) - abs(math.log(10)) < 0)
        for col in range(log_likelihood_matrix.shape[1]-1):
            temp = torch.nonzero(collisions_in_range[:,:,col])
            unique_rows = torch.unique(temp.flatten())
            collisions_in_range[unique_rows.unsqueeze(-1),unique_rows.unsqueeze(0),col] = True

        enhanced_collisions = collisions & collisions_in_range

        candidate_cols = enhanced_collisions.any(dim=1)
        self._linkage_matrix = 1.0/(enhanced_collisions).sum(dim=-1)

        #diff_collisions = enhanced_collisions.sum(dim=-1) - collisions.sum(dim=-1)


        #self._linkage_matrix.fill_diagonal_(1)
        for row in range(candidate_cols.shape[0]):
            candidates = torch.cat([torch.nonzero(candidate_cols[row].squeeze(),as_tuple=False).reshape(-1),
                                    torch.tensor([log_likelihood_matrix.shape[1]-1]).reshape(-1)],dim=0)
            self._candidate_indicies.append(torch.atleast_1d(candidates))
        distance_matrix = self.shortest_paths_floyd(self._linkage_matrix, undirected=True)
        return distance_matrix



    def __getNextInSequence(self,
                                             sample: torch.tensor,
                                             sample_indicies: torch.tensor,
                                             row_order: torch.tensor,
                                             max_depths: torch.tensor,
                                             candidate_indicies: torch.tensor,
                                             availableCols: torch.tensor,
                                             decision_log: torch.tensor,
                                             decision_counter: int):

        current_row_index = row_order[decision_counter]

        log_probabilities, top_k_indicies = self.row_beam_search(availableCols,
                                                                 self._base_row_decision_likelihoods_unnormalized,
                                                                 row_order,
                                                                 decision_counter,
                                                                 all_candidate_indicies=candidate_indicies,
                                                                 max_beam_width=1000,
                                                                 max_depth=max_depths[decision_counter])

        #log_probabilities.masked_fill_(~availableCols[:,top_k_indicies],-torch.inf)
        log_probabilities -= log_probabilities.logsumexp(dim=-1,keepdim=True)


        with record_function("Column_Sampling"):
            unmapped_matched_columns = torch.multinomial(log_probabilities.exp(), 1, replacement=True).type(torch.int32).squeeze()
            matched_columns = top_k_indicies[sample_indicies,unmapped_matched_columns].type(torch.int32)
            no_matched_columns = matched_columns >= self._base_row_decision_likelihoods_unnormalized.shape[1] -1

        # availableRows[sample_indicies, sampled_rows] = False

        decision_log[sample_indicies, decision_counter, 0] = row_order[decision_counter].type(torch.float64)
        decision_log[sample_indicies, decision_counter, 2] = log_probabilities[sample_indicies, unmapped_matched_columns].type(torch.float64)  # +row_probabilities
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
                  ESS_History: torch.tensor,
                  force_resample=False):

        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        ess_ratio = ess/sample_weights.shape[0]
        assert ess.isfinite().all()
        min = max(0,decision_counter-10)
        if decision_counter < ESS_History.shape[0]:
            ESS_History[decision_counter] = ess_ratio
        resample = (ESS_History[min:decision_counter] < 0.5).all()
        if force_resample:
            #if ess_ratio < 0.05:
            #    raise LowESSError(f"ESS ratio {ess_ratio} is too low at resampling; reduce expected_max_csp")
            #print(f"{decision_counter}: ESS ratio:, {ess_ratio:0.3f}; Resample")
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
            return
        else:
            #print(f"{decision_counter}: ESS ratio:, {ess_ratio:0.3f}")
            return
    #
    def deduplicate_masks(self,availableCols: torch.Tensor):
        unique_masks, inverse = torch.unique(availableCols, dim=0, return_inverse=True)
        return unique_masks, inverse

    def unique_rows_by_bits(self,
                            mask: torch.Tensor,
                            bit_idx: torch.Tensor,
                            return_counts: bool = False,
                            return_representatives: bool = True):
        """
        mask: [N, B] bool/0-1 tensor
        bit_idx: 1D LongTensor of columns (bits) to consider for uniqueness

        Returns:
          unique_keys: [K, len(bit_idx)] the unique row patterns restricted to bit_idx
          inverse: [N] int mapping each row -> its unique_keys index (0..K-1)
          counts: [K] (optional) how many rows map to each unique
          reps_full: [K, B] (optional) representative full rows from the original mask
        """
        # restrict to the subset of bits (columns)
        keys = mask[:, bit_idx]  # [N, |bits|]

        # unique rows based on those bits
        out = torch.unique(keys, dim=0, return_inverse=True, return_counts=return_counts)
        if return_counts:
            unique_keys, inverse, counts = out
        else:
            unique_keys, inverse = out
            counts = None

        reps_full = None
        if return_representatives:
            # pick the first occurrence row for each unique pattern
            # (stable argsort + cumsum trick)
            if counts is None:
                _, _, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
            order = torch.argsort(inverse, stable=True)
            starts = torch.cat([inverse.new_zeros(1), counts.cumsum(0)[:-1]])
            rep_idx = order[starts]
            reps_full = mask[rep_idx]  # representative full rows

        return unique_keys, inverse, counts, reps_full
    @torch.no_grad()
    def row_beam_search(self,
                        availableCols: torch.Tensor,
                        log_likelihoods: torch.Tensor,
                        ordered_rows: torch.Tensor,
                        current_row: int,
                        all_candidate_indicies: torch.tensor,
                        max_beam_width: int,
                        max_depth: int, ):

        candidate_indicies = all_candidate_indicies[ordered_rows[current_row]]
        n_cols = availableCols.shape[1]
        k = candidate_indicies.shape[0]
        final_row = min(current_row + max_depth+1, len(ordered_rows))

        relevant_cols = torch.zeros((availableCols.shape[1]), dtype=torch.bool)
        relevant_cols[candidate_indicies] = True
        for i in range(current_row+1, final_row):
            relevant_cols[all_candidate_indicies[ordered_rows[i]]] = True
        #



        #Check which decisions are still available in at least one sample
        _, reverse_index, _, unique_availableCols,   = self.unique_rows_by_bits(availableCols,torch.nonzero(relevant_cols).squeeze())
        n_unique_samples = unique_availableCols.shape[0]
        top_k_col_indexes = candidate_indicies.expand(n_unique_samples, -1).clone()
        top_k_likelihoods = log_likelihoods[ordered_rows[current_row],:].masked_fill(~unique_availableCols, -torch.inf)
        top_k_likelihoods = top_k_likelihoods[torch.arange(n_unique_samples).unsqueeze(-1), top_k_col_indexes]
        #Select top k indicies
        #top_k_likelihoods, top_k_col_indexes = torch.topk(candidate_likelihoods, k=k, dim=-1)

        #unique_availableCols, reverse_index = torch.unique(availableCols[:,top_k_col_indexes], dim=0, return_inverse=True)



        #print(f"{n_unique_samples}: Beam search")
        #build structures to use in the looping
        future_availableCols = unique_availableCols.clone().unsqueeze(1).expand(-1,k,-1).clone()
        future_logsumexps = torch.zeros((n_unique_samples,k,max_beam_width),dtype=torch.float32)

        print(f"Running Beam Search: unique: {n_unique_samples}, k:{k}, max_beam_width:{max_beam_width}, max_depth:{max_depth}")

        #Build indexes
        unique_sample_indexes = torch.arange(n_unique_samples).unsqueeze(-1).unsqueeze(-1)
        #column_indexes = torch.arange(n_cols).unsqueeze(-1).unsqueeze(0)
        k_indexes = torch.arange(k).unsqueeze(-1).unsqueeze(0)

        #return future_logsumexps[reverse_index, ...].sum(dim=-1), top_k_col_indexes

        #Mask out each current decision
        future_availableCols[unique_sample_indexes.squeeze(-1), torch.arange(k).unsqueeze(0), top_k_col_indexes] = False
        future_availableCols[:, :, n_cols - 1] = True
        future_availableCols = future_availableCols.unsqueeze(-2).expand(-1, -1, max_beam_width, -1).clone()
        future_logsumexps[unique_sample_indexes.squeeze(-1),k_indexes.squeeze().unsqueeze(0),:] = top_k_likelihoods.unsqueeze(-1).type(torch.float)
        current_beam_width = 1
        for i in range(current_row+1,final_row):
            current_candidate_indices = all_candidate_indicies[ordered_rows[i]]
            active_future_logsumexps = future_logsumexps[:,:,:current_beam_width]

            top_cols = torch.where(future_availableCols[:,:,:current_beam_width,current_candidate_indices],
                                   log_likelihoods[ordered_rows[i],current_candidate_indices].unsqueeze(0).unsqueeze(0),
                                   -torch.inf)
            top_cols += active_future_logsumexps.unsqueeze(-1)

            if top_cols.numel()*top_cols.element_size() > 1E9:
                raise ExcessiveBeamSearchError("Internal tensor exceeded approx 1 GB")


            proposed_beam_width = top_cols.shape[-1]*top_cols.shape[-2]
            top_cols = top_cols.reshape(n_unique_samples, k, -1)
            if proposed_beam_width > max_beam_width:
                top_cols,indexes = torch.topk(top_cols, max_beam_width, dim=-1)
                originating_beam_idx, col_idx = torch.unravel_index(indexes, (current_beam_width, len(current_candidate_indices)))
                proposed_beam_width = max_beam_width
            else:
                originating_beam_idx, col_idx = torch.unravel_index(torch.arange(proposed_beam_width), (current_beam_width, len(current_candidate_indices)))

            col_idx = current_candidate_indices[col_idx]

            proposed_beam_indexes = torch.arange(proposed_beam_width).unsqueeze(0).unsqueeze(0)
            #update Beams
            future_logsumexps[:,:,:proposed_beam_width] = top_cols
            future_availableCols[unique_sample_indexes,k_indexes,proposed_beam_indexes,:] = future_availableCols[unique_sample_indexes,k_indexes,originating_beam_idx,:]
            future_availableCols[unique_sample_indexes,k_indexes,proposed_beam_indexes,col_idx] = False
            future_availableCols[:,:,:,-1] = True
            running_probability = future_logsumexps.logsumexp(dim=-1).clone()
            running_probability = (running_probability - running_probability.logsumexp(dim=-1,keepdim=True)).exp()
            current_beam_width = proposed_beam_width

        #
        future_logsumexps = future_logsumexps[:,:,:current_beam_width].logsumexp(dim=-1)

        return future_logsumexps[reverse_index,...], top_k_col_indexes[reverse_index,...]
    #
    @torch.no_grad()
    def determineRowOrder(self, log_likelihood_matrix: torch.tensor):

        distance = self.compute_linkage_matrix(self._base_row_decision_likelihoods_unnormalized).numpy()
        threshold = np.max(distance[distance < float('inf')])
        distance[distance == float('inf')] = threshold+1
        #cluster_labels = torch.from_numpy(SpectralClustering(affinity='precomputed',assign_labels='cluster_qr').fit_predict(W))
        model = AgglomerativeClustering(n_clusters=None,metric='precomputed',linkage='single',distance_threshold=1.01).fit(distance)
        linkage = to_linkage(model)
        cluster_labels = torch.from_numpy(model.labels_)
        unique_clusters, inverse, counts = torch.unique(cluster_labels, return_counts=True, return_inverse=True)
        ordered_indexes = leaves_list(optimal_leaf_ordering(linkage,scipy.spatial.distance.squareform(distance)))
        ordered_cluster_labels = cluster_labels[ordered_indexes]
        max_depths = torch.zeros_like(ordered_cluster_labels)
        for i,cluster in enumerate(unique_clusters):
            cluster_mask = ordered_cluster_labels == cluster
            running_count = cluster_mask.cumsum(dim=-1)
            running_count[~cluster_mask] = 0
            running_count[cluster_mask] = counts[i] - running_count[cluster_mask]
            max_depths += running_count
        #
        neighbor_count = (self._distances[ordered_indexes,:] < self._max_predicted_dnm).sum(dim=-1) + 1
        return torch.from_numpy(ordered_indexes), max_depths, neighbor_count

    #
    def _sample(self, sample_shape=torch.Size()) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._distances.shape[0],), dtype=torch.bool)
        availableCols = torch.ones(sample_shape+(self._distances.shape[1]+1,), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float64)
        sample_indexes = torch.arange(sample_shape[0], dtype=torch.int32)
        self._calculateDecisionLogLikelihood()
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float64)
        decision_counter = 0
        gibbs_sample = False
        ESS_History = torch.ones((availableRows.shape[1],), dtype=torch.float64)*-1
        if not hasattr(self,'row_order'):
            row_order,max_depths,neighbor_count= self.determineRowOrder(self._base_row_decision_likelihoods_unnormalized)

            #self.cluster_count[...] = 10000
        for _ in tqdm(enumerate(row_order),desc="Matching Rows"):
            try:
                step_weights = self.__getNextInSequence(sample, sample_indexes, row_order,max_depths,self._candidate_indicies, availableCols, decision_log, decision_counter)
                sample_weights = sample_weights+step_weights

                self._resample(sample, sample_weights, availableRows,availableCols,decision_log,decision_counter,ESS_History,force_resample=max_depths[decision_counter] == 0)
            except Exception as e:
                raise SamplingError(f"Error during sampling of matching matrices {e} \n"
                                    f"Step: {decision_counter} of {availableRows.shape[0]} \n"
                                    f"CSP Distribution Parameters: {self.csp_distribution.param} \n"
                                    f"CSP_weight logits: {self._csp_mixture_weights} probits: {(self._csp_mixture_weights-self._csp_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"matching_weight_logits: {self._matching_mixture_weights} probits: {(self._matching_mixture_weights - self._matching_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e
                                    #f"missing_weight_logits: {self._missing_mixture_weights} probits: {(self._missing_mixture_weights - self._missing_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e

            decision_counter += 1

        #self._resample(sample, sample_weights, availableRows,availableCols,decision_log,decision_counter,ESS_History,force_resample=True)
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

    @csp_distribution.setter
    def csp_distribution(self, csp_distribution):
        self._csp_distribution = csp_distribution.clone()
        self._calculateDecisionLogLikelihood()

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

    @csp_mixture_weights.setter
    def csp_mixture_weights(self,value):
        self._csp_mixture_weights = value.detach().clone()
        self._calculateDecisionLogLikelihood()

    @property
    def matching_mixture_weights(self) -> torch.Tensor:
        return self._matching_mixture_weights

    @matching_mixture_weights.setter
    def matching_mixture_weights(self,value):
        self._matching_mixture_weights = value.detach().clone()
        self._calculateDecisionLogLikelihood()

    #@property
    #def missing_mixture_weights(self) -> torch.Tensor:
     #   return self._missing_mixture_weights
    #@missing_mixture_weights.setter
    #def missing_mixture_weights(self,value):
    #    self._missing_mixture_weights = value.detach().clone()
    #    self._calculateDecisionLogLikelihood()

#
