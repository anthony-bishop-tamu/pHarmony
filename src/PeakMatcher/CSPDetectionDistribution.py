import logging

import torch
import torch.distributions as torchdist
import numpy as np
from torch.profiler import record_function
from sklearn.cluster import AgglomerativeClustering
import math
class SamplingError(Exception):
    pass
class EnumerationError(Exception):
    pass
from tqdm import tqdm
def compute_joint_row_probability(likelihoods: torch.tensor,availableCols: torch.tensor = None):
        if likelihoods.shape[0] != 2:
            raise EnumerationError
        n_cols = likelihoods.shape[1]
        log_joint = likelihoods[0, :].unsqueeze(1) + likelihoods[1, :].unsqueeze(0)
        log_joint.fill_diagonal_(-torch.inf)
        log_joint[n_cols - 1, n_cols - 1] = likelihoods[0, n_cols - 1] + likelihoods[1, n_cols - 1]
        if availableCols is not None:
            log_joint = torch.where(availableCols.unsqueeze(0) & availableCols.unsqueeze(-1),log_joint,-torch.inf)
        log_joint_evidence = log_joint.logsumexp(dim=(-1, -2), keepdim=True)
        log_joint -= log_joint_evidence
        log_x = log_joint.logsumexp(dim=-1, keepdim=True)
        log_y = log_joint.logsumexp(dim=0,keepdim=True)
        return log_joint, log_x, log_y

class RowClusterSampler:
    def __init__(self,
                 log_likelihood: torch.tensor):
       self.shape = log_likelihood.shape
       self._likelihood = log_likelihood
       self._joint_probability = None
       if self._likelihood.shape[0] == 1:
           self._joint_probability = (log_likelihood - log_likelihood.logsumexp(dim=-1, keepdim=True)).exp().squeeze()
       elif self._likelihood.shape[0] == 2:
           self._joint_probability, x, y = compute_joint_row_probability(log_likelihood)
           self._joint_probability.exp_()
       else:
           #joint probability to large for explicit calculation, gibbs sample
           gibbs_sample,logits = self.gibbs_sample(100,1000,10,)
           self._gibbs_sample = (gibbs_sample,logits)
           self._train_auto_regression(gibbs_sample)


    def random_pairs(self,index_1d: torch.Tensor):
        """
        Given a 1-D tensor of indices, return a tensor of random,
        non-overlapping pairs plus (optionally) a leftover element.

        Returns
        -------
        pairs :  (⌊N/2⌋, 2) tensor   # each row is a pair
        leftover : (0 or 1,) tensor  # empty if N even
        """
        # 1) Flatten in case a view sneaks in
        idx = index_1d.flatten()

        # 2) Shuffle once with torch.randperm (fast and GPU-friendly)
        shuffled = idx[torch.randperm(idx.numel(), device=idx.device)]

        # 3) Slice & reshape
        n_pairs = shuffled.numel() // 2
        pairs = shuffled[: 2 * n_pairs].view(n_pairs, 2)

        # 4) Handle the odd element (if any)
        leftover = shuffled[2 * n_pairs:]  # shape (0,) or (1,)

        return pairs, leftover

    def sample_joint(self,joint_probability):
        shape = joint_probability.shape
        cumulative_probability = joint_probability.flatten()
        cumulative_probability = torch.cumsum(cumulative_probability, dim=0)
        u =torch.rand(1)
        idx = torch.searchsorted(cumulative_probability, u,right=False)
        coords = torch.unravel_index(idx,shape)
        return coords,joint_probability.flatten()[idx]

    def draw_sample(self):
        if self._joint_probability is not None:
            coords,probability = self.sample_joint(self._joint_probability)
            return coords,probability
        else:
            coords,probability = self._draw_auto_regression()
            return coords,probability

    def gibbs_sample(
            self,
            n_samples: int,
            burn_in: int = 100,
            thinning: int = 1,
            temperatures = torch.tensor([1.0]),
            base_temp_prob: float = 0.5,
    ): #-> Tuple[torch.Tensor, torch.Tensor]:
        """Draw samples using Gibbs + Metropolis moves.

        Parameters
        ----------
        n_samples : int
            Number of samples to *store* (after burn‑in & thinning).
        burn_in : int, default 100
            Number of full Gibbs sweeps to discard.
        thinning : int, default 1
            Keep one sample every *thinning* sweeps.
        temperatures : 1‑D Tensor, optional
            Positive temperature ladder.  Defaults to ``logspace(0, 2, 6)``.
        base_temp_prob : float in (0,1)
            Probability of drawing the *cold* chain (T = temperatures[0]).

        Returns
        -------
        samples : LongTensor [n_samples, n_rows]
            Column indices for each row.
        log_likelihoods : FloatTensor [n_samples]
            Sum of row log‑likelihoods for every stored sample.
        """
        device = self._likelihood.device
        n_rows, n_cols = self.shape

        # ------------------------------------------------------------------
        # 1)  Temperature ladder and its geometric CDF
        # ------------------------------------------------------------------
        if temperatures is None:
            # 1, 3.16, 10, 31.6, 100, 316  (covers ~5 orders of magnitude)
            temperatures = torch.logspace(0, 1, steps=10, device=device)
        assert (temperatures > 0).all(), "Temperatures must be positive."

        geom = base_temp_prob * (1.0 - base_temp_prob) ** torch.arange(
            len(temperatures), device=device
        )
        temp_cdf = torch.cumsum(geom, dim=0)
        temp_cdf[-1] = 1.0  # exact 1 for searchsorted

        # ------------------------------------------------------------------
        # 2)  State initialisation (greedy on currently available columns)
        # ------------------------------------------------------------------
        state = torch.full((n_rows,), -1, dtype=torch.long, device=device)
        available = torch.ones((n_cols,), dtype=torch.bool, device=device)

        for r in range(n_rows):
            best_col = torch.argmax(
                torch.where(available, self._likelihood[r], torch.full((), -torch.inf, device=device))
            )
            state[r] = best_col
            if best_col < n_cols - 1:
                available[best_col] = False

        # ------------------------------------------------------------------
        # 3)  Storage tensors
        # ------------------------------------------------------------------
        samples = torch.empty((n_samples, n_rows), dtype=torch.long, device=device)
        log_likes = torch.empty((n_samples,), dtype=torch.float32, device=device)

        # Counters
        stored = 0
        sweep = 0  # counts full Gibbs sweeps

        # ------------------------------------------------------------------
        # 4)  Gibbs sweeps until we have all requested samples
        # ------------------------------------------------------------------
        deltas_v_temperature = [ [] for _ in range(len(temperatures))]
        while stored < n_samples:
            # ----- 4.1  Draw temperature for *this* sweep ------------------
            t_idx = torch.searchsorted(temp_cdf, torch.rand((), device=device), right=True)
            T = temperatures[t_idx]
            beta = 1.0 / T

            # ----- 4.2  Update the rows two‑by‑two -------------------------
            pairs, leftover = self.random_pairs(torch.arange(n_rows, device=device))

            for rows in pairs:
                # Joint log‑probability table for the two rows
                available[state[rows[0]]] = True
                available[state[rows[1]]] = True
                joint_logp, _, _ = compute_joint_row_probability(
                    self._likelihood[rows], available
                )
                #joint_logp -= joint_logp.max()  # numerical stability

                # Propose new columns
                new_cols, _ = self.sample_joint(joint_logp.exp())  # returns (c0,c1)
                log_p_new = joint_logp[new_cols[0], new_cols[1]]
                log_p_old = joint_logp[state[rows[0]], state[rows[1]]]

                #print(f"log_p_new: {log_p_new}, log_p_old: {log_p_old}")
                assert available[new_cols[0]] and available[new_cols[1]]
                # Metropolis acceptance
                acc_prob = torch.exp(beta * (log_p_new - log_p_old)).clamp(max=1.0)
                deltas_v_temperature[t_idx].append((log_p_new - log_p_old).item())
                if torch.rand((), device=device) < acc_prob:
                    # Accept → update availability mask first

                    state[rows[0]] = new_cols[0]
                    state[rows[1]] = new_cols[1]

                if state[rows[0]] < n_cols - 1:
                    available[state[rows[0]]] = False
                if state[rows[1]] < n_cols - 1:
                    available[state[rows[1]]] = False

            # ----- 4.3  Single leftover row (if n_rows is odd) -------------
            if leftover.numel():
                r = leftover[0]
                available[state[r]] = True
                row_logp = torch.where(available, self._likelihood[r], torch.tensor(-torch.inf, device=device))
                row_logp -= row_logp.logsumexp(dim=-1, keepdim=True)

                new_col, _ = self.sample_joint(row_logp.exp())
                assert available[new_col]
                state[r] = new_col[0]
                if new_col[0] < n_cols - 1:
                    available[state[r]] = False

            # ----- 4.4  Record sample if we are in the cold chain ----------
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0 and t_idx == 0:
                samples[stored] = state
                log_likes[stored] = self._likelihood[torch.arange(n_rows, device=device), state].sum()
                stored += 1

            sweep += 1

        median_delta = torch.zeros((len(temperatures)))
        for i,idx in enumerate(temperatures):
            median_delta[i] = torch.median(torch.tensor(deltas_v_temperature[i]))

        return samples, log_likes

    def _compute_forward_loss(self,
                              weights: torch.tensor,
                              bias: torch.tensor,
                              gibbs_sample: torch.tensor):
        n_samples, n_rows, n_cols = (gibbs_sample.shape[0],gibbs_sample.shape[1],weights.shape[0])

        row_mask = torch.zeros((n_rows,),dtype=torch.bool)
        row_indexes = torch.arange(n_rows)
        sample_indexes = torch.arange(n_samples)
        score = torch.zeros((1,),dtype=torch.float)
        availableCols = torch.ones((n_samples,self._likelihood.shape[1],), dtype=torch.bool)
        for row in range(n_rows):
            if row > 0:
                local_score = weights[:,gibbs_sample[sample_indexes.unsqueeze(-1),row_indexes[row_mask]],row-1].logsumexp(dim=-1).T + bias
            else:
                local_score = bias.unsqueeze(0).expand(n_samples,-1)

            local_score = local_score.masked_fill(~availableCols,-torch.inf)
            local_score = local_score - local_score.logsumexp(dim=-1, keepdim=True)

            score = score + local_score[sample_indexes,gibbs_sample[sample_indexes,row]].sum(dim=-1)
            row_mask[row] = True
            mutable_col_mask  = gibbs_sample[sample_indexes,row] < n_cols-1
            availableCols[sample_indexes[mutable_col_mask],gibbs_sample[sample_indexes,row][mutable_col_mask]]= False
        #
        norm_bias = bias - bias.logsumexp(dim=-1, keepdim=True)
        return -score + (norm_bias.exp()*norm_bias).sum()
        #score is

    def _train_auto_regression(self,
                               gibbs_sample: torch.tensor):
        self._weights = torch.full((self._likelihood.shape[1],
                                     self._likelihood.shape[1],
                                    gibbs_sample.shape[1]-1),0, dtype=torch.float)

        self._weights.diagonal(offset=0,dim1=0,dim2=1).fill_diagonal_(-1000).requires_grad_(True)
        self._bias = torch.zeros((self._likelihood.shape[1]), dtype=torch.float, requires_grad=True)
        self._compute_forward_loss(self._weights, self._bias, gibbs_sample)

        optimizer = torch.optim.Adam([self._weights,self._bias], lr=1E0)
        nsteps = 1000000
        old_loss = torch.inf
        for step in range(nsteps):
            optimizer.zero_grad()
            loss = self._compute_forward_loss(self._weights, self._bias, gibbs_sample)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f'Step {step} Loss: {loss.item():.4f}')
                #print(self._bias)
            if abs(loss.item() < old_loss) < 1E-3:
                break
            old_loss = loss.item()

        print("Done")
    def _draw_auto_regression(self):
        sampled_columns = torch.zeros((self._likelihood.shape[0],),dtype=torch.int32)
        row_indexes = torch.arange(0,self._likelihood.shape[0])
        row_mask = torch.zeros((self._likelihood.shape[0],),dtype=torch.bool)
        probability = 1.0
        availableCols = torch.ones((self._likelihood.shape[1],),dtype=torch.bool)
        for row in row_indexes:
            if row > 0:
                logits = self._weights[:,sampled_columns[row_mask],row-1].detach().logsumexp(dim=-1) + self._bias.detach()
            else:
                logits = self._bias.detach().clone()

            logits = torch.where(availableCols, logits, -torch.inf)
            logits -= logits.logsumexp(dim=-1, keepdim=True)
            logits.exp_()
            col = torch.multinomial(logits,num_samples=1,replacement=False).squeeze()
            probability *= logits[col]
            sampled_columns[row] = col
            if col < self._likelihood.shape[1] - 1:
                availableCols[col] = False
            row_mask[row] = True
            assert(math.isfinite(probability))
        #
        return tuple(sampled_columns.tolist()), probability


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
        self._row_cluster_samplers = None
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


    def _compute_row_pair_MI(self, row1_index, row2_index):
        likelihoods = self._base_row_decision_likelihoods_unnormalized[[row1_index,row2_index],:]


        #joint = torch.zeros(self._base_row_decision_likelihoods.shape[1],
        #                    self._base_row_decision_probabilities.shape[1],dtype=torch.float32)
        log_joint,log_x,log_y = compute_joint_row_probability(likelihoods)
        joint = log_joint.exp()
        mi = torch.where(joint > 0,joint*(log_joint - (log_x + log_y)),0).sum()
        entropy = torch.where(joint > 0, -joint*log_joint, 0).sum()
        return mi/entropy

    def compute_MI_distance_matrix(self):
        n_rows = self._base_row_decision_likelihoods_unnormalized.shape[0]
        MI_distance_matrix = torch.zeros((n_rows,n_rows))
        for row_index_1 in range(n_rows):
            for row_index_2 in range(row_index_1+1,n_rows):
               MI_distance_matrix[row_index_1,row_index_2] = self._compute_row_pair_MI(row_index_1,row_index_2)
               MI_distance_matrix[row_index_2,row_index_1] = MI_distance_matrix[row_index_1,row_index_2]
        MI_distance_matrix.fill_diagonal_(1.0)
        MI_distance_matrix = 1 - MI_distance_matrix
        return MI_distance_matrix

    def _get_row_groups(self, max_cluster_size=100):
       MI_distance_matrix = self.compute_MI_distance_matrix()
       for threshold in [ 0.9999,0.999,0.99, 0.5, 0.4,0.2 ]:
            clustering_object = AgglomerativeClustering(n_clusters=None,
                                                   metric='precomputed',
                                                   linkage='single',
                                                   distance_threshold=threshold,
                                                   compute_full_tree=True,
                                                   )
            cluster_ids = torch.from_numpy(clustering_object.fit_predict(MI_distance_matrix))
            unique_clusters, count = torch.unique(cluster_ids,return_counts=True)
            if torch.max(count) <= max_cluster_size:
                break
       return cluster_ids, unique_clusters, count


    def _compute_row_proposals_by_group(self):
        cluster_ids, unique_clusters,count = self._get_row_groups()
        gibbs_clusters = (count > 3).sum()
        print(f"Training n_clusters={len(unique_clusters)}, {gibbs_clusters} require gibbs sampling, {len(unique_clusters) - gibbs_clusters} require direct marginals")
        self._row_cluster_samplers = []
        for i, unique_cluster in tqdm(enumerate(unique_clusters),desc='Computing Cluster Marginals'):
            row_mask = cluster_ids == unique_cluster
            row_indexes = row_mask.nonzero(as_tuple=True)[0]
            self._row_cluster_samplers.append((row_indexes,RowClusterSampler(self._base_row_decision_likelihoods_unnormalized[row_mask,:])))

    #


    def __getNextInSequence(self, sample: torch.tensor, sample_indicies: torch.tensor,
                            availableRows: torch.tensor,
                            availableCols: torch.tensor,
                            availableClusters: torch.tensor,
                            sample_weights: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):


        #Choose cluster
        cluster_index = decision_counter
        for sample_index in sample_indicies:
            cluster_row_indexes, sampler = self._row_cluster_samplers[cluster_index]
            coords, proposal_probability = sampler.draw_sample()
            matched_cols = torch.tensor(list(coords),dtype=torch.int32)

            decision_log[sample_index,decision_counter,0] = cluster_index
            decision_log[sample_index, decision_counter, 2] = proposal_probability.log()
            decision_log[sample_index, decision_counter, 3] = self._base_row_decision_likelihoods_unnormalized[cluster_row_indexes,matched_cols].sum()

            no_matched_cols = matched_cols == availableCols.shape[1]-1
            matched_cols[no_matched_cols] = -1

            sample[sample_index,cluster_row_indexes] = matched_cols


            if availableCols[sample_index,matched_cols[~no_matched_cols]].all() and sample_weights[sample_index].isfinite().all():
                sample_weights[sample_index] += decision_log[sample_index, decision_counter, 3] -   decision_log[sample_index, decision_counter, 2]
            else:
                sample_weights[sample_index] = -torch.inf

            unique, counts = torch.unique(sample[sample_index, :], return_counts=True, dim=-1)
            not_neg1_mask = unique > -1
            assert (counts[not_neg1_mask] == 1).all() or sample_weights[sample_index] == -torch.inf


            #print(f"Proposal_probability: {proposal_probability}")
            #if (matched_cols != cluster_row_indexes).all():
            #    print(f"Alert: poteintial mismatch {coords} != {cluster_row_indexes}, {proposal_probability}")

            availableRows[sample_index,cluster_row_indexes] = False
            availableCols[sample_index,matched_cols[~no_matched_cols]] = False
            availableClusters[sample_index,cluster_index] = False

        assert sample_weights.isfinite().any()




#

    def _resample(self, sample: torch.Tensor,
                  sample_weights: torch.Tensor,
                  availableClusters: torch.tensor,
                  availableRows: torch.Tensor,
                  availableCols: torch.Tensor,
                  decision_log,
                  force_resample=False):

        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        print(f"ESS ratio:, {ess/sample_weights.shape[0]:0.3f}")
        assert ess.isfinite().all()
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
            availableClusters[...,:] =availableClusters[indices,:]
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
        availableClusters = torch.ones(sample_shape+(len(self._row_cluster_samplers),), dtype=torch.bool)
        while availableRows.any():
            try:
                self.__getNextInSequence(sample, sample_indexes, availableRows, availableCols, availableClusters, sample_weights,decision_log,decision_counter)
                self._resample(sample, sample_weights, availableClusters, availableRows,availableCols,decision_log)
            except Exception as e:
                raise SamplingError(f"Error during sampling of matching matrices \n"
                                    f"Step: {decision_counter} of {availableRows.shape[0]} \n"
                                    f"CSP Distribution Parameters: {self.csp_distribution.param} \n"
                                    f"CSP_weight logits: {self._csp_mixture_weights} probits: {(self._csp_mixture_weights-self._csp_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"matching_weight_logits: {self._matching_mixture_weights} probits: {(self._matching_mixture_weights - self._matching_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"missing_weight_logits: {self._missing_mixture_weights} probits: {(self._missing_mixture_weights - self._missing_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e

            decision_counter += 1

        assert not availableClusters.any()
        self._resample(sample, sample_weights, availableClusters, availableRows,availableCols,decision_log,True)
        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        if self._row_cluster_samplers is None:
            self._compute_row_proposals_by_group()
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
