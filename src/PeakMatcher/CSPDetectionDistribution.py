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



def validateSample(sample: torch.tensor, availableCols: torch.tensor):
    for i in range(sample.shape[0]):
        unique, counts = torch.unique(sample[i], return_counts=True)
        assert (sample[i] >= -1).all()
        cols = sample[i][sample[i] > -1]
        assert not availableCols[i,cols].any()
        not_used = torch.nonzero(availableCols[i,:])
        assert not_used.tolist() not in sample[i].tolist()
        assert ~((unique[counts > 1] > -1).any())
class AutoRegressionModel:
    def __init__(self, shape: tuple, sample: torch.tensor, regularization: float):
        self._device='cpu'
        self._shape = shape
        self._alpha= regularization
        self._bias = torch.zeros(self._shape[1],dtype=torch.float,requires_grad=True,device=self._device)
        self._weights = torch.zeros((self._shape[1],self._shape[1],self._shape[0]),dtype=torch.float,device=self._device,requires_grad=True)

    def _compute_forward_loss_vec(self, batch_size: int = 30):
        R, C = self._shape  # rows, cols
        B = batch_size
        dev = self._device

        # --- 1. sample a mini-batch ------------------------------------------------
        sub = self._sample[torch.randperm(self._sample.size(0), device=dev)[:B]]  # [B, R]
        #
        # sub[b, r]  = column chosen by sample b at row r,   −1 means “none yet”

        # --- 2. one-hot encode every choice & build "available columns" mask -------
        onehot = F.one_hot(sub, num_classes=C)  # [B, R, C], float32
        chosen_up_to_r = onehot.cumsum(dim=1).clamp_max(1).bool()  # True if column was ever chosen ≤ r
        avail_mask = ~chosen_up_to_r  # [B, R, C]
        avail_mask[:,:,-1] = True

        # --- 3. pre-compute α·row_sums for **all** rows in one go ------------------
        #   row_sums_mat[r, :] = Σ_{c2, r' < r} w[c1, c2, r']
        row_sums_mat = (
            self._weights.sum(dim=1)  # -> [C, R]  (sum over 2nd col dim)
            .cumsum(dim=-1)  # cumulative over rows
            .transpose(0, 1)  # -> [R, C]
        )  # shape [R, C]
        alpha_term = (self._alpha * row_sums_mat)  # [R, C]

        # --- 4. second term: (1-α)·log∑exp w[:, prev_col, r-1] ----------------------
        # Pad a dummy “previous column” for r = 0 so gather() works.
        prev_cols = torch.cat(
            [torch.full((B, 1), 0, dtype=torch.long, device=dev),  # never read for r=0
             sub[:, :-1].clamp_min(0)], dim=1)  # [B, R]

        # Build an index tensor so we can gather in one shot:
        #   self._weights has shape [C, C, R]
        #   we need w[:, prev_cols[b,r], r]  for every b, r
        idx_r = torch.arange(R, device=dev)  # [R]
        W = self._weights  # [C, C, R]
        W_by_r = W.permute(2, 0, 1)  # [R, C_first, C_second]

        # Gather the slice along dim=2 (= second-column dim)
        gather_idx = prev_cols.unsqueeze(1)  # [B, 1, R]
        term2 = torch.take_along_dim(
            W_by_r.unsqueeze(0).expand(B, -1, -1, -1),  # [B, R, C_first, C_second]
            gather_idx.unsqueeze(2), dim=3  # gather along C_second
        ).squeeze(3)  # -> [B, R, C_first]
        term2 = (1.0 - self._alpha) * term2.logsumexp(dim=2)  # log∑exp over C_first
        term2 = term2  # [B, R]

        # --- 5. assemble logits for every (b, r, c) --------------------------------
        bias = self._bias  # [C]
        logits = alpha_term.unsqueeze(0) + term2.unsqueeze(2) + bias  # [B, R, C]

        # mask out columns that are no longer available
        logits = logits.masked_fill(~avail_mask, -torch.inf)

        # --- 6. compute log-probabilities & accumulated loss -----------------------
        log_probs = logits - logits.logsumexp(dim=2, keepdim=True)  # [B, R, C]
        picked_log_p = log_probs.gather(2, sub.unsqueeze(2)).squeeze(2)  # [B, R]
        score = -picked_log_p.sum(dim=1).mean()  # scalar

        return score

    def _compute_forward_loss(self,sample,batch_size=30,unormalized_likelihoods = None):
        n_rows = self._shape[0]
        n_cols = self._shape[1]
        n_samples = batch_size

        sub_sample = sample[torch.randperm(sample.shape[0],device=self._device)[:batch_size],:]



        row_mask = torch.zeros((n_rows,), dtype=torch.bool,device=self._device)
        row_indexes = torch.arange(n_rows,device=self._device)
        sample_indexes = torch.arange(n_samples,device=self._device)
        score = torch.zeros((1,), dtype=torch.float,device=self._device)
        weights = torch.zeros((n_samples,), dtype=torch.float, device=self._device)
        availableCols = torch.ones((n_samples, self._shape[1]), dtype=torch.bool,device=self._device)
        for row in range(n_rows):
            cols = sub_sample[sample_indexes.unsqueeze(-1), row_indexes[row_mask]]
            row_sums = self._weights[:,:,:row].sum(dim=(-1, -2))

            if row > 0:
                logits = self._alpha*row_sums.unsqueeze(0).expand(n_samples,n_cols) \
                     +(1-self._alpha)*self._weights[:,cols,row-1].logsumexp(dim=-1).T + self._bias.unsqueeze(0)
            else:
                logits = self._bias.unsqueeze(0).expand(n_samples,-1).clone()

            cols = sub_sample[sample_indexes, row]
            logits = logits.masked_fill(~availableCols,-torch.inf)
            log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
            score = score + -log_probs[sample_indexes,cols].sum()
            if unormalized_likelihoods is not None:
                local_weight = unormalized_likelihoods[row,cols] - logits[sample_indexes,cols]
                weights = weights + local_weight

            mutable_col_mask = cols < n_cols -1
            availableCols[sample_indexes[mutable_col_mask], cols[mutable_col_mask]] = False
            row_mask[row]=True
        #
        norm_weights = (weights - weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(norm_weights,2).sum()
        return score/n_samples, ess/n_samples
        # score is

    def _train_auto_regression(self,sample,scaling_factor=20.0,unormalized_likelihoods=None):

        sample[sample == -1] = self._shape[1]-1
        sample.to(self._device)
        batch_size = int(sample.shape[0])
        optimizer = torch.optim.AdamW([self._weights,self._bias], lr=1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3
        )
        nsteps = 100
        old_loss = torch.inf
        for step in range(nsteps):
            optimizer.zero_grad(set_to_none=True)
            score, ess = self._compute_forward_loss(sample,batch_size=batch_size,unormalized_likelihoods=unormalized_likelihoods)
            loss = score + scaling_factor*(1-ess)
            loss.backward()
            #torch.nn.utils.clip_grad_value_(self._weights,5.0)
            #torch.nn.utils.clip_grad_value_(self._bias,5.0)
            optimizer.step()
            if step % 1 == 0:
                print(f'Step {step} Loss: {loss.item():.4f}, Score: {score.item():.4f}, ESS: {ess.item():.4f} LR: {optimizer.param_groups[0]["lr"]:.4f}')
                # print(self._bias)
            if abs(loss.item() - old_loss) < 1E-3:
                break
            if loss.item() > old_loss:
                #batch_size = int(batch_size*1.2)
                if batch_size > sample.shape[0]:
                    batch_size = sample.shape[0]
            old_loss = loss.item()
            scheduler.step(loss.detach())

        print("Done")

    def _getNextInSequence(self,
                           unnormalized_likelihoods: torch.tensor,
                           sample: torch.tensor,
                           sample_indexes: torch.tensor,
                           availableRows: torch.tensor,
                           availableCols: torch.tensor,
                           decision_log: torch.tensor,
                           decision_counter: int):

        row = decision_counter
        sample_size = sample.shape[0]
        n_cols = self._shape[1]
        row_mask = ~availableRows[0] #All samples sample rows in the same order with this method

        row_indexes = torch.arange(self._shape[0],device=self._device)
        cols = sample[:, row_indexes[row_mask]]
        row_sums = self._weights[:, :, :row].sum(dim=(-1, -2))

        if row > 0:
            logits = self._alpha * row_sums.unsqueeze(0).expand(sample_size, n_cols) \
                     + (1 - self._alpha) * self._weights[:, cols, row - 1].logsumexp(dim=-1).T + self._bias.unsqueeze(0)
        else:
            logits = self._bias.unsqueeze(0).expand(sample_size, -1).clone()

        logits = torch.where(availableCols, logits, -torch.inf)
        logits -= logits.logsumexp(dim=-1, keepdim=True)
        col = torch.multinomial(logits.exp(), num_samples=1, replacement=False).squeeze().type(torch.int32)
        assert availableCols[sample_indexes, col].all()
        assert logits[sample_indexes, col].isfinite().all()

        sample_weights = unnormalized_likelihoods[row,col] - logits[sample_indexes, col]
        decision_log[sample_indexes, row, 0]= row
        decision_log[sample_indexes, row, 1] = col.type(torch.float64)
        decision_log[sample_indexes, row, 2] = unnormalized_likelihoods[row,col]
        decision_log[sample_indexes, row, 3] = logits[sample_indexes, col].type(torch.float64)

        true_col_mask = col < n_cols - 1
        col[~true_col_mask] = -1
        sample[sample_indexes, row] = col

        availableCols[sample_indexes[true_col_mask], col[true_col_mask]] = False
        availableRows[sample_indexes, row] = False

        row_mask[row] = True
        #

        return sample_weights


    def _draw_auto_regression(self,sample_size):
        n_rows = self._shape[0]
        n_cols = self._shape[1]
        sample = torch.full((sample_size,n_rows),-2,dtype=torch.int32,device=self._device)
        row_indexes = torch.arange(0, n_rows,dtype=torch.int32,device=self._device)
        sample_indexes = torch.arange(sample_size,device=self._device)
        row_mask = torch.zeros((n_rows,), dtype=torch.bool,device=self._device)
        availableCols = torch.ones((sample_size,n_cols,), dtype=torch.bool, device=self._device)
        proposal_weights = torch.zeros((sample_size,), dtype=torch.float,device=self._device)
        for row in row_indexes:
            cols = sample[:,row_indexes[row_mask]]
            row_sums = self._weights[:,:,:row].sum(dim=(-1, -2))

            if row > 0:
                logits = self._alpha*row_sums.unsqueeze(0).expand(sample_size,n_cols) \
                     +(1-self._alpha)*self._weights[:,cols,row-1].logsumexp(dim=-1).T + self._bias.unsqueeze(0)
            else:
                logits = self._bias.unsqueeze(0).expand(sample_size,-1).clone()


            logits = torch.where(availableCols, logits, -torch.inf)
            logits -= logits.logsumexp(dim=-1, keepdim=True)
            col = torch.multinomial(logits.exp(), num_samples=1, replacement=False).squeeze().type(torch.int32)
            assert availableCols[sample_indexes, col].all()
            proposal_weights += logits[sample_indexes,col]

            true_col_mask = col < n_cols - 1
            col[~true_col_mask] = -1
            sample[sample_indexes,row] = col

            availableCols[sample_indexes[true_col_mask],col[true_col_mask]] = False

            row_mask[row] = True
            assert (proposal_weights.isfinite()).all()
        #
        return sample, proposal_weights


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

    def __getNextInSequence_Plackett_Lucette(self,
                                             sample: torch.tensor,
                                             sample_indicies: torch.tensor,
                                             availableRows: torch.tensor,
                                             availableCols: torch.tensor,
                                             decision_log: torch.tensor,
                                             decision_counter: int):

        row_decision_matrix = self._base_row_decision_probabilities.detach()
        sample_weights = torch.zeros((sample.shape[0]),dtype=torch.float64)
        # row_log_evidence[...] = 0
        # prob_tensor = availableRows.type(torch.float64)
        # row_index_list = torch.nonzero(availableRows.type(torch.float64), as_tuple=True)[1].reshape(sample.shape[0],-1)
        with record_function("Row_Sampling"):
            row_probs = availableRows.type(torch.float32)
            sampled_rows = torch.multinomial(row_probs, num_samples=1, replacement=True).type(torch.int32).squeeze(1)
            # print(sampled_rows.unique().shape)
            # sampled_rows = row_index_list[torch.arange(sample.shape[0]),sampled_rows]

        with record_function("Column_Sampling"):
            probabilities = row_decision_matrix[sampled_rows, :] * availableCols
            probabilities /= probabilities.sum(dim=-1, keepdim=True)
            matched_columns = torch.multinomial(probabilities, 1, replacement=True).type(torch.int32).squeeze()

        no_matched_columns = matched_columns >= self._distances.shape[1]

        # availableRows[sample_indicies, sampled_rows] = False

        decision_log[sample_indicies, decision_counter, 0] = sampled_rows.type(torch.float64)
        decision_log[sample_indicies, decision_counter, 2] = row_decision_matrix[sampled_rows, matched_columns].type(
            torch.float64)  # +row_probabilities
        decision_log[sample_indicies, decision_counter, 3] = probabilities[sample_indicies, matched_columns].type(
            torch.float64)

        sample_weights[...]=1
        assert sample_weights.isfinite().all()

        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies, decision_counter, 1] = matched_columns.type(torch.float64)

        sample[sample_indicies, sampled_rows] = matched_columns
        availableRows[sample_indicies, sampled_rows] = False
        availableCols[sample_indicies[~no_matched_columns], matched_columns[~no_matched_columns]] = False

        return sample_weights

    def _resample(self, sample: torch.Tensor,
                  sample_weights: torch.Tensor,
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
            decision_log[...,...] =decision_log[indices,...]
            return ess/nsamples < 0.25
        else:
            return False
    #


    def calculatePairwiseMarginalProbabilityDistributions_2D(self,
                                                     likelihood_matrix: torch.tensor,
                                                     row_indexes: torch.tensor):
        n_cols = likelihood_matrix.shape[-1]
        marginal_probabilities = likelihood_matrix[row_indexes,:].unsqueeze(1).unsqueeze(-1) + likelihood_matrix[row_indexes,:].unsqueeze(0).unsqueeze(-2)
        #available_mask = availableCols.unsqueeze(-1) & availableCols.unsqueeze(1)
        #d = available_mask.diagonal(offset=0,dim1=1,dim2=2)
        #d.zero_()
        #available_mask[:,n_cols-1,n_cols-1] = 1

        #marginal_probabilities.masked_fill_(~available_mask, -torch.inf)

        marginal_probabilities -= marginal_probabilities.logsumexp(dim=(-1,-2), keepdim=True)
        marginal_probabilities.exp_()
        return marginal_probabilities


    def sampleRandomVariablePairFromMarginal(self,marginals: torch.tensor,
                                             availableCols: torch.tensor,
                                             cluster_row_indexes: torch.tensor,
                                             x_row_index: torch.tensor,
                                             y_row_index: torch.tensor):

        n_cols = availableCols.shape[-1]
        mapped_x = torch.searchsorted(cluster_row_indexes, x_row_index)
        mapped_y = torch.searchsorted(cluster_row_indexes, y_row_index)

        available_mask = availableCols.unsqueeze(-1) & availableCols.unsqueeze(1)
        d = available_mask.diagonal(offset=0,dim1=1,dim2=2)
        d.zero_()
        available_mask[:,n_cols-1,n_cols-1] = 1

        marginals = marginals[mapped_x,mapped_y].reshape(available_mask.shape[0],-1)
        sampled_indexes = torch.multinomial(marginals*available_mask.reshape(available_mask.shape[0],-1), num_samples=1, replacement=True).type(torch.int32).squeeze()
        col1, col2 = torch.unravel_index(sampled_indexes, available_mask.shape[1:])

        return col1, col2


    def compute_distance_matrix(self, sample: torch.Tensor, n_cols: int, block: int = 64) -> torch.Tensor:
        S, R = sample.shape
        C = n_cols
        device = sample.device
        sample[sample == -1] = n_cols-1
        X = F.one_hot(sample.to(torch.long), num_classes=C).to(torch.float32)  # (S,R,C)
        row_p = X.sum(0) / float(S)  # (R,C)
        log_row_p = torch.zeros_like(row_p)
        m = row_p > 0
        log_row_p[m] = row_p[m].log()

        dist = torch.empty((R, R), dtype=torch.float32, device=device)
        for i0 in range(0, R, block):
            i1 = min(R, i0 + block)
            A = X[:, i0:i1, :]  # (S, Bi, C)
            log_pi = log_row_p[i0:i1, :][:, None, :, None]  # (Bi,1,C,1)
            for j0 in range(0, R, block):
                j1 = min(R, j0 + block)
                B = X[:, j0:j1, :]  # (S, Bj, C)
                log_pj = log_row_p[j0:j1, :][None, :, None, :]  # (1,Bj,1,C)

                p = torch.einsum('sac,sbd->abcd', A, B) / float(S)  # (Bi,Bj,C,C)
                log_p = torch.zeros_like(p);
                mp = p > 0;
                log_p[mp] = p[mp].log()

                mi = (p * (log_p - (log_pi + log_pj))).sum(dim=(2, 3))
                H = -(p * log_p).sum(dim=(2, 3))
                dist[i0:i1, j0:j1] = torch.where(H == 0, torch.ones_like(H), 1.0 - mi / H)

        dist = 0.5 * (dist + dist.T)
        dist.fill_diagonal_(0.0)
        sample[sample == n_cols-1] = -1
        return dist
    def generateRowClusters(self,sample: torch.Tensor, n_cols: int):
        distance_matrix = self.compute_distance_matrix(sample,n_cols,100)
        cluster_indexes = torch.from_numpy(AgglomerativeClustering(n_clusters=None,metric='precomputed',linkage='single',distance_threshold=0.90).fit_predict(distance_matrix))
        unique_clusters,counts = torch.unique(cluster_indexes,return_counts=True)
        cluster_masks = []
        for i,idx in enumerate(unique_clusters):
            if counts[i] > 1:
                cluster_masks.append(cluster_indexes == idx)

        return cluster_masks
    #

    def gibbs_sample(self, sample: torch.Tensor,
                   sample_indicies: torch.tensor,
                   availableCols: torch.tensor,
                   n_sweeps: int):

        n_samples = sample.shape[0]
        n_cols = availableCols.shape[1]

        cluster_masks = self.generateRowClusters(sample,n_cols)

        for cluster in tqdm(range(len(cluster_masks)),desc='Gibbs sample over clusters'):

            cluster_row_index_order = torch.multinomial(cluster_masks[cluster].type(torch.float32).unsqueeze(0).expand(n_samples*n_sweeps,-1),replacement=False,num_samples=cluster_masks[cluster].sum()).type(torch.int32).squeeze()
            cluster_row_index_order = cluster_row_index_order.reshape(n_sweeps,n_samples,-1)
            cluster_row_indexes = torch.nonzero(cluster_masks[cluster]).squeeze()
            marginals = self.calculatePairwiseMarginalProbabilityDistributions_2D(self._base_row_decision_likelihoods_unnormalized,cluster_row_indexes)

            for sweep in tqdm(range(n_sweeps),desc="Gibbs sweeps"):
                rand_order = cluster_row_index_order[sweep,...]
                for i in range(rand_order.shape[1]):
                    row_1 = rand_order[:,i]
                    if i+1 < rand_order.shape[1]: #pairs
                        row_2 = rand_order[:,i+1]


                        availableCols[sample_indicies, sample[sample_indicies,row_1]] = True
                        availableCols[sample_indicies, sample[sample_indicies,row_2]] = True

                        sample[sample_indicies, row_1] = -2
                        sample[sample_indicies, row_2] = -2



                        #choose new state
                        col1,col2 = self.sampleRandomVariablePairFromMarginal(marginals,
                                                                              availableCols,
                                                                              cluster_row_indexes,
                                                                              row_1,
                                                                              row_2)


                        assert ((col1 != col2) | (col1 == n_cols-1)).all()
                        assert (availableCols[sample_indicies,col1[sample_indicies]]).all()
                        assert (availableCols[sample_indicies,col2[sample_indicies]]).all()
                        col1[col1 == (n_cols-1)] = -1
                        col2[col2 == (n_cols-1)] = -1

                        sample[sample_indicies, row_1] = col1.type(torch.int32)
                        sample[sample_indicies, row_2] = col2.type(torch.int32)


                        availableCols[sample_indicies, col1] = False
                        availableCols[sample_indicies, col2] = False
                        availableCols[:, -1] = True
                        #validateSample(sample,availableCols)
                    else: #single
                        #make available the column
                        old_cols = sample[sample_indicies,row_1].clone()
                        #old_available_cols_1 = availableCols.clone()
                        availableCols[sample_indicies,old_cols ] = True
                        #old_available_cols_2 = availableCols.clone()
                        sample[sample_indicies, row_1] = -2

                        #calculate the probability
                        marginal_probability_distributions_1D = self._base_row_decision_likelihoods_unnormalized[row_1, :].clone()
                        marginal_probability_distributions_1D.masked_fill_(~availableCols,-torch.inf)
                        marginal_probability_distributions_1D -= marginal_probability_distributions_1D.logsumexp(dim=-1, keepdim=True)
                        marginal_probability_distributions_1D.exp_()

                        #grab the new column
                        col1 = torch.multinomial(marginal_probability_distributions_1D, num_samples=1, replacement=False).type(
                            torch.int32).squeeze()
                        col1[col1 == (n_cols - 1)] = -1

                        #implement the selection
                        sample[sample_indicies, row_1] = col1.type(torch.int32)
                        availableCols[sample_indicies, col1] = False
                        availableCols[:, -1] = True
                       # validateSample(sample, availableCols)




    def _sample(self, sampler, sample_shape=torch.Size(), allow_resample=True) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._distances.shape[0],), dtype=torch.bool)
        availableCols = torch.ones(sample_shape+(self._distances.shape[1]+1,), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float64)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionLogLikelihood()
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float64)
        decision_counter = 0
        gibbs_sample = False
        while availableRows.any():
            try:
                step_weights = sampler(sample, sample_indexes, availableRows, availableCols, decision_log,decision_counter)
                sample_weights = sample_weights+step_weights

                if allow_resample:
                    self._resample(sample, sample_weights, availableRows,availableCols,decision_log)
            except Exception as e:
                raise SamplingError(f"Error during sampling of matching matrices \n"
                                    f"Step: {decision_counter} of {availableRows.shape[0]} \n"
                                    f"CSP Distribution Parameters: {self.csp_distribution.param} \n"
                                    f"CSP_weight logits: {self._csp_mixture_weights} probits: {(self._csp_mixture_weights-self._csp_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"matching_weight_logits: {self._matching_mixture_weights} probits: {(self._matching_mixture_weights - self._matching_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"missing_weight_logits: {self._missing_mixture_weights} probits: {(self._missing_mixture_weights - self._missing_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e

            decision_counter += 1

        if allow_resample:
            self._resample(sample, sample_weights, availableRows,availableCols,decision_log,True)
        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample, sample_indexes, availableRows,availableCols, decision_log, gibbs_sample



    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        sampler = lambda sample, sample_indicies, availableRows, availableCols, decision_log, decision_counter: \
            self._autoregression._getNextInSequence(self._base_row_decision_likelihoods_unnormalized,
                                                    sample,
                                                    sample_indicies,
                                                    availableRows,
                                                    availableCols,
                                                    decision_log,
                                                    decision_counter)
        if self._autoregression is None:
            sample, sample_indexes, availableRows, availableCols, decision_log, gibbs_sample = self._sample(
                self.__getNextInSequence_Plackett_Lucette, (1000,),allow_resample=True)
            self._autoregression = AutoRegressionModel(self._base_row_decision_likelihoods_unnormalized.shape, sample,
                                                       0)


            for i in range(5):
                self.gibbs_sample(sample, sample_indexes, availableCols, n_sweeps=100)
                self._autoregression._train_auto_regression(sample,
                                                            unormalized_likelihoods=self._base_row_decision_likelihoods_unnormalized)
                sample, sample_indexes, availableRows, availableCols, decision_log, gibbs_sample = self._sample(sampler,
                                                                                                                sample.shape[0:1],allow_resample=True)

        #
        sample, sample_indexes, availableRows, availableCols, decision_log, gibbs_sample  = self._sample(sampler,sample_shape)
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
