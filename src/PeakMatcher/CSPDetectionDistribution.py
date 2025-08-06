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
class AutoRegressionModel:
    def __init__(self, shape: tuple, sample: torch.tensor, regularization: float):
        self._device='cpu'
        self._shape = shape
        self._sample = sample.clone().to(self._device)
        self._alpha= regularization
        self._bias = torch.zeros(self._shape[1],dtype=torch.float,requires_grad=True,device=self._device)
        self._weights = torch.zeros((self._shape[1],self._shape[1],self._shape[0]),dtype=torch.float,device=self._device,requires_grad=True)
        self._train_auto_regression()



    def _compute_forward_loss(self,batch_size=30):
        n_rows = self._shape[0]
        n_cols = self._shape[1]
        n_samples = batch_size

        sub_sample = self._sample[torch.randperm(self._sample.shape[0],device=self._device)[:batch_size],:]



        row_mask = torch.zeros((n_rows,), dtype=torch.bool,device=self._device)
        row_indexes = torch.arange(n_rows,device=self._device)
        sample_indexes = torch.arange(n_samples,device=self._device)
        score = torch.zeros((1,), dtype=torch.float,device=self._device)
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
            mutable_col_mask = cols < n_cols - 1
            availableCols[sample_indexes[mutable_col_mask], cols[mutable_col_mask]] = False
            row_mask[row]=True
        #
        return score
        # score is

    def _train_auto_regression(self):

        batch_size = 30
        optimizer = torch.optim.Adam([self._weights, self._bias], lr=1)
        nsteps = 100
        old_loss = torch.inf
        for step in range(nsteps):
            optimizer.zero_grad()
            loss = self._compute_forward_loss(batch_size=batch_size)
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                print(f'Step {step} Loss: {loss.item():.4f}')
                # print(self._bias)
            if abs(loss.item() - old_loss) < 1E-3:
                break
            if loss.item() > old_loss:
                #batch_size = int(batch_size*1.2)
                if batch_size > self._sample.shape[0]:
                    batch_size = self._sample.shape[0]
            old_loss = loss.item()

        print("Done")


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
        self._base_row_decision_probabilities = (self._base_row_decision_likelihoods_unnormalized - self._base_row_decision_likelihoods_unnormalized.logsumexp(dim=-1,keepdim=True)).exp()

    def __getNextInSequence_Plackett_Lucette(self, sample: torch.tensor, sample_indicies: torch.tensor,
                            availableRows: torch.tensor,
                            availableCols: torch.tensor,
                            sample_weights: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):

        row_decision_matrix = self._base_row_decision_probabilities.detach()

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

        sample_weights += 1
        assert sample_weights.isfinite().all()

        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies, decision_counter, 1] = matched_columns.type(torch.float64)

        sample[sample_indicies, sampled_rows] = matched_columns
        availableRows[sample_indicies, sampled_rows] = False
        availableCols[sample_indicies[~no_matched_columns], matched_columns[~no_matched_columns]] = False

    def __getNextInSequence(self, sample: torch.tensor, sample_indicies: torch.tensor,
                            availableRows: torch.tensor,
                            availableCols: torch.tensor,
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

        assert sample_weights.isfinite().any()

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
        else:
            return
    #
    def _sample(self, sampler, sample_shape=torch.Size()) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._distances.shape[0],), dtype=torch.bool)
        availableCols = torch.ones(sample_shape+(self._distances.shape[1]+1,), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float64)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionLogLikelihood()
        decision_log = torch.full(sample_shape+(self._distances.shape[0],4),-2, dtype=torch.float64)
        decision_counter = 0
        while availableRows.any():
            try:
                sampler(sample, sample_indexes, availableRows, availableCols, sample_weights,decision_log,decision_counter)
                self._resample(sample, sample_weights, availableRows,availableCols,decision_log)
            except Exception as e:
                raise SamplingError(f"Error during sampling of matching matrices \n"
                                    f"Step: {decision_counter} of {availableRows.shape[0]} \n"
                                    f"CSP Distribution Parameters: {self.csp_distribution.param} \n"
                                    f"CSP_weight logits: {self._csp_mixture_weights} probits: {(self._csp_mixture_weights-self._csp_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"matching_weight_logits: {self._matching_mixture_weights} probits: {(self._matching_mixture_weights - self._matching_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n"
                                    f"missing_weight_logits: {self._missing_mixture_weights} probits: {(self._missing_mixture_weights - self._missing_mixture_weights.logsumexp(dim=0,keepdim=True)).exp()}\n") from e

            decision_counter += 1

        self._resample(sample, sample_weights, availableRows,availableCols,decision_log,True)
        #assert torch.abs(self.log_prob(sample) - partial_log_likelihood.sum()) <= 1
        return sample

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        sample = self._sample(self.__getNextInSequence_Plackett_Lucette,sample_shape)
        reg = AutoRegressionModel(self._base_row_decision_likelihoods_unnormalized.shape,sample,0)
        sample,weights = reg._draw_auto_regression(100)
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
