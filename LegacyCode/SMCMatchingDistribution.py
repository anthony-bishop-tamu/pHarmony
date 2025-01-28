import pyro
import torch
import numpy as np
import pyro.distributions.constraints as constraints
def calculatePositionProb(sample, shape):
    positionProbs = torch.zeros(shape, dtype=torch.float32)
    rows = torch.arange(shape[0]).unsqueeze(0).expand(sample.shape[0], shape[0])
    mask = sample.flatten() >= 0
    positionProbs.index_put_((rows.flatten()[mask], sample.flatten()[mask]), torch.tensor([1.0/sample.shape[0]]), accumulate=True)
    unique_elements, counts = torch.unique(sample, return_counts=True, dim=1)
    assert (positionProbs.sum(dim=-1) < 1.0+1E-3).all() and (positionProbs.sum(dim=0) < 1.0+1E-3).all()
    return positionProbs
#
class SMCMatchingDistribution(pyro.distributions.TorchDistribution):
    arg_constraints = { "_matching_logits": constraints.real}
    def __init__(self, matching_logits: torch.tensor,validate_args=None):

        assert(matching_logits.shape[0] >= matching_logits.shape[1])
        assert(matching_logits.shape[2] == 2)

        self._matching_logits = matching_logits - matching_logits.logsumexp(dim=-1, keepdim=True)

        self.eps_float32 = torch.finfo(torch.float32).eps
        self.max_float32 = torch.finfo(torch.float32).max
        self.min_float32 = torch.finfo(torch.float32).min

        super(SMCMatchingDistribution, self).__init__(batch_shape=torch.Size(), validate_args=validate_args)
        self._event_shape = (matching_logits.shape[0],)
        self._batch_shape = torch.Size([])
    #

    def _calculateDecisionMatrix(self):
        #parameter corrected loglikelihoods

        self._base_row_decision_likelihoods= torch.zeros((self._matching_logits.shape[0],
                                                          self._matching_logits.shape[1]+1),dtype=torch.float32)
        self._base_row_decision_likelihoods[:,:] = self._matching_logits[:,:,1].detach().sum(dim=-1).unsqueeze(1)
        self._base_row_decision_likelihoods[:,:-1] += self._matching_logits[:,:,0].detach() - self._matching_logits[:,:,1].detach()

    def __getNextInSequence(self, sample: torch.tensor, sample_indicies: torch.tensor,availableRows: torch.tensor,
                            sample_weights: torch.tensor,
                            row_decision_matrix: torch.tensor,
                            partial_log_likelihood: torch.tensor,
                            decision_log: torch.tensor,
                            decision_counter: int):


        prob_tensor = availableRows.type(torch.float32)
        sampled_rows = torch.multinomial(prob_tensor.type(torch.float32),1).type(torch.int32).squeeze(-1)
        probabilities = row_decision_matrix[sample_indicies,sampled_rows,:] - row_decision_matrix[sample_indicies,sampled_rows,:].logsumexp(dim=-1,keepdim=True)
        matched_columns = torch.multinomial(probabilities.exp(),1).type(torch.int32).squeeze()

        no_matched_columns = matched_columns >= self._matching_logits.shape[1]

        availableRows[sample_indicies,sampled_rows] = False
        row_probabilities = torch.log(prob_tensor[sample_indicies,sampled_rows]/prob_tensor[sample_indicies,sampled_rows])

        decision_log[sample_indicies,decision_counter,0] = sampled_rows.type(torch.float32)
        decision_log[sample_indicies, decision_counter, 2] = row_decision_matrix[sample_indicies, sampled_rows, matched_columns].type(torch.float32)

        decision_log[sample_indicies,decision_counter, 3] += row_probabilities

        sample_weights += row_probabilities

        matched_columns[no_matched_columns] = -1
        decision_log[sample_indicies,decision_counter,1] = matched_columns.type(torch.float32)

        sample[sample_indicies,sampled_rows]=matched_columns

        row_decision_matrix[sample_indicies,sampled_rows,:] = -1*torch.inf
        row_decision_matrix[~no_matched_columns,:,matched_columns[~no_matched_columns]] = -1*torch.inf
        row_decision_matrix[~no_matched_columns,:,:] -= self._matching_logits[:,matched_columns[~no_matched_columns],1].transpose(0,1).unsqueeze(-1).detach()

    #
    def _resample(self, sample: torch.Tensor, sample_weights: torch.Tensor,
                  row_decision_matrix: torch.tensor,
                  partial_log_likelihoods: torch.Tensor,
                  availableRows: torch.Tensor,
                  decision_log, forceResample = False):
        normalized_weights = (sample_weights - sample_weights.logsumexp(dim=-1, keepdim=True)).exp()
        ess = 1.0/torch.pow(normalized_weights,2).sum()
        nsamples = np.prod(sample.shape[0:-1])
        if ess < nsamples*0.5 or forceResample:
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
            decision_log[...,...] =decision_log[indices,...]
        else:
            return
    #
    def _sample(self, sample_shape=torch.Size()) -> torch.tensor:
        availableRows = torch.ones(sample_shape+(self._matching_logits.shape[0],), dtype=torch.bool)
        sample = torch.full(sample_shape+self._event_shape, -2, dtype=torch.int32)
        sample_weights = torch.zeros(sample_shape, dtype=torch.float32)
        sample_indexes = torch.unique(torch.nonzero(torch.ones_like(sample))[:, :-1])
        self._calculateDecisionMatrix()

        row_decision_matrix = torch.zeros(sample_shape+(self._matching_logits.shape[0],self._matching_logits.shape[1]+1), dtype=torch.float32)
        row_decision_matrix[...,:,:] = self._base_row_decision_likelihoods[:,:]
        partial_log_likelihood = torch.zeros(sample_shape,dtype=torch.float32)
        decision_log = torch.full(sample_shape+(self._matching_logits.shape[0],4),-2, dtype=torch.float32)
        decision_counter = 0
        while availableRows.any():
            self.__getNextInSequence(sample, sample_indexes, availableRows, sample_weights, row_decision_matrix,partial_log_likelihood,decision_log,decision_counter)
            self._resample(sample, sample_weights,row_decision_matrix,partial_log_likelihood,availableRows,decision_log)
            decision_counter += 1

        self._resample(sample,sample_weights,row_decision_matrix,partial_log_likelihood,availableRows,decision_log,forceResample=True)
        return sample

    def sample(self,sample_shape=torch.Size()) -> torch.tensor:
        sample = self._sample(sample_shape)
        return sample

    def log_prob(self, sample, sample_shape=torch.Size([])):
        #sample_weights should be logits
        self._calculateDecisionMatrix()
        matched_mask = sample != -1
        match_rows = torch.nonzero(matched_mask)
        match_columns = sample[matched_mask]
        nomatch_rows = torch.nonzero(~matched_mask)
        nomatch_columns = sample[~matched_mask]
        log_prob = self._matching_logits[nomatch_rows[..., 1], nomatch_columns,1].sum() + \
                   self._matching_logits[match_rows[..., 1], match_columns,0].sum()

        return log_prob

    @property
    def logits(self) -> torch.tensor:
        return self._matching_logits

#
if __name__ == '__main__':
    dim = 100
    logits = torch.rand((dim,dim,2),dtype=torch.float32)
    logits[:,:,0] += torch.eye(dim,dtype=torch.float32)*20

    dist = SMCMatchingDistribution(logits)
    samples = dist.sample(torch.Size([1000]))
    print(samples)
    positionProb = calculatePositionProb(samples,logits.shape[:-1]).numpy()
    log_prob = dist.log_prob(samples)