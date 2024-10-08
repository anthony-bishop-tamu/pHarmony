import pyro
import torch
class DistancesMM(pyro.distributions.TorchDistribution):
    def __init__(self, chi2_dof: torch.tensor,
                       csp_dist_params:torch.tensor,
                       nomatch_dist_params:torch.tensor,
                       mixtureProbabilities: torch.tensor,
                       validate_args=False):

        assert(chi2_dof.shape == (1,))
        assert(csp_dist_params.shape == (2,))
        assert(nomatch_dist_params.shape == (2,))
        assert(mixtureProbabilities.shape[2] == 3)#

        self._chi2_dof = chi2_dof
        self._csp_dist_params = csp_dist_params
        self._nomatch_dist_params = nomatch_dist_params
        self._mixtureProbabilities = mixtureProbabilities

        self._chi2_dist = torch.distributions.Chi2(chi2_dof)
        self._csp_dist = torch.distributions.Weibull(concentration=self._csp_dist_params[0],
                                                     scale=self._csp_dist_params[1])
        self._nomatch_dist = torch.distributions.Weibull(concentration=self._nomatch_dist_params[0],
                                                         scale=self._nomatch_dist_params[1])
        super(DistancesMM, self).__init__(batch_shape=torch.Size(), validate_args=validate_args)

        self._event_shape = mixtureProbabilities.shape[0:2]
        self._batch_shape = torch.Size([])
    def sample(self, sample_shape=torch.Size()):
        expanded = self._mixtureProbabilities.expand(sample_shape+self._mixtureProbabilities.shape)
        assignment_sample = torch.multinomial(expanded.reshape((int(expanded.numel()/expanded.shape[-1]),expanded.shape[-1])), 1, replacement=True).squeeze(-1)

        sample = torch.zeros(sample_shape + self.event_shape, dtype=torch.float32).flatten()
        chi2_choice = assignment_sample == 0
        csp_choice = assignment_sample == 1
        nomatch_choice = assignment_sample == 2
        sample[chi2_choice] = self._chi2_dist.sample(sample[chi2_choice].shape).squeeze(-1)
        sample[csp_choice] = self._csp_dist.sample(sample[csp_choice].shape)
        sample[nomatch_choice] = self._nomatch_dist.sample(sample[nomatch_choice].shape)

        sample=sample.reshape(sample_shape+self.event_shape)
        return sample
    #
    def log_prob(self, sample, sample_shape=torch.Size([])):
        return (self._chi2_dist.log_prob(sample)*self._mixtureProbabilities[:,:,0] +
        self._csp_dist.log_prob(sample)*self._mixtureProbabilities[:,:,1] +
        self._nomatch_dist.log_prob(sample)*self._mixtureProbabilities[:,:,2]).sum()


if __name__ == '__main__':
    dof =torch.tensor([2],dtype=torch.int32)
    csp_params = torch.tensor([5.0,20],dtype=torch.float32)
    nomatch_params = torch.tensor([1.9,1000],dtype=torch.float32)
    mixture_probabilities = torch.rand((10,8,3))
    mixture_probabilities = (mixture_probabilities - mixture_probabilities.logsumexp(dim=2,keepdim=True)).exp()
    dist = distancesMM(dof,csp_params,nomatch_params,mixture_probabilities)
    sample = dist.sample(torch.Size([10]))
    log_prob = dist.log_prob(sample)
    print(log_prob)