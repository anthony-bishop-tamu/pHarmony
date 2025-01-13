import torch
from torch.distributions import constraints
from sklearn.neighbors import KernelDensity
import numpy as np
from KDEpy import FFTKDE,TreeKDE
class Frechet(torch.distributions.Distribution):
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    support = constraints.positive  # Support of the distribution (x > 0)

    def __init__(self, alpha, scale, loc=0, validate_args=None):
        self.alpha = alpha  # Shape parameter (fitted_c from SciPy)
        self.scale = scale  # Scale parameter (fitted_scale from SciPy)
        self.loc = loc
        super(Frechet, self).__init__(torch.Size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        # Generate Weibull samples
        weibull_samples = torch.distributions.Weibull(self.alpha, self.scale).sample(sample_shape)
        # Apply the transformation to get Fréchet samples
        frechet_samples = weibull_samples.reciprocal()  # Shift by loc
        return frechet_samples

    def log_prob(self, x):
        # Log probability of the Fréchet distribution
        z = (x - self.loc) / self.scale
        return torch.log(self.alpha)  - torch.log(self.scale) - (self.alpha + 1) * torch.log(z) - z ** (-self.alpha)

    def median(self):
        return (self.scale/torch.pow(torch.log(torch.tensor([2.0])), 1.0/self.alpha)) + self.loc

    def mode(self):
        return self.scale*torch.pow(self.alpha/(1+self.alpha), 1.0/self.alpha) + self.loc

    @property
    def variance(self):
        if self.alpha < 2.0:
            return torch.tensor([torch.inf])
        else:
            return self.scale*self.scale*(torch.lgamma(1-2.0/self.alpha).exp() - torch.lgamma(1-1.0/self.alpha).exp()**2)
        #

class KDEDensity(torch.distributions.Distribution):
    pass
class DiscreteDistribution(torch.distributions.Distribution):
    def __init__(self, pdf: torch.Tensor, eval_grid: torch.tensor, validate_args=None):
        assert pdf.shape[-1] == eval_grid.shape[0]
        assert eval_grid.dim() == 1
        self._density = pdf
        self._eval_grid = eval_grid
        super(DiscreteDistribution, self).__init__(torch.Size(), validate_args=validate_args)
    def log_prob(self, x: torch.tensor) -> torch.Tensor:
        range = self._eval_grid[-1] - self._eval_grid[0]
        stepSize = range/(self._eval_grid.numel()-1)
        idx = ((x - self._eval_grid[0])/stepSize).round().long()
        if len(self._density.shape) == 2:
            if len(x.shape) == 2:
                return self._density[torch.arange(x.shape[0]).unsqueeze(1),idx]
            elif len(x.shape) == 1:
                return (self._density.exp()/self._density.shape[0]).sum(dim=0).log()[idx]
            else:
                raise NotImplementedError
        elif len(self._density.shape) == 1:
            return self._density[idx]
        else:
            raise NotImplementedError
    @property
    def eval_grid(self) -> torch.Tensor:
        return self._eval_grid
    @property
    def log_density(self) -> torch.tensor:
        return self._density
    #
    def applyPrior(self, prior: torch.distributions.Distribution):
        prior_log_prob = prior.log_prob(self.eval_grid)
        forward_density = self.log_density
        num = (forward_density + prior_log_prob).exp()
        num[num == 0] = torch.finfo(torch.float64).eps
        denom = ((num[...,:-1]+num[...,1:])/2 * torch.diff(self.eval_grid)).sum(dim=-1,keepdim=True).log()
        posterior = num.log() - denom
        newObj = DiscreteDistribution(posterior, self.eval_grid)
        newObj.prior = prior
        return newObj

class LogDiscreteDistribution(DiscreteDistribution):
    def __init__(self, pdf: torch.Tensor, eval_grid: torch.tensor, validate_args=None):
        pdf = (pdf.exp()*eval_grid).log()
        super(LogDiscreteDistribution, self).__init__(pdf,eval_grid.log())
    @property
    def eval_grid(self) -> torch.Tensor:
        return self._eval_grid.exp()
    @property
    def log_density(self) -> torch.tensor:
        return (self._density.exp()/self.eval_grid).log()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return (super(LogDiscreteDistribution, self).log_prob(x.log()).exp()/x).log()
    def applyPrior(self, prior: torch.distributions.Distribution):
        newDist = super(LogDiscreteDistribution, self).applyPrior(prior)
        newObj = LogDiscreteDistribution(newDist.log_density,newDist.eval_grid)
        newObj.prior = prior
        return newObj
class KDEDensity(DiscreteDistribution):
    def __init__(self, data: torch.tensor, eval_grid: torch.tensor, weights: torch.tensor = None):
        if weights is None:
            weights = torch.ones_like(data)
        self._weights = weights
        npdata = data.flatten().numpy()
        npweights = weights.flatten().numpy()
        self._data = npdata
        self._kde = FFTKDE(bw='silverman').fit(npdata, weights=npweights)
        self._eval_grid = eval_grid.flatten()
        super(KDEDensity, self).__init__(self._log_prob(self.eval_grid), self.eval_grid.flatten())
    #
    def _log_prob(self, x: torch.tensor) -> torch.tensor:
        return torch.from_numpy(self._kde.evaluate(x.flatten().numpy())).log()



class LogTransformedKDEDensity(KDEDensity,LogDiscreteDistribution):
    def __init__(self, data: torch.Tensor, eval_grid: torch.tensor, weights: torch.tensor=None):
        assert (data > 0).all()
        assert (eval_grid > 0).all()
        super(LogTransformedKDEDensity, self).__init__(data.log(), eval_grid.log(),weights)
    #
    def _log_prob(self, x: torch.tensor) -> torch.tensor:
        return torch.from_numpy(self._kde.evaluate(x.log().numpy())/self.eval_grid.numpy()).log()




