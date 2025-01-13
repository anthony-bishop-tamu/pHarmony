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
class KDEDensity(torch.distributions.Distribution):
    def __init__(self, data: torch.tensor, eval_grid: torch.tensor, weights: torch.tensor = None):
        if weights is None:
            weights = torch.ones_like(data)
        self._weights = weights
        npdata = data.flatten().numpy()
        npweights = weights.flatten().numpy()
        self._data = npdata
        self._kde = FFTKDE(bw=1.0).fit(npdata, weights=npweights)
        self._eval_grid = eval_grid.flatten()
        self._density = self._log_prob(self._eval_grid)
        super(KDEDensity, self).__init__(torch.Size(), validate_args=None)
    #
    def _log_prob(self, x: torch.tensor) -> torch.tensor:
        return torch.from_numpy(self._kde.evaluate(x.flatten().numpy())).log()
    def log_prob(self, x: torch.tensor) -> torch.Tensor:
        range = self._eval_grid[-1] - self._eval_grid[0]
        stepSize = range/(self._eval_grid.numel()-1)
        idx = ((x - self._eval_grid[0])/stepSize).round().long()
        return self._density[idx]
    @property
    def eval_grid(self) -> torch.Tensor:
        return self._eval_grid
    @property
    def log_density(self) -> torch.tensor:
        return self._density

    def _set_log_density(self, x: torch.tensor):
        assert x.shape == self._density.shape
        self._density = x
    def weigh_by_prior(self, prior: torch.tensor) -> torch.tensor:
        assert self.log_density.shape == prior.shape
        new_density = self.log_density + prior
        new_density -= new_density.logsumexp(dim=-1, keepdim=True)
        self._set_log_density(new_density)



class LogTransformedKDEDensity(KDEDensity):
    def __init__(self, data: torch.Tensor, eval_grid: torch.tensor, weights: torch.tensor=None):
        assert (data > 0).all()
        assert (eval_grid > 0).all()
        super(LogTransformedKDEDensity, self).__init__(data.log(), eval_grid.log(),weights)
    #
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return (super(LogTransformedKDEDensity, self).log_prob(x.log()).exp()/x).log()

    @property
    def eval_grid(self) -> torch.Tensor:
        return super(LogTransformedKDEDensity, self).eval_grid.exp()
    @property
    def log_density(self) -> torch.tensor:
        return (super(LogTransformedKDEDensity, self).log_density.exp()/self.eval_grid).log()

    def _set_log_density(self, x: torch.tensor):
        super(LogTransformedKDEDensity, self)._set_log_density((x.exp()*self.eval_grid).log())




class CrystalBall(torch.distributions.Distribution):
    def __init__(self, b: torch.tensor, m: torch.tensor, loc: torch.tensor, scale: torch.tensor):
        self.b = b
        self.m = m
        self.loc = loc
        self.scale = scale
        super(CrystalBall, self).__init__(torch.Size(), validate_args=None)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.loc) / self.scale
