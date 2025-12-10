import torch
from torch.distributions import constraints
class Frechet(torch.distributions.Distribution):
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    support = constraints.positive  # Support of the distribution (x > 0)

    def __init__(self, alpha, scale, validate_args=None):
        self._alpha = alpha.requires_grad_(True)  # Shape parameter (fitted_c from SciPy)
        self._scale = scale.requires_grad_(True) # Scale parameter (fitted_scale from SciPy)
        self._param = torch.cat([self._alpha,self._scale],dim=-1)
        super(Frechet, self).__init__(torch.Size(), validate_args=validate_args)

    def clone(self):
        return Frechet(self._alpha.detach().clone(), self._scale.detach().clone())
    def sample(self, sample_shape=torch.Size()):
        # Generate Weibull samples
        weibull_samples = torch.distributions.Weibull(self._alpha, self._scale).sample(sample_shape)
        # Apply the transformation to get Fréchet samples
        frechet_samples = weibull_samples.reciprocal()  # Shift by loc
        return frechet_samples

    def log_prob(self, x):
        # Log probability of the Fréchet distribution
        z = x / self._scale
        return torch.log(self._alpha)  - torch.log(self._scale) - (self._alpha + 1) * torch.log(z) - z ** (-self._alpha)

    def median(self):
        return (self._scale/torch.pow(torch.log(torch.tensor([2.0])), 1.0/self._alpha))

    def mode(self):
        return self._scale*torch.pow(self._alpha/(1+self._alpha), 1.0/self._alpha)

    def quantile(self, p):
        return self._scale*torch.pow(-torch.log(p),-1.0/self._alpha)

    def variance(self):
        if self._alpha < 2.0:
            return torch.tensor([torch.inf])
        else:
            return self._scale*self._scale*(torch.lgamma(1-2.0/self._alpha).exp() - torch.lgamma(1-1.0/self._alpha).exp()**2)
        #
    @property
    def alpha(self):
        return self._alpha
    @property
    def scale(self):
        return self._scale
    @property
    def param(self):
        return self._param
class UniformDistanceSquared(torch.distributions.Distribution):
    arg_constraints = {'_omega': constraints.positive }
    def __init__(self,omega, validate_args=None):
        self._omega = omega
        super(UniformDistanceSquared, self).__init__(torch.Size(), validate_args=validate_args)
    def log_prob(self, x):
        return  torch.full_like(x,torch.log(1.0/self._omega).item())
    def clone(self):
        return UniformDistanceSquared(self._omega.detach().clone())









