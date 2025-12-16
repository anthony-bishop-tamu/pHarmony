import torch
from torch.distributions import constraints
class Frechet(torch.distributions.Distribution):
    def __init__(self, alpha_logit, scale_logit, alpha_min, scale_min, validate_args=False):
        self._alpha_logit = alpha_logit.requires_grad_(True)  # Shape parameter (fitted_c from SciPy)
        self._scale_logit = scale_logit.requires_grad_(True) # Scale parameter (fitted_scale from SciPy)
        self._alpha_min = alpha_min
        self._scale_min = scale_min
        self._params = [self._alpha_logit,self._scale_logit]
        super(Frechet, self).__init__(torch.Size(), validate_args=validate_args)

    def clone(self):
        return Frechet(self._alpha_logit.detach().clone(),
                       self._scale_logit.detach().clone(),
                        self._alpha_min, self._scale_min)
    def sample(self, sample_shape=torch.Size()):
        # Generate Weibull samples
        weibull_samples = torch.distributions.Weibull(self.alpha, self.scale).sample(sample_shape)
        # Apply the transformation to get Fréchet samples
        frechet_samples = weibull_samples.reciprocal()  # Shift by loc
        return frechet_samples

    def log_prob(self, x):
        # Log probability of the Fréchet distribution
        z = x / self.scale
        return torch.log(self.alpha)  - torch.log(self.scale) - (self.alpha + 1) * torch.log(z) - z ** (-self.alpha)

    def median(self):
        return (self.scale/torch.pow(torch.log(torch.tensor([2.0])), 1.0/self.alpha))

    def mode(self):
        return self.scale*torch.pow(self.alpha/(1+self.alpha), 1.0/self.alpha)

    def quantile(self, p):
        return self.scale*torch.pow(-torch.log(p),-1.0/self.alpha)

    def variance(self):
        if self.alpha < 2.0:
            return torch.tensor([torch.inf])
        else:
            return self.scale*self.scale*(torch.lgamma(1-2.0/self.alpha).exp() - torch.lgamma(1-1.0/self.alpha).exp()**2)
        #
    @property
    def alpha(self):
        return self._alpha_min + torch.nn.functional.softplus(self._alpha_logit)
    @property
    def scale(self):
        return self._scale_min + torch.nn.functional.softplus(self._scale_logit)
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









