import torch
from torch.distributions import constraints
class Frechet(torch.distributions.Distribution):
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    support = constraints.positive  # Support of the distribution (x > 0)

    def __init__(self, alpha, scale, validate_args=None):
        self.alpha = alpha  # Shape parameter (fitted_c from SciPy)
        self.scale = scale  # Scale parameter (fitted_scale from SciPy)
        super(Frechet, self).__init__(torch.Size(), validate_args=validate_args)

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

    @property
    def variance(self):
        if self.alpha < 2.0:
            return torch.tensor([torch.inf])
        else:
            return self.scale*self.scale*(torch.lgamma(1-2.0/self.alpha).exp() - torch.lgamma(1-1.0/self.alpha).exp()**2)
        #


