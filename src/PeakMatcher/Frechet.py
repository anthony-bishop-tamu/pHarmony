import torch
from torch.distributions import constraints
class Frechet(torch.distributions.Distribution):
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    support = constraints.positive  # Support of the distribution (x > 0)

    def __init__(self, alpha, scale, validate_args=None):
        self._alpha = alpha  # Shape parameter (fitted_c from SciPy)
        self._scale = scale # Scale parameter (fitted_scale from SciPy)
        self._param = torch.cat([self._alpha,self._scale],dim=-1)
        super(Frechet, self).__init__(torch.Size(), validate_args=validate_args)

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
class RadialFrechet(Frechet):
    def __init__(self, alpha, scale, dim, loc=0, validate_args=None):
        self._dim = dim
        self._radialFactor = torch.log(2*torch.pi**(self._dim/2.0)) - torch.special.gammaln(self._dim/2.0)
        super(RadialFrechet, self).__init__(alpha, scale, loc, validate_args=validate_args)
    def log_prob(self, x):
        #x is distances^2
        frechet = super(RadialFrechet, self).log_prob(x)
        radialScale = torch.log(x)*(self._dim - 1.0) - self._radialFactor
        return frechet + radialScale
    #
#
class RadialChi2(torch.distributions.Chi2):
    def __init__(self, dof, validate_args=None):
        self._radialFactor = torch.log(2 * torch.pi ** (dof / 2.0)) - torch.special.gammaln(dof / 2.0)
        super(RadialChi2, self).__init__(dof, validate_args=validate_args)
    def log_prob(self, x):
        #x is distances^2
        chi2 = super(RadialChi2, self).log_prob(x)
        radialScale = torch.log(x)*(self.df - 1.0) - self._radialFactor
        return chi2 + radialScale
    #
class RegFrechet(Frechet):
    def __init__(self, alpha, max_val, n, validate_args=None):
        scale = torch.pow(max_val,alpha)*torch.log(torch.tensor([2.0]))/n
        super(RegFrechet, self).__init__(alpha, scale.detach(), validate_args=validate_args)
        self._max_val = max_val
        self._n = n
        self._param = torch.cat([self._param,self._max_val,self._n],dim=-1)
    @property
    def max_val(self):
        return self._max_val
    @property
    def n(self):
        return self._n
class UniformDistanceSquared(torch.distributions.Distribution):
    arg_constraints = {'_dim': constraints.positive, '_Rmax': constraints.positive}
    def __init__(self,dim, Rmax, validate_args=None):
        self._dim = dim
        self._Rmax = Rmax
        super(UniformDistanceSquared, self).__init__(torch.Size(), validate_args=validate_args)
    def log_prob(self, x):
        return torch.log(self._dim/(2*self._Rmax)).unsqueeze(-1) + ((self._dim-2.0)/2.0) * torch.log(x)

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







