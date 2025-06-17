import matplotlib.pyplot as plt
import numpy as np
import scipy.stats.distributions as dist
from Frechet import Frechet, UniformDistanceSquared
import torch
distances = np.array([0.31,	272.3,	594.3,	720.0,	187.1,
126.6,	22.6,	174.3,	294.1,	160.6,
886.6,	211.3,	34.1,	60.5,	533.8,
160.0,	282.1,	409.6,	380.3,	6.3])

# Choose logarithmic bin edges
distances = np.sort(distances)


# Choose a small fixed width (in linear space) to make bars narrow
bar_width = 0.05 * distances  # or set to a constant like 100

density_heights = np.ones_like(distances,dtype=np.float32)/np.sum(bar_width)
# Plot bars


domain = np.logspace(np.log10(0.1),np.log10(distances.max()),10000)

chi2_pdf = dist.chi2.pdf(domain,2) * 2.0/25
f = Frechet(torch.tensor([1.01]),torch.tensor([30.0]))
frechet_pdf = f.log_prob(torch.from_numpy(domain)).exp().detach().numpy()*2.0/25
uniform_distance_squared = UniformDistanceSquared(torch.tensor([2.0]),torch.tensor([np.max(distances)]))
uniform_distance_squared_pdf = uniform_distance_squared.log_prob(torch.from_numpy(domain)).exp().detach().squeeze().numpy()*21.0/25


# Log scale

plt.xscale('log')
chi2_mask = (5E-4 <= chi2_pdf) & (chi2_pdf <= np.max(chi2_pdf))
frechet_mask = (5E-4 <= frechet_pdf) & (frechet_pdf <= np.max(chi2_pdf))
uniform_distance_squared_mask = (5E-4 <= uniform_distance_squared_pdf) & (uniform_distance_squared_pdf <= np.max(chi2_pdf))

plt.plot(domain[chi2_mask], chi2_pdf[chi2_mask], color='black')
plt.plot(domain[frechet_mask], frechet_pdf[frechet_mask], color='blue')
plt.plot(domain[uniform_distance_squared_mask], uniform_distance_squared_pdf[uniform_distance_squared_mask], color='red')
density_heights = 0.05*np.max(chi2_pdf)
plt.bar(distances, density_heights, width=bar_width, align='center', edgecolor='black', bottom=5.001E-4)
# Labels
plt.yscale('log')
#plt.ylim([5E-4,np.max(chi2_pdf)])
plt.savefig('SampleFigure.svg')