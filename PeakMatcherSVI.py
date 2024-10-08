import pyro
import torch
from SMCMatchingDistribution import SMCMatchingDistribution
from DistancesMM import DistancesMM
import scipy.stats as stats
import argparse
import pandas as pd
import numpy as np
def calculatePositionProb(sample, shape):
    positionProbs = torch.zeros(shape, dtype=torch.float32)
    rows = torch.arange(shape[0]).unsqueeze(0).expand(sample.shape[0], shape[0])
    mask = sample.flatten() >= 0
    positionProbs.index_put_((rows.flatten()[mask], sample.flatten()[mask]), torch.tensor([1.0]), accumulate=True)
    positionProbs /= sample.shape[0]
    return positionProbs
#
def initialMatchinglogits(data: torch.tensor):
    params = torch.zeros(data.shape+(2,), dtype=torch.float32)
    params[:,:,0][data < 7] = 4
    return params

def initialNonMatchingParams(data: torch.tensor):
    shape,loc,scale = stats.weibull_min.fit(data.flatten().numpy(),floc=0)
    return torch.tensor([shape,scale],dtype=torch.float32)

def initialCSPParams(data: torch.tensor):
    initial_csp_assignments = torch.zeros(data.shape+(2,), dtype=torch.float32)
    initial_csp_assignments[:,:,1] = data - 7
    return initial_csp_assignments

def calculateCSPDistParameters(quantile, cutoff, median):

    numerator = torch.log(-1*torch.log(1-quantile)) - torch.log(torch.log(torch.tensor([2.0])))
    denominator = torch.log(cutoff) - torch.log(median)

    k = numerator / denominator
    lam = median/torch.pow(torch.log(torch.tensor([2.0])),1.0/k)
    return torch.tensor([k,lam],dtype=torch.float32)
#

# Define the model with a latent variable
def model(data,dof,no_match_distribution_parameters,initial_matching_logits,initial_csp_logits):
    # Latent variable z follows a normal distribution (prior)

    match_logits = pyro.param("matching_logits")
    csp_logits = pyro.param("csp_logits")
    cutoff = stats.chi2(dof).ppf(0.99)

    csp_distribution_median = pyro.sample("csp_distribution_median",pyro.distributions.Uniform(cutoff,300))
    csp_distribution_parameters = calculateCSPDistParameters(torch.tensor([0.02]),cutoff=torch.tensor([cutoff]),median=csp_distribution_median)

    matching_distribution = SMCMatchingDistribution(match_logits)
    matching_sample = pyro.sample("matching", matching_distribution, sample_shape=(100,))

    matching_probabilities = calculatePositionProb(matching_sample, data.shape)
    nomatch_probabilities = 1.0 - matching_probabilities
    csp_probits = (csp_logits - csp_logits.logsumexp(dim=2,keepdim=True)).exp()
    no_csp_probabilites = matching_probabilities*csp_probits[:,:,0]
    csp_probabilities = matching_probabilities*csp_probits[:,:,1]

    assignment_probabilities = torch.stack([no_csp_probabilites,csp_probabilities,nomatch_probabilities], dim=2)

    distance_mixture_model = DistancesMM(torch.tensor([dof]),
                                        csp_distribution_parameters,
                                        no_match_distribution_parameters,
                                        assignment_probabilities)
    pyro.sample("obs", distance_mixture_model, obs=data)

# Define the guide (variational approximation of the posterior)
def guide(data,dof,no_match_distribution_parameters,initial_matching_logits,initial_csp_logits):

    match_logits = pyro.param("matching_logits", initial_matching_logits)
    csp_logits = pyro.param("csp_logits",initial_csp_logits)

    csp_distribution_median_map = pyro.param("csp_distribution_median_map",torch.tensor([50.0]))

    pyro.sample("csp_distribution_median",pyro.distributions.Delta(csp_distribution_median_map))

    #matching_distribution = SMCMatchingDistribution(match_logits)
    #matching_sample = pyro.sample("matching", matching_distribution,sample_shape=(1000,))
#
def getPeakPositionsFromFile(filename, cs_cols, uncertaintycols=None, fixedError=None):
    df = pd.read_csv(filename,sep="\s+")
    positions = df[cs_cols].to_numpy(dtype=np.float32)
    if uncertaintycols is not None:
        uncertainties = df[uncertaintycols].to_numpy(dtype=np.float32)
    elif fixedError is not None:
        uncertainties = np.zeros_like(positions)
        uncertainties[:,:] = np.array(fixedError,dtype=np.float32)[np.newaxis,:]
    else:
        raise ValueError("Must specify either uncertaintycols or fixedError")
    #
    return torch.from_numpy(np.stack((positions,uncertainties),axis=2))
#
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list', type=str, help='reference peak list filename')
    parser.add_argument( '--reference_cs_column_names', type=str, nargs='+', help='reference cs column names')
    parser.add_argument('--target_peak_list', type=str, help='target peak list filename')
    parser.add_argument('--target_cs_column_names', type=str, nargs='+', help='target cs column names')


    return parser.parse_args()
#

args = parseArguments()
reference_peaks = getPeakPositionsFromFile(args.reference_peak_list,
                                               args.reference_cs_column_names,
                                               fixedError=[0.015,0.0015])
target_peaks = getPeakPositionsFromFile(args.target_peak_list,
                                            args.target_cs_column_names,
                                            fixedError=[0.015,0.0015])
components_distances_squared = torch.pow(reference_peaks[:,:,0].unsqueeze(dim=-2) - target_peaks[:,:,0].unsqueeze(dim=0),2)
components_distances_squared_normalized = (
        components_distances_squared/(torch.pow(reference_peaks[:,:,1].unsqueeze(dim=-2),2) +
                                      torch.pow(target_peaks[:,:,1].unsqueeze(dim=0),2))
)
distances_squared_normalized = components_distances_squared_normalized.sum(dim=-1)
distances_squared_normalized[distances_squared_normalized == 0] = torch.finfo(torch.float32).eps
transposed = False
if distances_squared_normalized.shape[0] < distances_squared_normalized.shape[1]:
    distances_squared_normalized = distances_squared_normalized.transpose(0,1)
    transposed = True

non_matching_params = initialNonMatchingParams(distances_squared_normalized)



# Set up the optimizer
optimizer = pyro.optim.Adam({"lr": 0.01})

# Set up SVI with the ELBO loss function
svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

initial_matching_logits = initialMatchinglogits(distances_squared_normalized)
initial_csp_logits = initialCSPParams(distances_squared_normalized)

# Training loop
num_steps = 500
for step in range(num_steps):
    # Perform a gradient step
    loss = svi.step(distances_squared_normalized,2,non_matching_params,initial_matching_logits,initial_csp_logits)

    if step % 50 == 0:
        print(f"Step {step} : Loss = {loss}")
        optimized_matching = pyro.param("matching_logits")
        csp_logits = pyro.param("csp_logits")
        csp_distribution_median_map = pyro.param("csp_distribution_median_map")
        print(csp_distribution_median_map)


optimized_matching = pyro.param("matching_logits")
csp_logits = pyro.param("csp_logits")
csp_distribution_median_map = pyro.param("csp_distribution_median_map")
sample = SMCMatchingDistribution(optimized_matching).sample((1000,))
probs = calculatePositionProb(sample,optimized_matching.shape[0:2])
print(csp_distribution_median_map)