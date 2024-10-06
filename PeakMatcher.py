import numpy as np
import scipy.stats as stats
import torch
import pandas as pd
import sys
import argparse
from CSPDetectionDistribution import CSPDetectionDistribution
from matplotlib import pyplot as plt

class SampleSizeToLargeError(Exception):
    pass
#
class EMConvergenceFailureError(Exception):
    pass
#
def getInitialWeibullParameters(distances):
    distances = distances.flatten().numpy()
    shape,loc,scale = stats.weibull_min.fit(distances,floc=0)
    return torch.tensor([shape,scale],dtype=torch.float32)
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
def getInitialCSPDistributionParameters(distances: torch.tensor):
    assert distances.dim() == 2
    cutoff = 5.0
    closestMatches = distances.min(dim=-1)[0]
    closestMatches = closestMatches[closestMatches > cutoff]

    assigmentParameters = torch.zeros(distances.shape+(2,),dtype=torch.float32)
    #assignmentParameters are logits
    assigmentParameters[...,1] = (distances > 5)*3
    assigmentParameters[...,1] = ((3 < distances) & (distances < 5))*1
    assigmentParameters[...,1] = (distances < 3)* -2

    concentration,loc,scale = stats.weibull_min.fit(closestMatches.detach().numpy(),floc=0)
    #scale is the inverse of rate (i.e. rate = 1.0/scale)
    #loc is ignored as it is fixed at zero

    return torch.tensor([concentration,scale],dtype=torch.float32), assigmentParameters
#

def verifyTensorConvergence(torchPreviousParameter: torch.tensor,
                               torchNewParameter: torch.tensor,
                               averageDeviation: torch.tensor,
                               maxDeviation: torch.tensor):
    difference = torchNewParameter - torchPreviousParameter
    converged = difference.abs().mean() < averageDeviation and difference.abs().max() < maxDeviation
    print(converged.item(), difference.abs().mean(), averageDeviation, difference.abs().max(), maxDeviation)
    return converged.item(), difference.abs().mean(), difference.abs().max()
#
def calculatePositionProb(sample, weights, shape):
    weights = (weights - weights.logsumexp(dim=0,keepdim=True)).exp()
    positionProbs = torch.zeros(shape, dtype=torch.float32)
    rows = torch.arange(shape[0]).unsqueeze(0).expand(sample.shape[0], shape[0])
    weights_expanded = weights.unsqueeze(1).expand(sample.shape[0], shape[0]).flatten()
    mask = sample.flatten() >= 0
    positionProbs.index_put_((rows.flatten()[mask], sample.flatten()[mask]), weights_expanded[mask], accumulate=True)
    unique_elements, counts = torch.unique(sample, return_counts=True, dim=1)
    assert (positionProbs.sum(dim=-1) < 1.0+1E-3).all() and (positionProbs.sum(dim=0) < 1.0+1E-3).all()
    return positionProbs
#
def validateSufficentSampling(samples: tuple, shape: tuple) -> bool:
    matchings, weights = samples
    firstHalf = calculatePositionProb(matchings[::2,...],weights[::2,...], shape)
    secondHalf = calculatePositionProb(matchings[1::2,...],weights[1::2,...], shape)
    converged, mean, max = verifyTensorConvergence(firstHalf,secondHalf,
                            torch.tensor([0.03],dtype=torch.float32),
                            torch.tensor([0.05],dtype=torch.float32))
    return converged
#
def EM_minimization_function(samples, dist: CSPDetectionDistribution):

    logLikelihoodTerm = dist.log_prob(samples[0],samples[1]).sum()
    value = logLikelihoodTerm
    csp_1_p = dist.csp_distribution().icdf(0.02)
    k = dist.csp_distribution_parameters()[0]
    df = dist.chi2_distribution().df
    position = stats.chi2(df).ppf(0.999)

    csp_dist_regularization = -1*torch.pow(torch.relu(position - csp_1_p),6) + -1*torch.pow(torch.relu(5.0-k),5)
    #csp_dist_regularization=0

    print("Loss: ", logLikelihoodTerm, csp_dist_regularization)
    return -1 *(value + csp_dist_regularization)
def maximization(samples: tuple,
                 distances: torch.tensor,
                 csp_assignment_params: torch.tensor,
                 csp_distribution_params: torch.tensor,
                 no_match_distribution_parameters: torch.tensor):


    optimizer = torch.optim.Adam([csp_assignment_params,csp_distribution_params], lr=1E-3)

    maxIterators = 10000
    prevLoss = torch.finfo(torch.float32).max
    for i in range(maxIterators):
        optimizer.zero_grad()
        dist = CSPDetectionDistribution(distances, csp_assignment_params, csp_distribution_params,
                                        no_match_distribution_parameters)
        loss = EM_minimization_function(samples, dist)
        loss.backward()
        csp_distribution_parameters.grad[torch.isnan(csp_distribution_params.grad)]=0
        optimizer.step()
        if abs(prevLoss - loss.item()) < 1e-6:
            break
        #
        prevLoss = loss.item()

    return loss.item()

#
def runEMStep(distances: torch.tensor,
              csp_probability_params: torch.tensor,
              csp_distribution_params: torch.tensor,
              no_match_distribution_params: torch.tensor,
              sampleSize: int):

        dist = CSPDetectionDistribution(distances,
                                        csp_probability_params,
                                        csp_distribution_params,
                                        no_match_distribution_params)
        samples = dist.sample((sampleSize,))
        maximization(samples,distances,
                     csp_probability_params,
                     csp_distribution_params,
                     no_match_distribution_params)
        return samples
#
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list', type=str, help='reference peak list filename')
    parser.add_argument( '--reference_cs_column_names', type=str, nargs='+', help='reference cs column names')
    parser.add_argument('--target_peak_list', type=str, help='target peak list filename')
    parser.add_argument('--target_cs_column_names', type=str, nargs='+', help='target cs column names')


    return parser.parse_args()
if __name__ == "__main__":
    torch.manual_seed(42)
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
    #

    initial_nonMatching_distribution_parameters = getInitialWeibullParameters(distances_squared_normalized)
    initial_csp_distribution_parameters, initial_assignment_parameters = (
        getInitialCSPDistributionParameters(distances_squared_normalized))

    initial_csp_distribution_parameters[0].clamp_(min=1.01)
    #initial_csp_distribution_parameters = torch.tensor([5.0,20],dtype=torch.float32)
    csp_distribution_parameters = torch.tensor(initial_csp_distribution_parameters,requires_grad=True)
    assignment_parameters = torch.tensor(initial_assignment_parameters,requires_grad=True)
    nonMatching_distribution_parameters = torch.tensor(initial_nonMatching_distribution_parameters,requires_grad=True)






    # get appropriate sample size
    sampleSize = distances_squared_normalized.shape[0]
    maxTries = 8
    for i in range(maxTries):
        print(f"Trying Sample Size: {sampleSize}")
        dist = CSPDetectionDistribution(distances_squared_normalized,
                                        assignment_parameters,
                                        csp_distribution_parameters,
                                        nonMatching_distribution_parameters)
        samples = dist.sample((sampleSize,))
        converged = validateSufficentSampling(samples, distances_squared_normalized.shape)
        if not converged:
            sampleSize *= 2
            if i == maxTries-1:
                raise SampleSizeTooLargeError()
        else:
            print(f"Sample Size: {sampleSize} is sufficient")
            break
        #
    #
    # run EM
    maxEMSteps = 100
    minSteps = 5
    samples=()
    for i in range(maxEMSteps):
        previous_assignment_parameters = assignment_parameters.detach().clone()
        previous_csp_distribution_parameters = csp_distribution_parameters.detach().clone()
        previous_nonMatching_distribution_parameters = nonMatching_distribution_parameters.detach().clone()

        samples = runEMStep(distances_squared_normalized,
                  assignment_parameters,
                  csp_distribution_parameters,
                  nonMatching_distribution_parameters,sampleSize)

        assignment_converged, assign_mean, assign_max = (
            verifyTensorConvergence((previous_assignment_parameters-previous_assignment_parameters.logsumexp(dim=2,keepdim=True)).exp(),
                                    (assignment_parameters-assignment_parameters.logsumexp(dim=2,keepdim=True)).exp(),
                                                      0.02,
                                                      0.03))
        csp_distribution_converged,csp_mean,csp_max = verifyTensorConvergence(previous_csp_distribution_parameters,
                                                             csp_distribution_parameters,
                                                             0.052,
                                                             0.03)
        nonMatching_distribution_converged = verifyTensorConvergence(previous_nonMatching_distribution_parameters,
                                                                     nonMatching_distribution_parameters,
                                                                     0.02,
                                                                     0.03)
        print(f"assignment change {assign_mean}, {assign_max}  csp_dist_change {csp_mean}, {csp_max}")
        if assignment_converged and csp_distribution_converged and nonMatching_distribution_converged and i >= minSteps:
            break
        else:
            if i == maxEMSteps - 1:
                raise EMConvergenceFailureError()
            #
        #
    #
    positionProbs = calculatePositionProb(samples[0],samples[1],distances_squared_normalized.shape)
    matches = distances_squared_normalized[positionProbs>0.90].numpy()
    assignment_parameters = (assignment_parameters - assignment_parameters.logsumexp(dim=2,keepdim=True)).exp().detach()
    csp_distribution_parameters = csp_distribution_parameters.detach().numpy()
    weights = (assignment_parameters*positionProbs.unsqueeze(-1))
    weights = weights[positionProbs > 0.90,:]
    weights = weights/weights.sum()
    chi2_weight = weights[:,0].sum().item()
    csp_weight = weights[:,1].sum().item()
    binwidth = 1
    match_bins = np.arange(0,np.max(matches),1)
    chi2_pdf = stats.chi2(2).pdf(match_bins)
    csp_pdf = (stats.weibull_min(csp_distribution_parameters[0],
                                   scale=csp_distribution_parameters[1]).pdf(match_bins)*csp_weight)
    plt.hist(matches,bins=match_bins,label='Matches',density=True)
    plt.plot(match_bins,chi2_pdf*chi2_weight,label='noCSP Distribution',linewidth=2)
    plt.plot(match_bins,csp_pdf,label='CSP Distribution',linewidth=2)
    plt.plot(match_bins,chi2_pdf*chi2_weight+csp_pdf,'r',label='matching_pdf')
    plt.plot(match_bins,chi2_pdf,label="full chi2",linewidth=2)
    plt.xlim([0,100])
    plt.yscale('log')
    plt.legend()
    plt.show()
#









