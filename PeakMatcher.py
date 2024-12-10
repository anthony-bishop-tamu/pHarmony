import numpy as np
import scipy.stats as stats
import torch
import pandas as pd
import sys
import argparse
from CSPDetectionDistribution import CSPDetectionDistribution
from matplotlib import pyplot as plt
from OutputHandling import buildPlot, outputResults

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
    cutoff = 3
    closestMatches = distances.min(dim=-1)[0]
    closestMatches = closestMatches[closestMatches > cutoff]

    assigmentParameters = torch.zeros(distances.shape+(2,),dtype=torch.float32)
    #assignmentParameters are logits
    assigmentParameters[...,1] += (distances >= 15)*8
    assigmentParameters[...,1] += ((5 < distances) & (distances < 15))*1
    assigmentParameters[...,1] += (distances <= 5)* -8

    #scale is the inverse of rate (i.e. rate = 1.0/scale)
    #loc is ignored as it is fixed at zero
    median = closestMatches.median(dim=-1)[0]
    #max = closestMatches.max(dim=-1)[0]
    max = torch.tensor([20.0])
    log_med_max = torch.log(3/max)
    q = torch.log(torch.tensor([0.99]))/torch.log(torch.tensor([0.1]))
    alpha = torch.log(q)/log_med_max
    scale = median*(-1*torch.log(torch.tensor([0.1])))**torch.tensor([1.0/alpha])

    return assigmentParameters, torch.tensor([alpha,scale],dtype=torch.float32)
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
def calculatePositionProb(sample, shape):
    positionProbs = torch.zeros(shape, dtype=torch.float32)
    rows = torch.arange(shape[0]).unsqueeze(0).expand(sample.shape[0], shape[0])
    mask = sample.flatten() >= 0
    positionProbs.index_put_((rows.flatten()[mask], sample.flatten()[mask]), torch.tensor([1.0/sample.shape[0]]), accumulate=True)
    unique_elements, counts = torch.unique(sample, return_counts=True, dim=1)
    assert (positionProbs.sum(dim=-1) < 1.0+1E-3).all() and (positionProbs.sum(dim=0) < 1.0+1E-3).all()
    return positionProbs
#
def validateSufficentSampling(samples: tuple, shape: tuple) -> bool:
    matchings = samples
    firstHalf = calculatePositionProb(matchings[::2,...], shape)
    secondHalf = calculatePositionProb(matchings[1::2,...], shape)
    converged, mean, max = verifyTensorConvergence(firstHalf,secondHalf,
                            torch.tensor([0.05],dtype=torch.float32),
                            torch.tensor([0.05],dtype=torch.float32))
    return converged
#
def calculateCSPDistParameters(quantile, cutoff, median):

    numerator = torch.log(-1*torch.log(1-quantile)) - torch.log(torch.log(torch.tensor([2.0])))
    denominator = torch.log(cutoff) - torch.log(median)

    k = numerator / denominator
    lam = median/torch.pow(torch.log(torch.tensor([2.0])),1.0/k)
    return torch.tensor([k,lam],dtype=torch.float32)
#
def EM_minimization_function(samples, dist: CSPDetectionDistribution):

    logLikelihoodTerm = dist.log_prob(samples).sum()#/(samples.numel()*dist._distances.numel())
    positionProb = calculatePositionProb(samples, dist._distances.shape)
    scale = dist.csp_distribution().scale
    alpha = dist.csp_distribution().alpha
    mode = scale*torch.pow(alpha/(1+alpha),1.0/alpha)
    print("mode", mode)
    reg = (torch.relu(5.0-mode)*1000)**5
    #reg = 0
    value = logLikelihoodTerm
    return -1 *value + reg
def maximization(samples: tuple,
                 distances: torch.tensor,
                 csp_assignment_params: torch.tensor,
                 csp_distribution_params: torch.tensor,
                 optimization_list: list,
                 no_match_distribution_parameters: torch.tensor):


    #optimizer = torch.optim.Adam([csp_assignment_params, csp_distribution_params], lr=1E-3)
    optimizer = torch.optim.Adam(optimization_list, lr=1E-3)
    maxIterators = 10000
    prevLoss = torch.finfo(torch.float32).max
    for i in range(maxIterators):
        optimizer.zero_grad()
        dist = CSPDetectionDistribution(distances, csp_assignment_params, csp_distribution_params,
                                        no_match_distribution_parameters)
        loss = EM_minimization_function(samples, dist)
        loss.backward()
        optimizer.step()
        print("Loss: ", loss.item(), prevLoss-loss.item())
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
                     [csp_distribution_params],
                     no_match_distribution_params)
        maximization(samples,distances,
                     csp_probability_params,
                     csp_distribution_params,
                     [csp_probability_params],
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
    initial_assignment_parameters,initial_csp_distribution_parameters = (
        getInitialCSPDistributionParameters(distances_squared_normalized))

    assignment_parameters = torch.tensor(initial_assignment_parameters,requires_grad=True)
    nonMatching_distribution_parameters = torch.tensor(initial_nonMatching_distribution_parameters,requires_grad=True)
    csp_distribution_parameters = torch.tensor(initial_csp_distribution_parameters,requires_grad=True)





    # get appropriate sample size
    sampleSize = distances_squared_normalized.shape[0]
    maxTries = 12
    samples=()
    for i in range(maxTries):
        print(f"Trying Sample Size: {sampleSize}")
        dist = CSPDetectionDistribution(distances_squared_normalized,
                                        assignment_parameters,
                                        initial_csp_distribution_parameters,
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
    minSteps = 5
    maxEMSteps = 1000
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
                                                      0.01,
                                                      0.01))
        csp_distribution_converged,csp_mean,csp_max = verifyTensorConvergence(previous_csp_distribution_parameters,
                                                             csp_distribution_parameters,
                                                             0.01,
                                                             0.01)
        nonMatching_distribution_converged = verifyTensorConvergence(previous_nonMatching_distribution_parameters,
                                                                     nonMatching_distribution_parameters,
                                                                     0.01,
                                                                     0.01)
        print(f"assignment change {assign_mean}, {assign_max}  csp_dist_change {csp_mean}, {csp_max}")
        if assignment_converged and csp_distribution_converged and nonMatching_distribution_converged and i >= minSteps:
            break
        else:
            if i == maxEMSteps - 1:
                raise EMConvergenceFailureError()
            #
        #
    #
    dist = CSPDetectionDistribution(distances_squared_normalized, assignment_parameters, csp_distribution_parameters,
                                    nonMatching_distribution_parameters)
    samples = dist.sample((sampleSize,))
    positionProbs = calculatePositionProb(samples,distances_squared_normalized.shape).detach().numpy()
    assignment_parameters = (
                assignment_parameters - assignment_parameters.logsumexp(dim=2, keepdim=True)).exp().detach()[:,:,1].numpy()
    outputResults(positionProbs,
                      assignment_parameters,
                      (pd.read_csv(args.reference_peak_list,sep="\s+"),int(transposed),args.reference_cs_column_names),
                      # tuple of a pandas dataframe and the dimension (0 or 1) in the representation, and a list of the resonance columns
                      (pd.read_csv(args.target_peak_list, sep="\s+"), int(not transposed), args.target_cs_column_names),
                      # tuple of a pandas dataframe and the dimension (0 or 1) in the representation
                      "Ref2Target_transferred.list",
                      "Ref2Target_transferred_HC.list",
                      "Match_probabilities.csv",
                      "CSP_probabilities.csv",
                      0.50)


    buildPlot(positionProbs,
                  assignment_parameters,
                  dist.no_csp_distribution(),
                  dist.csp_distribution(),
                  dist.non_matching_distribution(),
                  "fittedDistributions.png",
                  distances_squared_normalized,
                  0.50)

#









