import numpy as np
import scipy.stats as stats
import torch
import torch.distributions as dist
import pandas as pd
import sys
import argparse
from CSPDetectionDistribution import CSPDetectionDistribution
from matplotlib import pyplot as plt
from OutputHandling import buildPlot, outputResults
from Frechet import KDEDensity, LogTransformedKDEDensity
from pathlib import Path
class SampleSizeToLargeError(Exception):
    pass
#
class EMConvergenceFailureError(Exception):
    pass
def calculateBetaParametersFromMeanAndVariance(mean, variance):
    assert 0 < mean and mean < 1
    assert 0 < variance
    mu = ((mean*(1-mean))/variance - 1)
    alpha = mean*mu
    beta = (1-mean)*mu

    return torch.tensor([alpha, beta],dtype=torch.float64)

def getInitialNonMatchingParameters(distances,minimum_distance):
    #distances = distances.flatten()[distances.flatten() > 10]
    #s,loc,scale = stats.weibull_min.fit(distances,loc=0)
    return torch.tensor([minimum_distance,distances.max()+minimum_distance],dtype=torch.float64)
#
def getPeakPositionsFromFile(filename, cs_cols, uncertaintycols=None, fixedError=None):
    df = pd.read_csv(filename,sep="\s+")
    positions = df[cs_cols].to_numpy(dtype=np.float64)
    if uncertaintycols is not None:
        uncertainties = df[uncertaintycols].to_numpy(dtype=np.float64)
    elif fixedError is not None:
        uncertainties = np.zeros_like(positions)
        uncertainties[:,:] = np.array(fixedError,dtype=np.float64)[np.newaxis,:]
    else:
        raise ValueError("Must specify either uncertaintycols or fixedError")
    #
    return torch.from_numpy(np.stack((positions,uncertainties),axis=2))
#
def calculateMixtureWeights(csp_posterior_probabilities: torch.Tensor,
                            matching_posterior_probabilities: torch.Tensor,
                            csp_mixture_weight_priors: torch.Tensor,
                            matching_mixture_weight_priors: torch.Tensor) -> torch.Tensor:
    #csp_scale = matching_posterior_probabilities.sum()
    #matching_scale = matching_posterior_probabilities.numel()
    csp_scale = 1
    matching_scale = 1
    clamp_min = 1E-6

    pseudo_csp_posterior_probabilities = csp_posterior_probabilities.exp().clone()

    #pseudo_csp_posterior_probabilities = csp_posterior_probabilities.exp().detach() + clamp_min
    #pseudo_csp_posterior_probabilities /= pseudo_csp_posterior_probabilities.sum(dim=-1,keepdim=True)


    csp_mixture_weights = (pseudo_csp_posterior_probabilities*matching_posterior_probabilities.unsqueeze(-1)).detach()
    csp_mixture_weights = (csp_mixture_weights.sum(dim=(0,1)) + csp_scale*(csp_mixture_weight_priors))/(matching_posterior_probabilities.sum() + csp_scale*(csp_mixture_weight_priors).sum())
    matching_mixture_weights = (torch.tensor([matching_posterior_probabilities.detach().sum(),(1-matching_posterior_probabilities.detach()).sum()])
                                +matching_scale*(matching_mixture_weight_priors))/(matching_posterior_probabilities.numel() + (matching_scale*(matching_mixture_weight_priors)).sum())
    if csp_mixture_weights[1] < 1E-3:
        csp_mixture_weights = torch.tensor([1.0-1E-3,1E-3],dtype=torch.float64)

    assert (1 >= csp_mixture_weights).all() and (csp_mixture_weights >= 0).all()
    assert (1 >= matching_mixture_weights).all() and (matching_mixture_weights >= 0).all()
    return csp_mixture_weights, matching_mixture_weights
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
                            torch.tensor([0.01],dtype=torch.float64),
                            torch.tensor([0.05],dtype=torch.float64))
    return converged
#
def calculateCSPDistParameters(quantile, cutoff, median):

    numerator = torch.log(-1*torch.log(1-quantile)) - torch.log(torch.log(torch.tensor([2.0])))
    denominator = torch.log(cutoff) - torch.log(median)

    k = numerator / denominator
    lam = median/torch.pow(torch.log(torch.tensor([2.0])),1.0/k)
    return torch.tensor([k,lam],dtype=torch.float64)
#
def EM_minimization_function(samples, dist: CSPDetectionDistribution,
                             csp_mixture_weights: torch.Tensor,
                             matching_mixture_weights: torch.Tensor,
                             csp_mixture_priors: torch.Tensor,
                             matching_mixture_priors: torch.Tensor,):

    logLikelihoodTerm = dist.log_prob(samples).sum()#/(samples.numel()*dist._distances.numel())

    positionProb = calculatePositionProb(samples, dist._distances.shape)

    alpha = dist.csp_distribution.alpha
    mode_reg = -1* torch.abs(torch.relu(4.0-dist.csp_distribution.median()))**6


    loss = (-1 * logLikelihoodTerm +
            -1*((csp_mixture_priors-1.0)*csp_mixture_weights).sum()+
            -1*((matching_mixture_priors-1.0)*matching_mixture_weights).sum()) + mode_reg
    assert loss.isfinite().all()
    return loss
def maximization(samples: tuple,
                 distances: torch.tensor,
                 csp_mixture_weights: torch.tensor,
                 matching_mixture_weights: torch.tensor,
                 csp_mixture_priors: torch.tensor,
                 matching_mixture_priors: torch.tensor,
                 csp_distribution_params: torch.tensor,
                 optimization_list: list,
                 no_match_distribution_parameters: torch.tensor,
                 learning_rate: float):


    #optimizer = torch.optim.Adam([csp_assignment_params, csp_distribution_params], lr=1E-3)
    optimizer = torch.optim.Adam(optimization_list, lr=learning_rate)
    maxIterators = 10000
    prevLoss = torch.finfo(torch.float64).max
    for i in range(maxIterators):
        optimizer.zero_grad()
        dist = CSPDetectionDistribution(distances, csp_mixture_weights, matching_mixture_weights, csp_distribution_params,
                                        no_match_distribution_parameters)
        loss = EM_minimization_function(samples, dist,csp_mixture_weights,matching_mixture_weights,csp_mixture_priors,matching_mixture_priors)
        loss.backward()
        optimizer.step()
        print("Loss: ", loss.item(), prevLoss-loss.item(), csp_distribution_params.grad, no_match_distribution_parameters.grad)
        if prevLoss < loss.item():
            optimizer = torch.optim.Adam(optimization_list, lr=optimizer.param_groups[0]['lr']*0.5)
        elif prevLoss - loss.item() < 1e-7 and (csp_distribution_params.grad.abs() < 1e-3).all():
            break
        elif (torch.abs(csp_distribution_params.grad) < 1E-2).all():
            optimizer = torch.optim.Adam(optimization_list, lr=optimizer.param_groups[0]['lr'] * 1.1)

        #

        prevLoss = loss.item()

    return loss.item()

#
def runEMStep(distances: torch.tensor,
              csp_mixture_weights: torch.tensor,
              matching_mixture_weights: torch.tensor,
              csp_mixture_priors: torch.tensor,
              matching_mixture_priors: torch.tensor,
              csp_distribution_params: torch.tensor,
              no_match_distribution_params: torch.tensor,
              sampleSize: int,
              learning_rate: float,
              output_directory: Path,
              display_distributions: bool = True):


        dist = CSPDetectionDistribution(distances,
                                        csp_mixture_weights,
                                        matching_mixture_weights,
                                        csp_distribution_params,
                                        no_match_distribution_params)
        samples = dist.sample((sampleSize,))
        positionProbs = calculatePositionProb(samples, distances_squared_normalized.shape).detach()
        # Calculate new mixture weights
        csp_mixture_weights, matching_mixture_weights = calculateMixtureWeights(dist.csp_posterior_probabilities,positionProbs,csp_mixture_priors,matching_mixture_priors)
        maximization(samples, distances,
                     csp_mixture_weights.detach().log(),
                     matching_mixture_weights.detach().log(),
                     csp_mixture_priors,
                     matching_mixture_priors,
                     csp_distribution_params,
                     [csp_distribution_params],
                     no_match_distribution_params,
                         learning_rate)
        if display_distributions:
            fig = buildPlot(positionProbs,
                        csp_mixture_weights.detach().numpy(),
                        dist.no_csp_distribution,
                        dist.csp_distribution,
                        dist.non_matching_distribution,
                        output_directory/"fittedDistributions.png",
                        distances_squared_normalized,
                        0.50)
            fig.show()
        #
        return samples, csp_mixture_weights.log(), matching_mixture_weights.log()
#
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list', required=True, type=str, help='reference peak list filename')
    parser.add_argument( '--reference_cs_column_names', required=True, type=str, nargs='+', help='reference cs column names (e.g. \'w1\', \'w2\')')
    parser.add_argument('--target_peak_list', required=True, type=str, help='target peak list filename')
    parser.add_argument('--target_cs_column_names', required=True, type=str, nargs='+', help='target cs column names (e.g. \'w1\', \'w2\')')
    parser.add_argument('--reference_peak_list_error', required=True, type=float, nargs='+', help='Uncertainty in each dimension for the reference peak list (e.g. \" 0.0015 0.015 \" for a 2D HSQC [15N, 1H]')
    parser.add_argument('--target_peak_list_error', required=True, type=float, nargs='+', help='Uncertainty in each dimension for the target peak list (e.g. \" 0.0015, 0.015 \" for a 2D HSQC [15N, 1H]')
    parser.add_argument("--minimum_distance", type=float, help="Minimum normalized distance between two peaks, all normalized distances lower than this value will be set to this value",default=0.005)
    parser.add_argument('--expected_fraction_csp', type=float, help="Estimate of the fraction of peaks expected to undergo a chemical shift perturbation", default=0.1)
    parser.add_argument("--variance_scale_fraction_csp",type=float, help="scaling factor for variance of the prior distribution of csp distribution weight", default=2.0)
    parser.add_argument('--expected_fraction_missing', type=float, help="Estimate of the fraction of peaks that you think will be missing between spectra", default=0.1)
    parser.add_argument("--variance_scale_fraction_missing",type=float, help="scaling factor for variance of the prior distribution of matching distribution weight", default=2.0)
    parser.add_argument("--output_directory",type=Path,help="Directory path to output the results to", default="./peak_matcher_output")
    parser.add_argument( "--display_distributions", action='store_true', help="Display the distributions plots", )

    return parser.parse_args()
if __name__ == "__main__":
    #torch.manual_seed(42)
    args = parseArguments()

    display_distributions = args.display_distributions

    output_directory = args.output_directory
    output_directory.mkdir(parents=True,exist_ok=True)

    reference_peaks = getPeakPositionsFromFile(args.reference_peak_list,
                                               args.reference_cs_column_names,
                                               fixedError=args.reference_peak_list_error)
    target_peaks = getPeakPositionsFromFile(args.target_peak_list,
                                            args.target_cs_column_names,
                                            fixedError=args.target_peak_list_error)

    components_distances_squared = torch.pow(reference_peaks[:,:,0].unsqueeze(dim=-2) - target_peaks[:,:,0].unsqueeze(dim=0),2)
    components_distances_squared_normalized = (
            components_distances_squared/(torch.pow(reference_peaks[:,:,1].unsqueeze(dim=-2),2) +
                                          torch.pow(target_peaks[:,:,1].unsqueeze(dim=0),2))
    )
    distances_squared_normalized = components_distances_squared_normalized.sum(dim=-1)
    distances_squared_normalized[distances_squared_normalized < args.minimum_distance] = args.minimum_distance
    transposed = False

    if distances_squared_normalized.shape[0] < distances_squared_normalized.shape[1]:
        distances_squared_normalized = distances_squared_normalized.transpose(0,1)
        transposed = True
    #

    initial_nonMatching_parameters = getInitialNonMatchingParameters(distances_squared_normalized,minimum_distance=args.minimum_distance)
    initial_csp_distribution_parameters = torch.tensor([3,30],dtype=torch.float64)

    nonMatching_distribution_parameters = initial_nonMatching_parameters
    csp_distribution_parameters = initial_csp_distribution_parameters.clone().detach().requires_grad_(True)
    csp_mixture_weights =  torch.tensor([0,0],dtype=torch.float64)
    matching_mixture_weights =  torch.tensor([0,0],dtype=torch.float64)

    expected_no_csp_ratio = 1.0 - args.expected_fraction_csp
    no_csp_std = args.expected_fraction_csp #Std deviation is arbitrarily set to being the same as the expected fraction csp


    csp_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_no_csp_ratio,variance=args.variance_scale_fraction_csp*no_csp_std**2)  #[no csp, csp ] (Given a match!)

    expected_missing_ratio = args.expected_fraction_missing
    expected_match_ratio = min(distances_squared_normalized.shape)/distances_squared_normalized.numel()
    match_std = expected_match_ratio*expected_missing_ratio
    matching_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_match_ratio,variance=args.variance_scale_fraction_missing*match_std**2)  #[matching, nonmatching)


    # get appropriate sample size
    sampleSize = distances_squared_normalized.shape[0]
    maxTries = 12
    samples=()
    for i in range(maxTries):
        print(f"Trying Sample Size: {sampleSize}")
        dist = CSPDetectionDistribution(distances_squared_normalized,
                                        csp_mixture_weights,
                                        matching_mixture_weights,
                                        csp_distribution_parameters,
                                        nonMatching_distribution_parameters)
        samples = dist.sample((sampleSize,))
        converged = validateSufficentSampling(samples, distances_squared_normalized.shape)
        positionProb = calculatePositionProb(samples, distances_squared_normalized.shape)

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
    minSteps = 2
    maxEMSteps = 1000
    learning_rate = 1E-3
    for i in range(maxEMSteps):
        previous_csp_distribution_parameters = csp_distribution_parameters.detach().clone()
        previous_nonMatching_distribution_parameters = nonMatching_distribution_parameters.detach().clone()
        previous_csp_mixture_weights = csp_mixture_weights
        previous_matching_mixture_weights = matching_mixture_weights

        try:
            samples, csp_mixture_weights, matching_mixture_weights = runEMStep(
                  distances_squared_normalized,
                  csp_mixture_weights,
                  matching_mixture_weights,
                  csp_mixture_priors,
                  matching_mixture_priors,
                  csp_distribution_parameters,
                  nonMatching_distribution_parameters,
                  sampleSize,
                  learning_rate,
                  output_directory,
                  display_distributions)
        except ValueError:
            print("Learning rate too large, adjusting")
            learning_rate /= 2.0
            csp_distribution_parameters = previous_csp_distribution_parameters.detach().clone().requires_grad_(True)
            continue

        learning_rate = 1E-3


        csp_distribution_converged,csp_mean,csp_max = verifyTensorConvergence(previous_csp_distribution_parameters[0],
                                                             csp_distribution_parameters[0],
                                                             0.01,
                                                             0.01)
        print(f"csp_dist_change {csp_mean}, {csp_max}")


        if csp_distribution_converged and i >= minSteps:
            break
        else:
            if i == maxEMSteps - 1:
                raise EMConvergenceFailureError()
            #
        #
    #
    dist = CSPDetectionDistribution(distances_squared_normalized, csp_mixture_weights, matching_mixture_weights, csp_distribution_parameters,
                                    nonMatching_distribution_parameters)
    samples = dist.sample((sampleSize,))
    print("CSP Median", dist.csp_distribution.median(), "CSP Mode", dist.csp_distribution.mode())
    positionProbs = calculatePositionProb(samples,distances_squared_normalized.shape).detach()
    print("Outputing final position probabilities")
    outputResults(positionProbs.numpy(),
                      dist.csp_posterior_probabilities.exp(),
                      (pd.read_csv(args.reference_peak_list,sep="\s+"),int(transposed),args.reference_cs_column_names),
                      # tuple of a pandas dataframe and the dimension (0 or 1) in the representation, and a list of the resonance columns
                      (pd.read_csv(args.target_peak_list, sep="\s+"), int(not transposed), args.target_cs_column_names),
                      # tuple of a pandas dataframe and the dimension (0 or 1) in the representation
                      output_directory/"Ref2Target_transferred.list",
                      output_directory/"Ref2Target_transferred_HC.list",
                      output_directory/"Match_probabilities.csv",
                      output_directory/"CSP_probabilities.csv",
                      0.50)

    print("Outputing plots")
    print((output_directory/"fittedDistributions.png").resolve())
    fig = buildPlot(positionProbs,
                  csp_mixture_weights.exp().detach().numpy(),
                  dist.no_csp_distribution,
                  dist.csp_distribution,
                  dist.non_matching_distribution,
                  distances_squared_normalized,
                  0.50)
    fig.savefig(output_directory/"fittedDistributions.png")
    if display_distributions:
        fig.show()

#









