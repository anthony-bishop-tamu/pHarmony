import os

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
from Frechet import Frechet, UniformDistanceSquared
from pathlib import Path
import copy
import time
from torch.profiler import profile, record_function, ProfilerActivity
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

    return torch.tensor([alpha, beta],dtype=torch.float32)
def initalizeAllComponents(distances):
    csp_conditional_assignments = torch.ones_like(distances)*0.1
    csp_conditional_assignments[distances < 5] = 0.9
    csp_conditional_assignments = torch.stack((csp_conditional_assignments,1.0-csp_conditional_assignments),dim=2).log()

    matching_probabilities = torch.ones_like(distances)*0.1/distances.shape[1]
    matching_probabilities[torch.arange(distances.shape[0]),torch.argmin(distances, dim=-1)] = 0.9
    matching_probabilities=matching_probabilities.log()
    #sinkhorn iterate
    for i in range(100):
        matching_probabilities -= matching_probabilities.logsumexp(dim=0,keepdim=True)
        matching_probabilities -= matching_probabilities.logsumexp(dim=1,keepdim=True)
        #print(matching_probabilities.logsumexp(dim=0).exp().max(),matching_probabilities.logsumexp(dim=1).exp().max())

    initial_weights = torch.stack(((csp_conditional_assignments+matching_probabilities.unsqueeze(-1))[:,:,0],
                                   (csp_conditional_assignments+matching_probabilities.unsqueeze(-1))[:,:,1],
                                   1.0-matching_probabilities),dim=2)
    initial_weights = (initial_weights - initial_weights.logsumexp(dim=2,keepdim=True)).exp() #enforce normalization for intial weights

    no_csp_weights = initial_weights[:,:,0]
    csp_weights = initial_weights[:,:,1]
    no_matching_weights = initial_weights[:,:,2]

    csp_distribution = Frechet(alpha=torch.tensor([2.0],dtype=torch.float32),
                                     scale=torch.tensor([30], dtype=torch.float32))

    non_matching_distribution = UniformDistanceSquared(dim=torch.tensor([2.0],dtype=torch.float32),
                                                       Rmax=distances.max(dim=-1)[0])

    #non_matching_distribution = torch.distributions.Uniform(0,distances.max()+0.005)


    csp_conditional_mixture_weights = (initial_weights[:,:,:2].sum(dim=(0,1))/initial_weights[:,:,0:2].sum())
    matching_mixture_weights = torch.stack(((initial_weights[:,:,0:2]).sum(), (1.0-initial_weights[:,:,0:2]).sum()),dim=0)
    matching_mixture_weights /= matching_mixture_weights.sum()

    return csp_distribution, non_matching_distribution, csp_conditional_mixture_weights, matching_mixture_weights
#
def getPeakPositionsFromFile(filename, cs_cols, uncertaintycols=None, fixedError=None):
    df = pd.read_csv(filename,sep="\s+")
    positions = df[cs_cols].to_numpy(dtype=np.float32)
    if positions.shape[0] == 0:
        raise RuntimeError(f"No peaks detected in File: {filename}")
    if uncertaintycols is not None:
        uncertainties = df[uncertaintycols].to_numpy(dtype=np.float32)
    elif fixedError is not None:
        uncertainties = np.zeros_like(positions)
        uncertainties[:,:] = np.array(fixedError,dtype=np.float32)[np.newaxis,:]
    else:
        raise ValueError("Must specify either uncertaintycols or fixedError")
    #
    return torch.from_numpy(np.stack((positions,uncertainties),axis=2)),df
#
def calculateMixtureWeights(csp_posterior_probabilities: torch.Tensor,
                            matching_posterior_probabilities: torch.Tensor,
                            csp_mixture_weight_priors: torch.Tensor,
                            matching_mixture_weight_priors: torch.Tensor,
                            missing_mixture_weight_priors: torch.Tensor) -> torch.Tensor:
    #csp_scale = matching_posterior_probabilities.sum()
    #matching_scale = matching_posterior_probabilities.numel()
    clamp_min = 1E-6

    pseudo_csp_posterior_probabilities = csp_posterior_probabilities.exp().clone()

    #pseudo_csp_posterior_probabilities = csp_posterior_probabilities.exp().detach() + clamp_min
    #pseudo_csp_posterior_probabilities /= pseudo_csp_posterior_probabilities.sum(dim=-1,keepdim=True)


    csp_mixture_weights = (pseudo_csp_posterior_probabilities*matching_posterior_probabilities.unsqueeze(-1)).detach()
    csp_mixture_weights = (csp_mixture_weights.sum(dim=(0,1)) + (csp_mixture_weight_priors))/(matching_posterior_probabilities.sum() + (csp_mixture_weight_priors).sum())
    matching_mixture_weights = (torch.tensor([matching_posterior_probabilities.detach().sum(),(1-matching_posterior_probabilities.detach()).sum()])
                                +(matching_mixture_weight_priors))/(matching_posterior_probabilities.numel() + matching_mixture_weight_priors.sum())
    missing_mixture_weights = (torch.tensor([matching_posterior_probabilities.detach().sum(),(1-matching_posterior_probabilities.detach().sum(dim=-1)).sum()])
                                +(missing_mixture_weight_priors))/(matching_posterior_probabilities.shape[0] + missing_mixture_weight_priors.sum())
    if csp_mixture_weights[1] < 1E-3:
        csp_mixture_weights = torch.tensor([1.0-1E-3,1E-3],dtype=torch.float32)

    assert (1 >= csp_mixture_weights).all() and (csp_mixture_weights >= 0).all()
    assert (1 >= matching_mixture_weights).all() and (matching_mixture_weights >= 0).all()
    assert (1 >= missing_mixture_weights).all() and (missing_mixture_weights >= 0).all()
    return csp_mixture_weights, matching_mixture_weights,missing_mixture_weights
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
                            torch.tensor([0.01],dtype=torch.float32),
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
def EM_minimization_function(samples, dist: CSPDetectionDistribution,
                             csp_mixture_weights: torch.Tensor,
                             matching_mixture_weights: torch.Tensor,
                             missing_mixture_weights: torch.Tensor,
                             csp_mixture_priors: torch.Tensor,
                             matching_mixture_priors: torch.Tensor,
                             missing_mixture_priors: torch.Tensor,):

    logLikelihoodTerm = dist.log_prob(samples).sum()#/(samples.numel()*dist._distances.numel())

    #positionProb = calculatePositionProb(samples, dist._distances.shape)

    scale_regularization = torch.relu((1.0 - dist.csp_distribution.scale)*10)**6
    alpha_regularization = torch.relu((0 - dist.csp_distribution.alpha)*10)**6
    median_regularization = torch.relu((0 - dist.csp_distribution.median())*10)**6
    quantile_regularization = torch.relu((0 - dist.csp_distribution.quantile(torch.tensor([0.05])))*10)**6
    if dist.csp_distribution.alpha.item() > 2:
        variance_regularization = torch.relu((100-dist.csp_distribution.variance())*10)**6
    else:
        variance_regularization = 0

    loss = (-1 * logLikelihoodTerm +
            -1*((csp_mixture_priors-1.0)*csp_mixture_weights).sum()+
            -1*((matching_mixture_priors-1.0)*matching_mixture_weights).sum() +
            -1*((missing_mixture_priors - 1.0) * missing_mixture_weights).sum() +
            scale_regularization + alpha_regularization + median_regularization + quantile_regularization + variance_regularization)
    assert loss.isfinite().all()
    return loss
#
def optimizeOffSet(reference_peak_positions: torch.Tensor,
                   target_peak_positions: torch.Tensor,
                   offset: torch.Tensor,
                   matching_probabilities: torch.Tensor,
                   csp_probabilities: torch.Tensor,
                   learning_rate: float,
                   gradient_convergence: float):
    optimizer = torch.optim.LBFGS([offset], line_search_fn='strong_wolfe', lr=0.01)
    maxIterators = 10000
    prevLoss = torch.finfo(torch.float32).max
    previous_offset = offset.detach().clone()
    for i in range(maxIterators):
        def closure():
            optimizer.zero_grad()
            distances = calculateDistancesSquaredNormalized(reference_peak_positions, target_peak_positions, offset)
            no_csp_matches = (csp_probabilities[:,:,0]*matching_probabilities) > 0.95
            if no_csp_matches.sum() > 5:
                loss = distances[no_csp_matches].sum()
            else:
                selection_mask =  matching_probabilities > 0.99
                loss = torch.median(distances[selection_mask])
            #
            loss.backward() #minimize the squared distance of the matching no csp peaks (normalized by the confidence of the matching peaks
            return loss
        #
        loss = optimizer.step(closure)

        print(f"OffSet step {i} Loss: ", loss.item(), prevLoss-loss.item(), offset.grad, offset)
        if not (offset.isfinite().all() and offset.grad.isfinite().all()):
            with torch.no_grad():
                offset[:] = previous_offset[:]
            optimizer = torch.optim.LBFGS([offset], lr=optimizer.param_groups[0]['lr'] * 0.5,line_search_fn='strong_wolfe')
            print("Lowering Learning rate")
            continue
        elif abs(prevLoss - loss.item()) < 1e-5:
            break

        #
        previous_offset = offset.detach().clone()
        prevLoss = loss.item()

    distances = calculateDistancesSquaredNormalized(reference_peak_positions, target_peak_positions, offset)
    return distances

def maximization(samples: tuple,
                 distances: torch.tensor,
                 csp_mixture_weights: torch.tensor,
                 matching_mixture_weights: torch.tensor,
                 missing_mixture_weights: torch.tensor,
                 csp_mixture_priors: torch.tensor,
                 matching_mixture_priors: torch.tensor,
                 missing_mixture_priors: torch.tensor,
                 csp_distribution,
                 optimization_list: list,
                 no_match_distribution,
                 learning_rate: float,
                 gradient_convergence: float):


    #optimizer = torch.optim.Adam([csp_assignment_params, csp_distribution_params], lr=1E-3)
    optimizer = torch.optim.Adam(optimization_list, lr=learning_rate)
    maxIterators = 10000
    prevLoss = torch.finfo(torch.float32).max
    previous_alpha = csp_distribution.alpha.detach().clone()
    previous_scale = csp_distribution.scale.detach().clone()
    for i in range(maxIterators):
        optimizer.zero_grad()
        dist = CSPDetectionDistribution(distances, csp_mixture_weights, matching_mixture_weights, missing_mixture_weights, csp_distribution,
                                        no_match_distribution)
        loss = EM_minimization_function(samples, dist,
                                        csp_mixture_weights,matching_mixture_weights, missing_mixture_weights,
                                        csp_mixture_priors,matching_mixture_priors,missing_mixture_priors)
        loss.backward()
        if prevLoss > loss.item():
            previous_alpha = csp_distribution.alpha.detach().clone()
            previous_scale = csp_distribution.scale.detach().clone()
        optimizer.step()
        if i % 1 == 0:
            print(f"Step {i} Loss: ", loss.item(), prevLoss-loss.item(), csp_distribution.alpha.item(), csp_distribution.scale.item(),
              csp_distribution.alpha.grad.item(), csp_distribution.scale.grad.item(), optimizer.param_groups[0]['lr'])
        #
        if not (torch.tensor([csp_distribution.alpha,csp_distribution.scale,csp_distribution.alpha.grad,csp_distribution.scale.grad]).isfinite().all() and
            csp_distribution.alpha.item() > 0 and csp_distribution.scale.item() > 0 and prevLoss-loss.item() >= -1E-3):
            with torch.no_grad():
                csp_distribution.alpha[0] = previous_alpha[0]
                csp_distribution.scale[0] = previous_scale[0]
            optimizer = torch.optim.Adam(optimization_list, lr=optimizer.param_groups[0]['lr'] * 0.5)

            print("Lowering Learning rate")
            continue
        elif prevLoss - loss.item() < 1e-7 and (torch.abs(torch.tensor([csp_distribution.alpha.grad,csp_distribution.scale.grad])) < gradient_convergence).all():
            break

        prevLoss = loss.item()
       # if i % 100 == 1 and i > 100:
       #     optimizer = torch.optim.Adam(optimization_list, lr=optimizer.param_groups[0]['lr'] * 2)
        #


    return distances
def runEMStep(distances: torch.tensor,
              csp_mixture_weights: torch.tensor,
              matching_mixture_weights: torch.tensor,
              missing_mixture_weights: torch.tensor,
              csp_mixture_priors: torch.tensor,
              matching_mixture_priors: torch.tensor,
              missing_mixture_priors: torch.tensor,
              csp_distribution: torch.distributions.Distribution,
              non_matching_distribution: torch.distributions.Distribution,
              sampleSize: int,
              learning_rate: float,
              gradient_convergence: float,
              display_distributions: bool = False):

        dist = CSPDetectionDistribution(distances,
                                        csp_mixture_weights,
                                        matching_mixture_weights,
                                        missing_mixture_weights,
                                        csp_distribution,
                                        non_matching_distribution)
        #expectation step
        samples = dist.sample((sampleSize,))
        positionProbs = calculatePositionProb(samples, distances.shape).detach()

        # Calculate new mixture weights
        csp_mixture_weights, matching_mixture_weights,missing_mixture_weights = calculateMixtureWeights(dist.csp_posterior_probabilities,positionProbs,
                                                                                                        csp_mixture_priors,matching_mixture_priors,missing_mixture_priors)
        #maximization step
        maximization(samples,
                     distances.detach(),
                     csp_mixture_weights.log(),
                     matching_mixture_weights.log(),
                     missing_mixture_weights.log(),
                     csp_mixture_priors,
                     matching_mixture_priors,
                     missing_mixture_priors,
                     csp_distribution,
                     [csp_distribution.alpha,csp_distribution.scale],
                     non_matching_distribution,
                     learning_rate,
                     gradient_convergence)
        new_non_matching_distribution = non_matching_distribution
        dist = CSPDetectionDistribution(distances,
                                        csp_mixture_weights.log(),
                                        matching_mixture_weights.log(),
                                        missing_mixture_weights.log(),
                                        csp_distribution,
                                        non_matching_distribution)

        loss = EM_minimization_function(samples,dist,
                                        csp_mixture_weights.log(),matching_mixture_weights.log(), missing_mixture_weights.log(),
                                        csp_mixture_priors, matching_mixture_priors, missing_mixture_priors)
        print("Loss: ", loss.item())
        if display_distributions:
            fig = buildPlot(positionProbs,
                        csp_mixture_weights.detach().numpy(),
                        dist.no_csp_distribution,
                        csp_distribution,
                        new_non_matching_distribution,
                        distances,
                        0.50)
        
            fig.show()
        #
        return samples, csp_mixture_weights.log(), matching_mixture_weights.log(), missing_mixture_weights.log(), csp_distribution, new_non_matching_distribution
#
def runEM(distances_squared_normalized: torch.tensor,
              initial_csp_mixture_weights: torch.tensor,
              initial_matching_mixture_weights: torch.tensor,
              initial_missing_mixture_weights: torch.tensor,
              csp_mixture_priors: torch.tensor,
              matching_mixture_priors: torch.tensor,
              missing_mixture_priors: torch.tensor,
              initial_csp_distribution: torch.distributions.Distribution,
              initial_non_matching_distribution: torch.distributions.Distribution,
              sampleSize: int,
              dist: CSPDetectionDistribution,
              initial_matching_probs,
              learning_rate: float,
              gradient_convergence: float,
              display_distributions: bool = False):

    minSteps = 2
    maxEMSteps = 1000
    csp_mixture_weights = initial_csp_mixture_weights
    matching_mixture_weights = initial_matching_mixture_weights
    missing_mixture_weights = initial_missing_mixture_weights
    csp_distribution = initial_csp_distribution
    non_matching_distribution = initial_non_matching_distribution
    matching_probs = initial_matching_probs
    for i in range(maxEMSteps):
        previous_dist = dist
        previous_matching_probs = matching_probs
        samples, csp_mixture_weights, matching_mixture_weights, missing_mixture_weights, csp_distribution, non_matching_distribution = runEMStep(
            distances_squared_normalized,
            csp_mixture_weights,
            matching_mixture_weights,
            missing_mixture_weights,
            csp_mixture_priors,
            matching_mixture_priors,
            missing_mixture_priors,
            csp_distribution,
            non_matching_distribution,
            sampleSize,
            learning_rate,
            gradient_convergence)

        dist = CSPDetectionDistribution(distances_squared_normalized, csp_mixture_weights, matching_mixture_weights,
                                        missing_mixture_weights,
                                        csp_distribution,
                                        non_matching_distribution)
        matching_probs = calculatePositionProb(samples, distances_squared_normalized.shape).detach()

        csp_distribution_converged, csp_mean, csp_max = verifyTensorConvergence(dist.csp_posterior_probabilities.exp(),
                                                                                previous_dist.csp_posterior_probabilities.exp(),
                                                                                0.05,
                                                                                0.05)
        nonMatching_distribution_converged, nonMatching_mean, nonMatching_max = verifyTensorConvergence(
            previous_matching_probs,
            matching_probs,
            0.05,
            0.05)
        print(f"csp_dist_change {csp_mean}, {csp_max}, nonMatching_dist_change {nonMatching_mean}, {nonMatching_max}")

        if csp_distribution_converged and i >= minSteps and nonMatching_distribution_converged:
            break
        else:
            if i == maxEMSteps - 1:
                raise EMConvergenceFailureError()
            #
        #
    #
    return dist, matching_probs

def parseArguments():
    #torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list', required=True, type=Path, help='reference peak list filename')
    parser.add_argument( '--reference_cs_column_names', required=True, type=str, nargs='+', help='reference cs column names (e.g. \'w1\', \'w2\')')
    parser.add_argument('--target_peak_list', required=True, type=Path, help='target peak list filename')
    parser.add_argument('--target_cs_column_names', required=True, type=str, nargs='+', help='target cs column names (e.g. \'w1\', \'w2\')')
    parser.add_argument('--reference_peak_list_error', required=True, type=float, nargs='+', help='Uncertainty in each dimension for the reference peak list (e.g. \" 0.0015 0.015 \" for a 2D HSQC [15N, 1H]')
    parser.add_argument('--target_peak_list_error', required=True, type=float, nargs='+', help='Uncertainty in each dimension for the target peak list (e.g. \" 0.0015, 0.015 \" for a 2D HSQC [15N, 1H]')
    #parser.add_argument("--minimum_distance", type=float, help="Minimum normalized distance between two peaks, all normalized distances lower than this value will be set to this value",default=0.005)
    parser.add_argument('--expected_fraction_csp', type=float, help="Estimate of the fraction of peaks expected to undergo a chemical shift perturbation", default=0.1)
    parser.add_argument("--variance_scale_fraction_csp",type=float, help="scaling factor for variance of the prior distribution of csp distribution weight", default=2.0)
    parser.add_argument('--expected_fraction_missing', type=float, help="Estimate of the fraction of peaks that you think will be missing between spectra", default=0.1)
    parser.add_argument("--variance_scale_fraction_missing",type=float, help="scaling factor for variance of the prior distribution of matching distribution weight", default=2.0)
    parser.add_argument("--gradient_convergence",type=float, help="Gradient convergence criterion", default=1E-5)
    parser.add_argument("--output_directory",type=Path,help="Directory path to output the results to", default="./peak_matcher_output")
    parser.add_argument( "--display_distributions", action='store_true', help="Display the distributions plots", )
    parser.add_argument( "--confidence_cutoff", type=lambda x: float(x) if 0.0 <= float(x) <= 1.0 else argparse.ArgumentTypeError("Value must be between 0.0 and 1.0."), help="Minimum posterior probability for outputing match", default=0.90)
    parser.add_argument( "--compute_reference_offset",action='store_true', help="Compute reference offset between peak lists", default=False)
    return parser.parse_args()

def calculateDistancesSquaredNormalized(reference_peak_positions: torch.Tensor,
                              target_peak_positions: torch.Tensor,
                              offset: torch.Tensor) -> np.ndarray:

    #Offset is added to the reference Peaks
    assert offset.shape[-1] == reference_peak_positions.shape[-2]
    assert reference_peak_positions.shape[-2] == target_peak_positions.shape[-2]

    components_distances_squared = torch.pow(
        (reference_peak_positions[:, :, 0]+offset).unsqueeze(dim=-2) - target_peak_positions[:, :, 0].unsqueeze(dim=0), 2)
    components_distances_squared_normalized = (
            components_distances_squared / (torch.pow(reference_peak_positions[:, :, 1].unsqueeze(dim=-2), 2) +
                                            torch.pow(target_peak_positions[:, :, 1].unsqueeze(dim=0), 2))
    )
    distances_squared_normalized = components_distances_squared_normalized.sum(dim=-1)
    with torch.no_grad():
        distances_squared_normalized[distances_squared_normalized == 0] = torch.finfo(torch.float32).eps

    return distances_squared_normalized
#
def MatchPeaks(reference_peak_positions: torch.Tensor,
               target_peak_positions: torch.Tensor,
               expected_fraction_csp: float = 0.1,
               variance_scale_fraction_csp: float = 2.0,
               expected_fraction_missing: float = 0.1,
               variance_scale_fraction_missing: float = 2.0,
               gradient_convergence: float = 1E-5,
               fixedOffset: torch.Tensor = None,
               output=None):

    if output is None:
        output = open(os.devnull,'w')

    #intialization
    if fixedOffset is None:
        offset = torch.zeros((reference_peak_positions.shape[-2],), dtype=torch.float32, requires_grad=True)
    else:
        offset = fixedOffset.detach().clone()

    assert offset.shape[-1] == reference_peak_positions.shape[-2]

    distances_squared_normalized = calculateDistancesSquaredNormalized(reference_peak_positions, target_peak_positions, offset)
    csp_distribution, non_matching_distribution, csp_mixture_weights, matching_mixture_weights = initalizeAllComponents(distances_squared_normalized.detach())

    #build priors
    expected_no_csp_ratio = 1.0 - expected_fraction_csp
    no_csp_std = args.expected_fraction_csp  # Std deviation is arbitrarily set to being the same as the expected fraction csp

    csp_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_no_csp_ratio,
                                                                    variance=variance_scale_fraction_csp * no_csp_std ** 2)  # [no csp, csp ] (Given a match!)

    expected_missing_ratio = expected_fraction_missing
    expected_match_ratio = min(distances_squared_normalized.shape) / distances_squared_normalized.numel()
    match_std = expected_match_ratio * expected_missing_ratio
    matching_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_match_ratio,
                                                                         variance=variance_scale_fraction_missing * match_std ** 2)  # [matching, nonmatching)
    missing_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=1.0 - expected_fraction_missing,
                                                                        variance=variance_scale_fraction_missing * expected_fraction_missing ** 2)
    initial_missing_mixture_weights = missing_mixture_priors.detach().clone()
    initial_matching_mixture_weights = matching_mixture_weights.detach().clone()
    initial_csp_mixture_weights = csp_mixture_weights.detach().clone()
    initial_csp_distribution = csp_distribution
    initial_non_matching_distribution = non_matching_distribution


    #GET SAMPLE SIZE
    sampleSize = distances_squared_normalized.shape[0]
    maxTries = 8
    samples=()
    for i in range(maxTries):
        print(f"Trying Sample Size: {sampleSize}",file=output)
        dist = CSPDetectionDistribution(distances_squared_normalized,
                                        initial_csp_mixture_weights,
                                        initial_matching_mixture_weights,
                                        initial_missing_mixture_weights,
                                        initial_csp_distribution,
                                        initial_non_matching_distribution)
        samples = dist.sample((sampleSize,))
        converged = validateSufficentSampling(samples, distances_squared_normalized.shape)
        matching_probs = calculatePositionProb(samples, distances_squared_normalized.shape).detach()
        if not converged:
            sampleSize *= 2
            if i == maxTries-1:
                raise SampleSizeToLargeError
        else:
            print(f"Sample Size: {sampleSize} is sufficient",file=output)
            break
        #
    #
    for i in range(maxTries):
    #RUN EM
    # run EM
        dist, matching_probs = runEM(distances_squared_normalized,
          dist.csp_mixture_weights,
          dist.matching_mixture_weights,
          dist.missing_mixture_weights,
          csp_mixture_priors,
          matching_mixture_priors,
          missing_mixture_priors,
          dist.csp_distribution,
          dist.non_matching_distribution,
          sampleSize,
          dist,
          matching_probs,
          learning_rate=1,
          gradient_convergence=gradient_convergence)
        if offset.requires_grad:
             previous_offset = offset.detach().clone()
             distances_squared_normalized = optimizeOffSet(reference_peak_positions,target_peak_positions,offset,matching_probs,dist.csp_posterior_probabilities.exp(),
                         learning_rate=1,gradient_convergence=gradient_convergence)
             offset_difference = torch.abs(previous_offset - offset)
             print(f"Offset_diference {offset_difference}")
             if (offset_difference/torch.sqrt(torch.mean((reference_peak_positions[:,:,1]**2),dim=0) + torch.mean(target_peak_positions[:,:,1]**2,dim=0)) < 0.1).all():
                 break

        else:
            break
        #

    #



    return dist, matching_probs.detach(), distances_squared_normalized,offset
#

if __name__ == "__main__":
    #torch.manual_seed(42)
    start_time = time.time()
    args = parseArguments()
    output_directory = args.output_directory
    output_directory.mkdir(exist_ok=True, parents=True)
    display_distributions = args.display_distributions

    reference_peak_positions, reference_peaks = getPeakPositionsFromFile(args.reference_peak_list,
                                               args.reference_cs_column_names,
                                               fixedError=args.reference_peak_list_error,)
    target_peak_positions, target_peaks = getPeakPositionsFromFile(args.target_peak_list,
                                            args.target_cs_column_names,
                                            fixedError=args.target_peak_list_error)

    if args.compute_reference_offset:
       offset = None
    else:
        offset = torch.zeros((reference_peak_positions.shape[-2],), dtype=torch.float32, requires_grad=True)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        posteriorMatchingDistribution, matchingProbabilities, distances_squared_normalized,offset = MatchPeaks(reference_peak_positions,
                                                                      target_peak_positions,
                                                                      args.expected_fraction_csp,
                                                                      args.variance_scale_fraction_csp,
                                                                      args.expected_fraction_missing,
                                                                      args.variance_scale_fraction_missing,
                                                                      args.gradient_convergence,
                                                                      offset,
                                                                      sys.stdout)


    name_stem = f"{args.reference_peak_list.name}_{args.target_peak_list.name}"
    outputResults(matchingProbabilities.numpy(),
                      posteriorMatchingDistribution.csp_posterior_probabilities.exp(),
                  (reference_peaks,args.reference_cs_column_names),
                   (target_peaks,args.target_cs_column_names),
                      output_directory/f"{name_stem}_transferred.txt",
                      output_directory/f"{name_stem}_transferred_HC.txt",
                      output_directory/f"{name_stem}_transferred.list",
                      output_directory/"Match_probabilities.csv",
                      output_directory/"CSP_probabilities.csv",
                      args.confidence_cutoff)

    print("Outputing plots")
    fig = buildPlot(matchingProbabilities,
                  posteriorMatchingDistribution.csp_mixture_weights.exp().detach().cpu().numpy(),
                  posteriorMatchingDistribution.no_csp_distribution,
                  posteriorMatchingDistribution.csp_distribution,
                  distances_squared_normalized.detach(),
                  0.50)
    fig.savefig(output_directory/f"{name_stem}_fittedDistributions.png")
    if display_distributions:
        fig.show()

    print(f"Computed Offset: {offset} ")
    print("Done")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time/60.0} min")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#









