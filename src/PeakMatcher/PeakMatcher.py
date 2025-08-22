import os

import scipy
import numpy as np
import torch
import pandas as pd
import sys
import argparse
from .CSPDetectionDistribution import CSPDetectionDistribution
from .OutputHandling import buildPlot, outputResults
from .Frechet import Frechet, UniformDistanceSquared, RegFrechet
from pathlib import Path
from torch.distributions import Beta
import time
import logging
from ._version import __version__
class SampleSizeToLargeError(Exception):
    pass
#
class EMConvergenceFailureError(Exception):
    pass

class NoPeaksFoundError(Exception):
    pass

class ArgumentError(Exception):
    pass
# 1. Choose a numeric value: between DEBUG (10) and INFO (20)
VERBOSE_LEVEL = 15

# 2. Register the name → value mapping
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

# 3. Add a helper method so you can call logger.verbose(...)
def verbose(self, msg, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, msg, args, **kwargs)

logging.Logger.verbose = verbose     # monkey-patch the class
def setup_logger(
    log_file = None,
    *,
    level: int = logging.INFO,
    overwrite: bool = True,
    keep_console: bool = True,
) -> logging.Logger:

    handlers = []

    if log_file is None:
        handlers.append(logging.StreamHandler(sys.stdout))
    else:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        handlers.append(logging.FileHandler(log_path, mode=mode, encoding="utf-8"))

        if keep_console:
            handlers.append(logging.StreamHandler(sys.stdout))

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,           # Python ≥3.8; wipes existing handlers
    )

    # Return a logger scoped to the caller’s module
    return logging.getLogger(__name__)
def calculateBetaParametersFromMeanAndVariance(mean, variance):
    assert 0 < mean and mean < 1
    assert 0 < variance
    mu = ((mean*(1-mean))/variance - 1)
    alpha = mean*mu
    beta = (1-mean)*mu

    return torch.tensor([alpha, beta],dtype=torch.float64)
def initalizeAllComponents(distances, dims, max_predicted_dm, max_CSP_count):
    csp_conditional_assignments = torch.ones_like(distances)*0.05
    csp_conditional_assignments[distances < 3] = 0.95
    csp_conditional_assignments = torch.stack((csp_conditional_assignments,1.0-csp_conditional_assignments),dim=2).log()

    matching_probabilities = torch.ones_like(distances)*0.01/distances.shape[1]
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

    #csp_distribution = RegFrechet(torch.tensor([1],dtype=torch.float64,requires_grad=True),
    #                                 torch.tensor([max_predicted_dm],dtype=torch.float64),
    #                                 torch.tensor([max_CSP_count],dtype=torch.float64))
    #csp_distribution = Frechet(csp_distribution.alpha.detach().clone().requires_grad_(True),csp_distribution.scale.detach().clone().requires_grad_(True))
    csp_distribution=Frechet(torch.tensor([1.0],requires_grad=True),torch.tensor([60.0],requires_grad=True))

    non_matching_distribution = UniformDistanceSquared(dim=torch.tensor(dims,dtype=torch.float64),
                                                       Rmax=(distances.max()/2).expand(distances.shape[0]))

    #non_matching_distribution = torch.distributions.Uniform(0,distances.max()+0.005)


    csp_conditional_mixture_weights = (initial_weights[:,:,:2].sum(dim=(0,1))/initial_weights[:,:,0:2].sum())
    matching_mixture_weights = torch.stack(((initial_weights[:,:,0:2]).sum(), (1.0-initial_weights[:,:,0:2]).sum()),dim=0)
    matching_mixture_weights /= matching_mixture_weights.sum()

    return csp_distribution, non_matching_distribution, csp_conditional_mixture_weights.log(), matching_mixture_weights.log()
#
def getPeakPositionsFromFile(filename, cs_cols, uncertaintycols=None, fixedError=None):
    df = pd.read_csv(filename,sep="\s+")
    positions = df[cs_cols].to_numpy(dtype=np.float64)
    if positions.shape[0] == 0:
        raise NoPeaksFoundError(f"No peaks detected in file")
    if uncertaintycols is not None:
        uncertainties = df[uncertaintycols].to_numpy(dtype=np.float64)
    elif fixedError is not None:
        uncertainties = np.zeros_like(positions)
        uncertainties[:,:] = np.array(fixedError,dtype=np.float64)[np.newaxis,:]
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
        csp_mixture_weights = torch.tensor([1.0-1E-3,1E-3],dtype=torch.float64)

    assert (1 >= csp_mixture_weights).all() and (csp_mixture_weights >= 0).all()
    assert (1 >= matching_mixture_weights).all() and (matching_mixture_weights >= 0).all()
    assert (1 >= missing_mixture_weights).all() and (missing_mixture_weights >= 0).all()
    return csp_mixture_weights, matching_mixture_weights,missing_mixture_weights
def verifyTensorConvergence(torchPreviousParameter: torch.tensor,
                               torchNewParameter: torch.tensor,
                               averageDeviation: torch.tensor,
                               maxDeviation: torch.tensor):
    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    difference = torchNewParameter - torchPreviousParameter
    converged = difference.abs().mean() < averageDeviation and difference.abs().max() < maxDeviation
    PEAK_MATCHER_LOGGER.info("CONVERGENCE?=%s MaxDeviation=%.3e Limit=%.3e", converged, difference.abs().max(), maxDeviation)
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
def validateSufficentSampling(sample1, sample2, shape: tuple) -> bool:
    firstHalf = calculatePositionProb(sample1, shape)
    secondHalf = calculatePositionProb(sample2, shape)
    converged, mean, max = verifyTensorConvergence(firstHalf,secondHalf,
                            torch.tensor([0.01],dtype=torch.float64),
                            torch.tensor([0.10],dtype=torch.float64))
    return converged
#
def calculateCSPDistParameters(quantile, cutoff, median):

    numerator = torch.log(-1*torch.log(1-quantile)) - torch.log(torch.log(torch.tensor([2.0])))
    denominator = torch.log(cutoff) - torch.log(median)

    k = numerator / denominator
    lam = median/torch.pow(torch.log(torch.tensor([2.0])),1.0/k)
    return torch.tensor([k,lam],dtype=torch.float64)
#
def calculateMaxD2FromCSP(csp: float, scaling_factors: torch.tensor, errors: torch.tensor) -> float:
    distances = csp/scaling_factors
    distances_normalized = distances/errors
    distances_normalized_squared = distances_normalized**2
    return torch.max(distances_normalized_squared).item()
def EM_minimization_function(samples, dist: CSPDetectionDistribution,
                             csp_mixture_priors: torch.Tensor,
                             matching_mixture_priors: torch.Tensor,
                             missing_mixture_priors: torch.Tensor,
                             max_predicted_dnm: float):

    csp_mixture_weights = dist.csp_mixture_weights
    matching_mixture_weights = dist.matching_mixture_weights
    missing_mixture_weights = dist.missing_mixture_weights

    logLikelihoodTerm = dist.log_prob(samples).sum()#/(samples.numel()*dist._distances.numel())

    #positionProb = calculatePositionProb(samples, dist._distances.shape)

    quantile_regularization = torch.relu((3 - dist.csp_distribution.quantile(torch.tensor([0.001])))*10)**6
    #quantile_regularization += torch.relu(max_predicted_dnm-dist.csp_distribution.quantile(torch.tensor([0.95])))**6
    #quantile_regularization = 0

    loss = (-1 * logLikelihoodTerm +
            -1*((csp_mixture_priors-1.0)*csp_mixture_weights).sum()+
            -1*((matching_mixture_priors-1.0)*matching_mixture_weights).sum() +
            -1*((missing_mixture_priors - 1.0) * missing_mixture_weights).sum() +
            quantile_regularization)
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
    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    optimizer = torch.optim.LBFGS([offset], line_search_fn='strong_wolfe', lr=0.01)
    maxIterators = 1000
    prevLoss = torch.finfo(torch.float64).max
    previous_offset = offset.detach().clone()
    for i in range(maxIterators):
        def closure():
            optimizer.zero_grad()
            distances = calculateDistancesSquaredNormalized(reference_peak_positions, target_peak_positions, offset)
            no_csp_matches = (csp_probabilities[:,:,0]*matching_probabilities) > 0.90
            if no_csp_matches.sum() > 5:
                loss = distances[no_csp_matches].sum()
            else:
                selection_mask =  matching_probabilities > 0.90
                loss = torch.median(distances[selection_mask])
            #
            loss.backward() #minimize the squared distance of the matching no csp peaks (normalized by the confidence of the matching peaks
            return loss
        #
        loss = optimizer.step(closure)

        PEAK_MATCHER_LOGGER.verbose("Offset Optimization step=%i, change=%d, grad=%d, offset=%s", loss.item(), prevLoss - loss.item(), offset.grad, offset)
        if not (offset.isfinite().all() and offset.grad.isfinite().all()):
            with torch.no_grad():
                offset[:] = previous_offset[:]
            optimizer = torch.optim.LBFGS([offset], lr=optimizer.param_groups[0]['lr'] * 0.5,line_search_fn='strong_wolfe')
            PEAK_MATCHER_LOGGER.verbose("Lowering Learning rate")
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
                 dist: CSPDetectionDistribution,
                 csp_mixture_priors: torch.tensor,
                 matching_mixture_priors: torch.tensor,
                 missing_mixture_priors: torch.tensor,
                 max_predicted_dm: float,
                 learning_rate: float,
                 gradient_convergence: float):

    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    # optimizer = torch.optim.Adam([csp_assignment_params, csp_distribution_params], lr=1E-3)
    csp_distribution = dist.csp_distribution
    optimizer = torch.optim.AdamW([csp_distribution.alpha, csp_distribution.scale], lr=learning_rate,weight_decay=1e-2)
    maxIterators = 10000
    prevLoss = torch.finfo(torch.float64).max
    previous_alpha = csp_distribution.alpha.detach().clone()
    previous_scale = csp_distribution.scale.detach().clone()
    for i in range(maxIterators):
        optimizer.zero_grad()
        dist._detach()
        loss = EM_minimization_function(samples, dist,
                                        csp_mixture_priors, matching_mixture_priors, missing_mixture_priors,
                                        max_predicted_dm)
        loss.backward()
        if prevLoss > loss.item():
            previous_alpha = csp_distribution.alpha.detach().clone()
            previous_scale = csp_distribution.scale.detach().clone()
        optimizer.step()

        # with torch.no_grad():
        #    csp_distribution.alpha.clamp_(min=1.0)
        if i % 1 == 0:
            PEAK_MATCHER_LOGGER.info(
                "Step=%6d Loss=%12.3e, diff=%12.3e, csp_alpha=%12.3e, csp_scale=%12.3e csp_alpha_grad=%12.3e csp_scale_grad=%12.3e, lr=%12.3e csp_dist_var= %12.3e max_predicted_dnm=%12.3e",
                i, loss.item(), prevLoss - loss.item(), csp_distribution.alpha.item(), csp_distribution.scale.item(),
                csp_distribution.alpha.grad.item(), csp_distribution.scale.grad.item(), optimizer.param_groups[0]['lr'], csp_distribution.variance(),
                max_predicted_dm)
        #
        if not (torch.tensor([csp_distribution.alpha, csp_distribution.alpha.grad]).isfinite().all() and
                csp_distribution.alpha.item() > 0 and csp_distribution.scale.item() > 0 and prevLoss - loss.item() >= -1E-3):
            with torch.no_grad():
                csp_distribution.alpha[0] = previous_alpha[0]
                csp_distribution.scale[0] = previous_scale[0]
            optimizer = torch.optim.AdamW([csp_distribution.alpha,csp_distribution.scale], lr=optimizer.param_groups[0]['lr'] * 0.5, weight_decay=1e-2)

            PEAK_MATCHER_LOGGER.verbose("Lowering Learning rate")
            continue
        elif prevLoss - loss.item() < 1e-7 and (
                torch.abs(torch.tensor([csp_distribution.alpha.grad])) < gradient_convergence).all():
            break
        #csp_distribution = RegFrechet(csp_distribution.alpha, csp_distribution.max_val, csp_distribution.n)
        # csp_distribution = Frechet(csp_distribution.alpha,csp_distribution.scale)
        prevLoss = loss.item()
    # if i % 100 == 1 and i > 100:
    #     optimizer = torch.optim.Adam(optimization_list, lr=optimizer.param_groups[0]['lr'] * 2)
    #
    return dist
def runEMStep(distances: torch.tensor,
              dist: CSPDetectionDistribution,
              csp_mixture_priors: torch.tensor,
              matching_mixture_priors: torch.tensor,
              missing_mixture_priors: torch.tensor,
              max_predicted_dnm: float,
              sampleSize: int,
              learning_rate: float,
              gradient_convergence: float,
              display_distributions: bool = False):

        #csp_distribution = dist.csp_distribution.clone()

        #expectation step
        samples = determineSampleSize(sampleSize,dist)
        positionProbs = calculatePositionProb(samples, distances.shape).detach()

        # Calculate new mixture weights
        csp_mixture_weights, matching_mixture_weights,missing_mixture_weights = calculateMixtureWeights(dist.csp_posterior_probabilities,positionProbs,
                                                                                                        csp_mixture_priors,matching_mixture_priors,missing_mixture_priors)
        #maximization step
        dist = maximization(samples,
                     distances.detach(),
                     dist,
                     csp_mixture_priors,
                     matching_mixture_priors,
                     missing_mixture_priors,
                     max_predicted_dnm,
                     learning_rate,
                     gradient_convergence)

        dist.csp_mixture_weights = csp_mixture_weights.log()
        dist.matching_mixture_weights = matching_mixture_weights.log()
        dist.missing_mixture_weights = missing_mixture_weights.log()



        if display_distributions:
            fig = buildPlot(positionProbs,
                        csp_mixture_weights.detach().numpy(),
                        dist.no_csp_distribution,
                        dist.csp_distribution,
                        dist.non_matching_distribution,
                        distances,
                        0.50)
        
            fig.show()
        #
        return samples, dist
#
def runEM(distances_squared_normalized: torch.tensor,
              initial_csp_mixture_weights: torch.tensor,
              initial_matching_mixture_weights: torch.tensor,
              initial_missing_mixture_weights: torch.tensor,
              csp_mixture_priors: torch.tensor,
              matching_mixture_priors: torch.tensor,
              missing_mixture_priors: torch.tensor,
              max_predicted_dnm: float,
              initial_csp_distribution: torch.distributions.Distribution,
              initial_non_matching_distribution: torch.distributions.Distribution,
              learning_rate: float,
              gradient_convergence: float,
              display_distributions: bool = False):

    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    minSteps = 0
    maxEMSteps = 1000
    csp_mixture_weights = initial_csp_mixture_weights
    matching_mixture_weights = initial_matching_mixture_weights
    missing_mixture_weights = initial_missing_mixture_weights
    csp_distribution = initial_csp_distribution
    non_matching_distribution = initial_non_matching_distribution

    dist = CSPDetectionDistribution(distances_squared_normalized,
                                    max_predicted_dnm,
                                    csp_mixture_weights,
                                    matching_mixture_weights,
                                    missing_mixture_weights,
                                    csp_distribution,
                                    non_matching_distribution)

    inital_sample = determineSampleSize(distances_squared_normalized.shape[0],dist)
    sampleSize = len(inital_sample)
    matching_probs = calculatePositionProb(inital_sample,distances_squared_normalized.shape)
    for i in range(maxEMSteps):
        previous_dist = dist.clone()

        previous_matching_probs = matching_probs
        samples, dist = runEMStep(
            distances_squared_normalized,
            dist,
            csp_mixture_priors,
            matching_mixture_priors,
            missing_mixture_priors,
            max_predicted_dnm,
            sampleSize,
            learning_rate,
            gradient_convergence)

        sampleSize = len(samples)

        matching_probs = calculatePositionProb(samples, distances_squared_normalized.shape).detach()

        PEAK_MATCHER_LOGGER.info("CSP posterior convergence")
        csp_distribution_converged, csp_mean, csp_max = verifyTensorConvergence(dist.csp_posterior_probabilities.exp()[:,:,1]*matching_probs,
                                                                                previous_dist.csp_posterior_probabilities.exp()[:,:,1]*matching_probs,
                                                                                0.05,
                                                                                0.05)
        PEAK_MATCHER_LOGGER.info("Matching posterior convergence")
        nonMatching_distribution_converged, nonMatching_mean, nonMatching_max = verifyTensorConvergence(
            previous_matching_probs,
            matching_probs,
            0.05,
            0.10)
#        PEAK_MATCHER_LOGGER.info("CSP distribution convergece")
        PEAK_MATCHER_LOGGER.info(f"CSP_dist: { dist.csp_distribution.alpha}, {dist.csp_distribution.scale }")
        csp_dist_converged, csp_dist_mean, csp_dist_max = verifyTensorConvergence(torch.cat([dist.csp_distribution.alpha, dist.csp_distribution.scale], dim=0),
                                                                                  torch.cat([previous_dist.csp_distribution.alpha, previous_dist.csp_distribution.scale], dim=0),
                                                                                  0.05,0.05)



        if csp_distribution_converged and i >= minSteps and nonMatching_distribution_converged and csp_dist_converged:
            PEAK_MATCHER_LOGGER.info("Converged?: True")
            break
        else:
            PEAK_MATCHER_LOGGER.info("Converged?: False")
            if i == maxEMSteps - 1:
                raise EMConvergenceFailureError()
            #
        #
    #
    return dist, matching_probs

def isPositive(x):
    if float(x) > 0:
        return float(x)
    else:
        argparse.ArgumentTypeError("Value must be > than 0")
def isBetween0And1(x):
    if 0.0 <= float(x) <= 1.0:
        return float(x)
    else:
        argparse.ArgumentTypeError("Value must be between 0.0 and 1.0.")

def parseArguments():
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list', required=True, type=Path, help='reference peak list filename')
    parser.add_argument( '--reference_cs_column_names', required=True, type=str, nargs='+', help='reference cs column names (e.g. \'w1\', \'w2\')')
    parser.add_argument('--target_peak_list', required=True, type=Path, help='target peak list filename')
    parser.add_argument('--target_cs_column_names', required=True, type=str, nargs='+', help='target cs column names (e.g. \'w1\', \'w2\')')
    parser.add_argument('--reference_peak_list_error', required=True, type=isPositive, nargs='+', help='Uncertainty in each dimension for the reference peak list (e.g. \" 0.0015 0.015 \" for a 2D HSQC [15N, 1H]')
    parser.add_argument('--target_peak_list_error', required=True, type=isPositive, nargs='+', help='Uncertainty in each dimension for the target peak list (e.g. \" 0.0015, 0.015 \" for a 2D HSQC [15N, 1H]')
    #parser.add_argument("--minimum_distance", type=float, help="Minimum normalized distance between two peaks, all normalized distances lower than this value will be set to this value",default=0.005)
    parser.add_argument('--expected_fraction_csp', type=isBetween0And1, help="Estimate of the fraction of peaks expected to undergo a chemical shift perturbation", default=0.05)
    parser.add_argument("--variance_scale_fraction_csp",type=isPositive, help="scaling factor for variance of the prior distribution of csp distribution weight", default=1.0)
    parser.add_argument('--expected_fraction_missing', type=isBetween0And1, help="Estimate of the fraction of peaks that you think will be missing between spectra", default=0.02)
    parser.add_argument("--variance_scale_fraction_missing",type=isPositive, help="scaling factor for variance of the prior distribution of matching distribution weight", default=2.0)
    parser.add_argument("--expected_max_csp", type=isPositive, help="Estimate of the maximum expected CSP (ppm); Default is in units of proton ppm", default=0.2)
    parser.add_argument("--gradient_convergence",type=isPositive, help="Gradient convergence criterion", default=1E-5)
    parser.add_argument("--output_directory",type=Path,help="Directory path to output the results to", default="./peak_matcher_output")
    parser.add_argument( "--display_distributions", action='store_true', help="Display the distributions plots", )
    parser.add_argument( "--confidence_cutoff", type=isBetween0And1 , help="Minimum posterior probability for outputing match", default=0.90)
    parser.add_argument( "--compute_reference_offset",action='store_true', help="Compute reference offset between peak lists", default=False)
    parser.add_argument("--log_file",action='store_true', help="Write log file", default=False)
    parser.add_argument( "--CSP_scaling_factors",type=isPositive, nargs="+", help="nucleus scaling factors for CSP calculation (e.g. 0.252 0.101 1.00 for a C, N, H dimensional experiment", required=True)

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
        distances_squared_normalized[distances_squared_normalized == 0] = torch.finfo(torch.float64).eps

    return distances_squared_normalized
#
def calculateReferencePeakDistances(reference_peak_positions: torch.Tensor, csp_scaling_factors: torch.Tensor) -> torch.Tensor:
    distances_squared  = (reference_peak_positions.unsqueeze(0) - reference_peak_positions.unsqueeze(1))/csp_scaling_factors.unsqueeze(0).unsqueeze(0)
    distances_squared = torch.square(distances_squared).sum(dim=-1)
    return torch.sqrt(distances_squared)
def determineSampleSize(startingSample: int,dist: CSPDetectionDistribution):
    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    if isinstance(startingSample,int):
        startingSample = dist.sample((startingSample,))
    #
    size = startingSample.shape[0]
    sample1 = startingSample
    sample2 = dist.sample((size,))
    PEAK_MATCHER_LOGGER.info(f"Validating Sample Size: {size} ")
    maxTries = 6
    i = 1
    while not validateSufficentSampling(sample1,sample2,dist.distances.shape) and i < maxTries:
        size *= 2
        PEAK_MATCHER_LOGGER.info(f"Increasing Sample Size to: {size}")
        sample1 = dist.sample((int(size),))
        sample2 = dist.sample((int(size),))
        i += 1
    #
    PEAK_MATCHER_LOGGER.info(f"New sample size {size}")
    return sample1
#
def MatchPeaks(reference_peak_positions: torch.Tensor,
               target_peak_positions: torch.Tensor,
               expected_fraction_csp,
               variance_scale_fraction_csp,
               expected_fraction_missing,
               variance_scale_fraction_missing,
               max_predicted_dnm,
               gradient_convergence,
               fixedOffset: torch.Tensor = None):


    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    #intialization
    if fixedOffset is None:
        offset = torch.zeros((reference_peak_positions.shape[-2],), dtype=torch.float64, requires_grad=True)
    else:
        offset = fixedOffset.detach().clone()

    assert offset.shape[-1] == reference_peak_positions.shape[-2]

    dims = reference_peak_positions.shape[-2]

    distances_squared_normalized = calculateDistancesSquaredNormalized(reference_peak_positions, target_peak_positions, offset)


    #build priors
    no_csp_std = expected_fraction_csp  # Std deviation is arbitrarily set to being the same as the expected fraction csp

    csp_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_fraction_csp,
                                                                    variance=variance_scale_fraction_csp * no_csp_std ** 2)  # [no csp, csp ] (Given a match!)

    expected_missing_ratio = expected_fraction_missing
    expected_match_ratio = min(distances_squared_normalized.shape) / distances_squared_normalized.numel()
    match_std = expected_match_ratio * expected_missing_ratio
    matching_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_match_ratio,
                                                                         variance=variance_scale_fraction_missing * match_std ** 2)  # [matching, nonmatching)
    fraction_possible_matched_rows = min(1,float(distances_squared_normalized.shape[1])/distances_squared_normalized.shape[0])
    max_fraction_missing_rows = 1 - fraction_possible_matched_rows
    expected_fraction_missing_rows =  max(expected_fraction_missing,1-(1-max_fraction_missing_rows)*(1-expected_fraction_missing))
    #expected_fraction_missing_rows = expected_fraction_missing
    PEAK_MATCHER_LOGGER.info(
        f"fraction_possilbe_matched_rows: {fraction_possible_matched_rows}, expected_fraction_missing_rows: {expected_fraction_missing_rows}, expected_fraction_missing: {expected_fraction_missing}")
    missing_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=1.0 - expected_fraction_missing_rows,
                                                                        variance=variance_scale_fraction_missing * expected_fraction_missing_rows ** 2)
    PEAK_MATCHER_LOGGER.info(f" MissingMixture_priors: {missing_mixture_priors}")


    max_fraction_csp = scipy.stats.beta.ppf(0.95,csp_mixture_priors[0],csp_mixture_priors[1])
    max_CSP_count=(1-expected_fraction_missing_rows)*distances_squared_normalized.shape[0]*max_fraction_csp
    csp_distribution, non_matching_distribution, csp_mixture_weights, matching_mixture_weights = initalizeAllComponents(
        distances_squared_normalized.detach(), dims, max_predicted_dnm, max_CSP_count=max_CSP_count)
    PEAK_MATCHER_LOGGER.info(f"max_predicted_dnm: {max_predicted_dnm}, Max expected CSPs{max_CSP_count}")


    initial_missing_mixture_weights = missing_mixture_priors.log().detach().clone()
    initial_matching_mixture_weights = matching_mixture_priors.log().detach().clone()
    initial_csp_mixture_weights = csp_mixture_priors.log().detach().clone()
    initial_csp_distribution = csp_distribution
    initial_non_matching_distribution = non_matching_distribution


    maxTries = 8
    for i in range(maxTries):
    #RUN EM
    # run EM
        dist, matching_probs = runEM(distances_squared_normalized,
          initial_csp_mixture_weights,
          initial_matching_mixture_weights,
          initial_missing_mixture_weights,
          csp_mixture_priors,
          matching_mixture_priors,
          missing_mixture_priors,
          max_predicted_dnm,
          initial_csp_distribution,
          initial_non_matching_distribution,
          learning_rate=1E-2,
          gradient_convergence=gradient_convergence)
        if offset.requires_grad:
             previous_offset = offset.detach().clone()
             distances_squared_normalized = optimizeOffSet(reference_peak_positions,target_peak_positions,offset,matching_probs,dist.csp_posterior_probabilities.exp(),
                         learning_rate=1,gradient_convergence=gradient_convergence)
             offset_difference = torch.abs(previous_offset - offset)
             PEAK_MATCHER_LOGGER.info(f"Offset_diference {offset_difference}")
             if (offset_difference/torch.sqrt(torch.mean((reference_peak_positions[:,:,1]**2),dim=0) + torch.mean(target_peak_positions[:,:,1]**2,dim=0)) < 0.1).all():
                 break

        else:
            break
        #

    #



    return dist, matching_probs.detach(), distances_squared_normalized,offset
#
def standalone_match_peaks(reference_peak_list: Path,
                            reference_cs_column_names: list,
                            reference_peak_list_error: list,
                            target_peak_list: Path,
                            target_cs_column_names: list,
                            target_peak_list_error: list,
                            output_directory: Path,
                            expected_fraction_csp: float,
                            variance_scale_fraction_csp: float,
                            expected_fraction_missing: float,
                            variance_scale_fraction_missing: float,
                            expected_max_csp: float,
                            gradient_convergence: float,
                            compute_reference_offset: bool,
                            display_distributions: bool,
                            confidence_cutoff: float,
                            CSP_scaling_factors: list,
                            log_file: bool = False,
                            log_level: int = logging.INFO,
                            ):

    #validate inputs
    if log_file:
        log_output = output_directory / "log.txt"
        PEAK_MATCHER_LOGGER = setup_logger(log_output, level=log_level)
    else:
        PEAK_MATCHER_LOGGER = setup_logger(None, level=log_level)

    start_time = time.time()
    PEAK_MATCHER_LOGGER.info(f"Version: {__version__}")

    try:
        dims = len(reference_cs_column_names)
        if len(reference_cs_column_names) != dims:
            raise ArgumentError("number of reference cs columns (dimensions) must equal number of target cs columns (dimensions)")
        if len(reference_peak_list_error) != dims:
            raise ArgumentError("Reference peak list error: Exactly one value for error must be provided for each reference dimension")
        if len(target_peak_list_error) != dims:
            raise ArgumentError("target peak list error: Exactly one value for error must be provided for each target dimension")
        if CSP_scaling_factors is not None and len(CSP_scaling_factors) != dims:
            raise ArgumentError("Must provide a CSP scaling factor for each matched dimension (omit flag to skip CSP calculation)")



        output_directory = output_directory.resolve()
        output_directory.mkdir(exist_ok=True, parents=True)


        try:
            reference_peak_positions, reference_peaks = getPeakPositionsFromFile(reference_peak_list,
                                                                             reference_cs_column_names,
                                                                             fixedError=reference_peak_list_error)
            target_peak_positions, target_peaks = getPeakPositionsFromFile(target_peak_list,
                                                                           target_cs_column_names,
                                                                           fixedError=target_peak_list_error)
        except Exception as e:
            PEAK_MATCHER_LOGGER.exception(f"Exception raised while parsing peak positions")
            raise e


        if compute_reference_offset:
            offset = None
        else:
            offset = torch.zeros((reference_peak_positions.shape[-2],), dtype=torch.float64, requires_grad=True)

        max_predicted_dnm = calculateMaxD2FromCSP(expected_max_csp,torch.tensor(CSP_scaling_factors,dtype=torch.float),torch.tensor(reference_peak_list_error,dtype=torch.float))

        # with profile(activities=[ProfilerActivity.CPU]) as prof:
        posteriorMatchingDistribution, matchingProbabilities, distances_squared_normalized, offset = MatchPeaks(
            reference_peak_positions,
            target_peak_positions,
            expected_fraction_csp,
            variance_scale_fraction_csp,
            expected_fraction_missing,
            variance_scale_fraction_missing,
            max_predicted_dnm,
            gradient_convergence,
            offset)

        name_stem = f"{reference_peak_list.name}_{target_peak_list.name}"
        outputResults(matchingProbabilities.numpy(),
                      posteriorMatchingDistribution.csp_posterior_probabilities.exp(),
                      distances_squared_normalized.detach().numpy(),
                      (reference_peaks, reference_cs_column_names),
                      (target_peaks, target_cs_column_names),
                      output_directory / f"{name_stem}_transferred.csv",
                      output_directory / f"{name_stem}_transferred_HC.csv",
                      output_directory / f"{name_stem}_transferred.list",
                      output_directory / "Match_probabilities.csv",
                      output_directory / "CSP_probabilities.csv",
                      CSP_scaling_factors,
                      confidence_cutoff)

        PEAK_MATCHER_LOGGER.info("Outputing plots")
        fig = buildPlot(matchingProbabilities,
                        posteriorMatchingDistribution.csp_mixture_weights.exp().detach().cpu().numpy(),
                        posteriorMatchingDistribution.no_csp_distribution,
                        posteriorMatchingDistribution.csp_distribution,
                        distances_squared_normalized.detach(),
                        0.50)
        PEAK_MATCHER_LOGGER.info(f"Output Directory: {output_directory}")
        fig.savefig(output_directory / f"{name_stem}_fittedDistributions.png")
        if display_distributions:
            fig.show()

        PEAK_MATCHER_LOGGER.info(f"Final Offset: {offset} ")
        PEAK_MATCHER_LOGGER.info("Done")
        end_time = time.time()
        elapsed_time = end_time - start_time
        PEAK_MATCHER_LOGGER.info(f"Elapsed Time: {elapsed_time / 60.0:0.2f} min")
    except Exception as e:
        PEAK_MATCHER_LOGGER.exception(f"FatalError")
        PEAK_MATCHER_LOGGER.exception(f"{e}")


#
def main():
    args = parseArguments()
    standalone_match_peaks(args.reference_peak_list,
                            args.reference_cs_column_names,
                            args.reference_peak_list_error,
                            args.target_peak_list,
                            args.target_cs_column_names,
                            args.target_peak_list_error,
                            args.output_directory,
                            args.expected_fraction_csp,
                            args.variance_scale_fraction_csp,
                            args.expected_fraction_missing,
                            args.variance_scale_fraction_missing,
                            args.expected_max_csp,
                            args.gradient_convergence,
                            args.compute_reference_offset,
                            args.display_distributions,
                            args.confidence_cutoff,
                            args.CSP_scaling_factors,
                            args.log_file)










