import numpy as np
import torch
from PeakMatcher.CSPDetectionDistribution import CSPDetectionDistribution
from PeakMatcher.Frechet import Frechet, UniformDistanceSquared
import logging
from tqdm import tqdm
import math
from PeakMatcher._log import configure_logging
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

    alpha = max(1.0,alpha)
    beta = max(1.0,beta)

    return torch.tensor([alpha, beta],dtype=torch.float64)
def initalizeAllComponents(distances, dims):
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

    csp_distribution=Frechet(torch.tensor([2.0],requires_grad=True),torch.tensor([30.0],requires_grad=True))

    non_matching_distribution = UniformDistanceSquared(dim=torch.tensor(dims,dtype=torch.float64))



    csp_conditional_mixture_weights = (initial_weights[:,:,:2].sum(dim=(0,1))/initial_weights[:,:,0:2].sum())

    return csp_distribution, non_matching_distribution, csp_conditional_mixture_weights.log()
def calculateMixtureWeights(csp_posterior_probabilities: torch.Tensor,
                            matching_posterior_probabilities: torch.Tensor,
                            csp_mixture_weight_priors: torch.Tensor) -> torch.Tensor:

    #csp_mixture_weight_priors = torch.zeros_like(csp_mixture_weight_priors) #REMOVINGINFLUENCE OF PRIOR
    pseudo_csp_posterior_probabilities = csp_posterior_probabilities.exp().clone()

    csp_mixture_weights = (pseudo_csp_posterior_probabilities*matching_posterior_probabilities.unsqueeze(-1)).detach()
    csp_mixture_weights = (csp_mixture_weights.sum(dim=(0,1)) + (csp_mixture_weight_priors))/(matching_posterior_probabilities.sum() + (csp_mixture_weight_priors).sum())

    #if csp_mixture_weights[1] < 1E-3:
    #    csp_mixture_weights = torch.tensor([1.0-1E-3,1E-3],dtype=torch.float64)

    assert (1 >= csp_mixture_weights).all() and (csp_mixture_weights >= 0).all()
    return csp_mixture_weights
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
def EM_minimization_function(samples, dist: CSPDetectionDistribution,
                             csp_mixture_priors: torch.Tensor,
                             max_predicted_dnm: float):

    csp_mixture_weights = dist.csp_mixture_weights

    logLikelihoodTerm = dist.log_prob(samples).sum()

    #quantile_regularization = torch.relu((3 - dist.csp_distribution.quantile(torch.tensor([0.001])))*10)**6
    val = dist.csp_distribution.log_prob(max_predicted_dnm)
    valm1 = dist.csp_distribution.log_prob(math.exp(math.log(max_predicted_dnm)-1))
    if hasattr(dist,'min_alpha'):
        sharpness_reg = (torch.relu(dist.min_alpha - dist.csp_distribution.alpha)*10)**6
    else:
        sharpness_reg = 0
    loss = (-1 * logLikelihoodTerm +
            -1*((csp_mixture_priors-1.0)*csp_mixture_weights).sum() +
            sharpness_reg)
    assert loss.isfinite().all()
    return loss
#
def maximization(samples: tuple,
                 dist: CSPDetectionDistribution,
                 csp_mixture_priors: torch.tensor,
                 max_predicted_dm: float,
                 learning_rate: float,
                 gradient_convergence: float):

    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    # optimizer = torch.optim.Adam([csp_assignment_params, csp_distribution_params], lr=1E-3)
    csp_distribution = dist.csp_distribution
    optimizer = torch.optim.AdamW([csp_distribution.alpha, csp_distribution.scale ], lr=learning_rate,weight_decay=1e-2)
    maxIterators = 1000
    prevLoss = torch.finfo(torch.float64).max
    previous_alpha = csp_distribution.alpha.detach().clone()
    previous_scale = csp_distribution.scale.detach().clone()
    for i in tqdm(range(maxIterators),desc="Optimizing Parameters", total=maxIterators):
        optimizer.zero_grad()
        dist._detach()
        loss = EM_minimization_function(samples, dist,
                                        csp_mixture_priors,
                                        max_predicted_dm)
        loss.backward()
        if prevLoss > loss.item():
            previous_alpha = csp_distribution.alpha.detach().clone()
            previous_scale = csp_distribution.scale.detach().clone()
        if torch.tensor([csp_distribution.alpha.grad, csp_distribution.scale.grad]).isfinite().all():
            optimizer.step()
        else:
            optimizer = torch.optim.AdamW([csp_distribution.alpha, csp_distribution.scale],
                                          lr=optimizer.param_groups[0]['lr'] * 0.5, weight_decay=1e-2)

            PEAK_MATCHER_LOGGER.verbose("Lowering Learning rate")
            continue

        # with torch.no_grad():
        #    csp_distribution.alpha.clamp_(min=1.0)
        if i % 1 == 0:
            PEAK_MATCHER_LOGGER.verbose(
                "Step=%6d Loss=%12.3e, diff=%12.3e, csp_alpha=%12.3e, csp_scale=%12.3e csp_alpha_grad=%12.3e csp_scale_grad=%12.3e, lr=%12.3e csp_dist_var= %12.3e max_predicted_dnm=%12.3e",
                i, loss.item(), prevLoss - loss.item(), csp_distribution.alpha.item(), csp_distribution.scale.item(),
                csp_distribution.alpha.grad.item(), csp_distribution.scale.grad.item(),  optimizer.param_groups[0]['lr'], csp_distribution.variance(),
                max_predicted_dm)
        #
        if not (torch.tensor([csp_distribution.alpha, csp_distribution.alpha.grad, csp_distribution.scale, csp_distribution.scale.grad]).isfinite().all() and
                csp_distribution.alpha.item() > 0 and csp_distribution.scale.item() > 0 and prevLoss - loss.item() >= -1E-3):
            with torch.no_grad():
                csp_distribution.alpha[0] = previous_alpha[0]
                csp_distribution.scale[0] = previous_scale[0]
            optimizer = torch.optim.AdamW([csp_distribution.alpha, csp_distribution.scale ], lr=optimizer.param_groups[0]['lr'] * 0.5, weight_decay=1e-2)

            PEAK_MATCHER_LOGGER.verbose("Lowering Learning rate")
            continue
        elif prevLoss - loss.item() < 1e-7 and (
                torch.abs(torch.tensor([csp_distribution.alpha.grad ])) < gradient_convergence).all():
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
              max_predicted_dnm: float,
              sampleSize: int,
              learning_rate: float,
              gradient_convergence: float):

        #expectation step
        samples = determineSampleSize(sampleSize,dist)
        positionProbs = calculatePositionProb(samples, distances.shape).detach()

        # Calculate new mixture weights
        csp_mixture_weights = calculateMixtureWeights(dist.csp_posterior_probabilities,positionProbs,csp_mixture_priors)

        n_csps = max(5.0,(dist.csp_posterior_probabilities.softmax(dim=-1)[...,1] * positionProbs).sum().item())
        dist.csp_distribution = Frechet(dist.csp_distribution.alpha,dist.csp_distribution.scale)
        #maximization step
        dist = maximization(samples,
                     dist,
                     csp_mixture_priors,
                     max_predicted_dnm,
                     learning_rate,
                     gradient_convergence)

        dist.csp_mixture_weights = csp_mixture_weights.log()




        return samples, dist
#
def runEM(distances_squared_normalized: torch.tensor,
              ndim: int,
              initial_csp_mixture_weights: torch.tensor,
              csp_mixture_priors: torch.tensor,
              max_predicted_dnm: float,
              initial_csp_distribution: torch.distributions.Distribution,
              initial_non_matching_distribution: torch.distributions.Distribution,
              learning_rate: float,
              gradient_convergence: float,
              display_distributions: bool = False):

    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    minSteps = 0
    maxEMSteps = 20
    csp_mixture_weights = initial_csp_mixture_weights
    csp_distribution = initial_csp_distribution
    non_matching_distribution = initial_non_matching_distribution

    dist = CSPDetectionDistribution(distances_squared_normalized,
                                    ndim,
                                    torch.tensor([max_predicted_dnm],dtype=torch.float),
                                    csp_mixture_weights,
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
        #PEAK_MATCHER_LOGGER.info(f"CSP_dist: { dist.csp_distribution.alpha}, {dist.csp_distribution.scale }")
        #csp_dist_converged, csp_dist_mean, csp_dist_max = verifyTensorConvergence(torch.cat([dist.csp_distribution.alpha, dist.csp_distribution.scale], dim=0),
        #                                                                          torch.cat([previous_dist.csp_distribution.alpha, previous_dist.csp_distribution.scale], dim=0),
        #                                                                          0.05,0.05)


        if csp_distribution_converged and i >= minSteps and nonMatching_distribution_converged:
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


def calculateDistancesSquaredNormalized(reference_peak_positions: torch.Tensor,
                              target_peak_positions: torch.Tensor) -> np.ndarray:

    #Offset is added to the reference Peaks
    assert reference_peak_positions.shape[-2] == target_peak_positions.shape[-2]

    components_distances_squared = torch.pow(
        (reference_peak_positions[:, :, 0]).unsqueeze(dim=-2) - target_peak_positions[:, :, 0].unsqueeze(dim=0), 2)
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
    while not validateSufficentSampling(sample1,sample2,dist.distances.shape):
        size *= 2
        if size > dist.distances.shape[0]*1000:
            raise SampleSizeToLargeError(
                f"Stopped at sample size {size}: sample variance is still too high, likely due to inefficent beam searching; consider reducing expected max CSP")

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
               max_predicted_dnm,
               gradient_convergence):

    configure_logging(__name__,level="VERBOSE")
    PEAK_MATCHER_LOGGER = logging.getLogger(__name__)
    #intialization


    dims = reference_peak_positions.shape[-2]

    distances_squared_normalized = calculateDistancesSquaredNormalized(reference_peak_positions, target_peak_positions)

    #build priors
    no_csp_std = expected_fraction_csp  # Std deviation is arbitrarily set to being the same as the expected fraction csp

    csp_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=1.0-expected_fraction_csp,
                                                                    variance=variance_scale_fraction_csp * no_csp_std ** 2)  # [no csp, csp ] (Given a match!)


    csp_distribution, non_matching_distribution, csp_mixture_weights = initalizeAllComponents(
        distances_squared_normalized.detach(), dims)
    PEAK_MATCHER_LOGGER.info(f"max_predicted_dnm: {max_predicted_dnm}")


    initial_csp_mixture_weights = csp_mixture_priors.log().detach().clone()
    initial_csp_distribution = csp_distribution
    initial_non_matching_distribution = non_matching_distribution



    dist, matching_probs = runEM(distances_squared_normalized,
          dims,
          initial_csp_mixture_weights,
          csp_mixture_priors,
          max_predicted_dnm,
          initial_csp_distribution,
          initial_non_matching_distribution,
          learning_rate=1E-2,
          gradient_convergence=gradient_convergence)

    #



    return dist, matching_probs.detach(), distances_squared_normalized
#










