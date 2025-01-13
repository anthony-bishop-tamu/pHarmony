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
        print(matching_probabilities.logsumexp(dim=0).exp().max(),matching_probabilities.logsumexp(dim=1).exp().max())

    initial_weights = torch.stack(((csp_conditional_assignments+matching_probabilities.unsqueeze(-1))[:,:,0],
                                   (csp_conditional_assignments+matching_probabilities.unsqueeze(-1))[:,:,1],
                                   1.0-matching_probabilities),dim=2)
    initial_weights = (initial_weights - initial_weights.logsumexp(dim=2,keepdim=True)).exp() #enforce normalization for intial weights

    no_csp_weights = initial_weights[:,:,0]
    csp_weights = initial_weights[:,:,1]
    no_matching_weights = initial_weights[:,:,2]

    eval_grid = torch.from_numpy(np.linspace(np.log(0.0005),np.log(max(distances.flatten().numpy()))+0.0005,100)).exp()#hyperparameter

    csp_distribution = LogTransformedKDEDensity(distances.flatten(),eval_grid,csp_weights.flatten())
    no_matching_distribution = LogTransformedKDEDensity(distances.flatten(),eval_grid,no_matching_weights.flatten())

    fig, ax = plt.subplots()
    hist = ax.hist(distances.flatten().numpy(),bins=eval_grid.numpy(),density=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(eval_grid.numpy(),no_matching_distribution.log_prob(eval_grid).exp().numpy())
   # no_matching_distribution.weigh_by_prior(torch.ones_like(eval_grid)/distances.max())
   # ax.plot(eval_grid.numpy(),no_matching_distribution.log_prob(eval_grid).exp().numpy())

    fig.show()

    csp_conditional_mixture_weights = (initial_weights[:,:,:2].sum(dim=(0,1))/initial_weights[:,:,0:2].sum()).log()
    matching_mixture_weights = torch.stack(((initial_weights[:,:,0:2]).sum(), (1.0-initial_weights[:,:,0:2]).sum()),dim=0)
    matching_mixture_weights /= matching_mixture_weights.sum()

    return csp_distribution, no_matching_distribution, csp_conditional_mixture_weights, matching_mixture_weights
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
              csp_distribution: torch.distributions.Distribution,
              non_matching_distribution: torch.distributions.Distribution,
              sampleSize: int,
              learning_rate: float,
              output_directory: Path,
              display_distributions: bool = True):


        dist = CSPDetectionDistribution(distances,
                                        csp_mixture_weights,
                                        matching_mixture_weights,
                                        csp_distribution,
                                        non_matching_distribution)
        #expectation step
        samples = dist.sample((sampleSize,))
        positionProbs = calculatePositionProb(samples, distances_squared_normalized.shape).detach()
        # Calculate new mixture weights
        csp_mixture_weights, matching_mixture_weights = calculateMixtureWeights(dist.csp_posterior_probabilities,positionProbs,csp_mixture_priors,matching_mixture_priors)
        #maximization step
        new_csp_distribution = LogTransformedKDEDensity(distances.flatten(),csp_distribution.eval_grid,(dist.csp_posterior_probabilities[:,:,1].exp()*positionProbs).flatten())
        new_non_matching_distribution = non_matching_distribution

        dist = CSPDetectionDistribution(distances,
                                        csp_mixture_weights.log(),
                                        matching_mixture_weights.log(),
                                        csp_distribution,
                                        non_matching_distribution)

        loss = EM_minimization_function(samples,dist,csp_mixture_weights.log(),matching_mixture_weights.log(),csp_mixture_priors,matching_mixture_priors)
        print("Loss: ", loss.item())
        if display_distributions:
            fig = buildPlot(positionProbs,
                        csp_mixture_weights.detach().numpy(),
                        dist.no_csp_distribution,
                        new_csp_distribution,
                        new_non_matching_distribution,
                        "fittedDistributions.png",
                        distances_squared_normalized,
                        0.50)
        
            fig.show()
        #
        return samples, csp_mixture_weights.log(), matching_mixture_weights.log(), new_csp_distribution, new_non_matching_distribution
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
    #Get intial KDEs
    initial_csp_distribution, initial_non_matching_distribution, initial_csp_mixture_weights, initial_matching_mixture_weights = initalizeAllComponents(distances_squared_normalized)

    expected_no_csp_ratio = 1.0 - args.expected_fraction_csp
    no_csp_std = args.expected_fraction_csp #Std deviation is arbitrarily set to being the same as the expected fraction csp

    csp_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_no_csp_ratio,variance=args.variance_scale_fraction_csp*no_csp_std**2)  #[no csp, csp ] (Given a match!)

    expected_missing_ratio = args.expected_fraction_missing
    expected_match_ratio = min(distances_squared_normalized.shape)/distances_squared_normalized.numel()
    match_std = expected_match_ratio*expected_missing_ratio
    matching_mixture_priors = calculateBetaParametersFromMeanAndVariance(mean=expected_match_ratio,variance=args.variance_scale_fraction_missing*match_std**2)  #[matching, nonmatching)

    initial_non_matching_distribution = torch.distributions.Uniform(0.0005,distances_squared_normalized.max()+0.0005)

    # get appropriate sample size
    sampleSize = distances_squared_normalized.shape[0]
    maxTries = 12
    samples=()
    for i in range(maxTries):
        print(f"Trying Sample Size: {sampleSize}")
        dist = CSPDetectionDistribution(distances_squared_normalized,
                                        initial_csp_mixture_weights,
                                        initial_matching_mixture_weights,
                                        initial_csp_distribution,
                                        initial_non_matching_distribution)
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
    csp_mixture_weights = initial_csp_mixture_weights
    matching_mixture_weights = initial_matching_mixture_weights
    csp_distribution = initial_csp_distribution
    non_matching_distribution = initial_non_matching_distribution
    for i in range(maxEMSteps):
        previous_dist = dist
        previous_matching_probs = positionProb

        samples, csp_mixture_weights, matching_mixture_weights,csp_distribution, non_matching_distribution = runEMStep(
                  distances_squared_normalized,
                  csp_mixture_weights,
                  matching_mixture_weights,
                  csp_mixture_priors,
                  matching_mixture_priors,
                  csp_distribution,
                  non_matching_distribution,
                  sampleSize,
                  learning_rate)
        dist = CSPDetectionDistribution(distances_squared_normalized, csp_mixture_weights, matching_mixture_weights,
                                        csp_distribution,
                                        non_matching_distribution)
        positionProbs = calculatePositionProb(samples, distances_squared_normalized.shape).detach()

        csp_distribution_converged,csp_mean,csp_max = verifyTensorConvergence(dist.csp_posterior_probabilities.exp(),
                                                             previous_dist.csp_posterior_probabilities.exp(),
                                                             0.05,
                                                             0.05)
        nonMatching_distribution_converged, nonMatching_mean, nonMatching_max = verifyTensorConvergence(previous_matching_probs,
                                                                     positionProbs,
                                                                     0.05,
                                                                     0.05)
        print(f"csp_dist_change {csp_mean}, {csp_max}, nonMatching_dist_change {nonMatching_mean}, {nonMatching_max}")


        if csp_distribution_converged and i >= minSteps:
            break
        else:
            if i == maxEMSteps - 1:
                raise EMConvergenceFailureError()
            #
        #
    #
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









