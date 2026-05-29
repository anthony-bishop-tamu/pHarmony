from pathlib import Path
import argparse
import time
import pandas as pd
import torch
from pHarmony._pHarmony import MatchPeaks
from pHarmony.OutputHandling import outputResults, buildPlot
import numpy as np
from pHarmony import __version__
from pHarmony._log import configure_logging, get_logger

class ArgumentError(Exception):
    pass
class NoPeaksFoundError(Exception):
    pass
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
        # torch.autograd.set_detect_anomaly(True)
        parser = argparse.ArgumentParser()
        parser.add_argument('--reference_peak_list', required=True, type=Path, help='reference peak list filename')
        parser.add_argument('--reference_cs_column_names', required=True, type=str, nargs='+',
                            help='reference cs column names (e.g. \'w1\', \'w2\')')
        parser.add_argument('--target_peak_list', required=True, type=Path, help='target peak list filename')
        parser.add_argument('--target_cs_column_names', required=True, type=str, nargs='+',
                            help='target cs column names (e.g. \'w1\', \'w2\')')
        parser.add_argument('--reference_peak_list_error', required=True, type=isPositive, nargs='+',
                            help='Uncertainty in each dimension for the reference peak list (e.g. \" 0.0015 0.015 \" for a 2D HSQC [15N, 1H]')
        parser.add_argument('--target_peak_list_error', required=True, type=isPositive, nargs='+',
                            help='Uncertainty in each dimension for the target peak list (e.g. \" 0.0015, 0.015 \" for a 2D HSQC [15N, 1H]')
        parser.add_argument('--expected_fraction_csp', type=isBetween0And1,
                            help="Estimate of the fraction of peaks expected to undergo a chemical shift perturbation",
                            default=0.1)
        parser.add_argument("--variance_scale_fraction_csp", type=isPositive,
                            help="scaling factor for variance of the prior distribution of csp distribution weight",
                            default=3.0)
        parser.add_argument("--expected_max_csp", type=isPositive,
                            help="Estimate of the maximum expected CSP (ppm); Default is in units of proton ppm",
                            default=0.1)
        parser.add_argument("--output_directory", type=Path, help="Directory path to output the results to",
                            default="./peak_matcher_output")
        parser.add_argument("--confidence_cutoff", type=isBetween0And1,
                            help="Minimum posterior probability for outputing match", default=0.95)
        parser.add_argument("--log_file", action='store_true', help="Write log file", default=False)
        parser.add_argument("--CSP_scaling_factors", type=isPositive, nargs="+",
                            help="nucleus scaling factors for CSP calculation (e.g. 0.252 0.101 1.00 for a C, N, H dimensional experiment",
                            required=True)

        return parser
def calculateMaxD2FromCSP(csp: float, scaling_factors: torch.tensor, errors: torch.tensor) -> float:
    distances = csp/scaling_factors
    distances_normalized = distances/errors
    distances_normalized_squared = distances_normalized**2
    return torch.min(distances_normalized_squared).item()
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
def run(args: argparse.Namespace):
    if args.log_file:
        log_output = args.output_directory / "log.txt"
        configure_logging(log_file=log_output, level="VERBOSE", overwrite=True)
    else:
        configure_logging(level="VERBOSE")

    logger = get_logger(__name__)

    start_time = time.time()
    logger.info(f"Version: {__version__}")

    try:
        dims = len(args.reference_cs_column_names)
        if len(args.reference_cs_column_names) != dims:
            raise ArgumentError("number of reference cs columns (dimensions) must equal number of target cs columns (dimensions)")
        if len(args.reference_peak_list_error) != dims:
            raise ArgumentError("Reference peak list error: Exactly one value for error must be provided for each reference dimension")
        if len(args.target_peak_list_error) != dims:
            raise ArgumentError("target peak list error: Exactly one value for error must be provided for each target dimension")
        if args.CSP_scaling_factors is not None and len(args.CSP_scaling_factors) != dims:
            raise ArgumentError("Must provide a CSP scaling factor for each matched dimension (omit flag to skip CSP calculation)")



        output_directory = args.output_directory.resolve()
        output_directory.mkdir(exist_ok=True, parents=True)


        try:
            reference_peak_positions, reference_peaks = getPeakPositionsFromFile(args.reference_peak_list,
                                                                             args.reference_cs_column_names,
                                                                             fixedError=args.reference_peak_list_error)
            target_peak_positions, target_peaks = getPeakPositionsFromFile(args.target_peak_list,
                                                                           args.target_cs_column_names,
                                                                           fixedError=args.target_peak_list_error)
        except Exception as e:
            logger.exception(f"Exception raised while parsing peak positions")
            raise e


        max_predicted_dnm = calculateMaxD2FromCSP(args.expected_max_csp,torch.tensor(args.CSP_scaling_factors,dtype=torch.float),torch.tensor(args.reference_peak_list_error,dtype=torch.float))

        # with profile(activities=[ProfilerActivity.CPU]) as prof:
        posteriorMatchingDistribution, matchingProbabilities, distances_squared_normalized = MatchPeaks(
            reference_peak_positions,
            target_peak_positions,
            args.expected_fraction_csp,
            args.variance_scale_fraction_csp,
            max_predicted_dnm)

        name_stem = f"{args.reference_peak_list.name}_{args.target_peak_list.name}"
        outputResults(matchingProbabilities.numpy(),
                      posteriorMatchingDistribution.csp_posterior_probabilities.exp(),
                      distances_squared_normalized.detach().numpy(),
                      (reference_peaks, args.reference_cs_column_names),
                      (target_peaks, args.target_cs_column_names),
                      output_directory / f"{name_stem}_transferred.csv",
                      output_directory / f"{name_stem}_transferred_HC.csv",
                      output_directory / f"{name_stem}_transferred.list",
                      output_directory / "Match_probabilities.csv",
                      output_directory / "CSP_probabilities.csv",
                      args.CSP_scaling_factors,
                      args.confidence_cutoff)

        logger.info("Outputing plots")
        fig = buildPlot(matchingProbabilities,
                        posteriorMatchingDistribution.csp_mixture_weights.exp().detach().cpu().numpy(),
                        posteriorMatchingDistribution.no_csp_distribution,
                        posteriorMatchingDistribution.csp_distribution,
                        posteriorMatchingDistribution.non_matching_distribution,
                        distances_squared_normalized.detach(),
                        0.50)
        logger.info(f"Output Directory: {output_directory}")
        fig.savefig(output_directory / f"{name_stem}_fittedDistributions.png",bbox_inches='tight')
        logger.info("Done")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed Time: {elapsed_time / 60.0:0.2f} min")
    except Exception as e:
        logger.exception(f"FatalError")
        logger.exception(f"{e}")
        return 1
    return 0
def main(argv=None) -> int:
    args = parseArguments().parse_args(argv)
    return run(args)
