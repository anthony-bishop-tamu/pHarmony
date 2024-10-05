import numpy as np
import scipy.stats as stats
import torch
import pandas as pd
from EM_CSPDetectionDistribution import runEMStep
import sys
import argparse

def getInitialWeibullParameters(distances):
    distances = distances.flatten().numpy()
    shape,loc,scale = stats.weibull_min.fit(distances,loc=0)
    return shape,scale
#
def getPeakPositionsFromFile(filename, cs_cols, uncertaintycols=None, fixedError=None):
    df = pd.read_csv(filename,sep="\s+")
    positions = df[cs_cols].to_numpy(dtype=np.float32)
    if uncertaintycols is not None:
        uncertainties = df[uncertaintycols].to_numpy(dtype=np.float32)
    elif fixedError is not None:
        uncertainties = np.zeros_like(positions)
        uncertainties[:,:] = np.array(fixedError,dtype=np.float32)[:, np.newaxis]
    else:
        raise ValueError("Must specify either uncertaintycols or fixedError")
    #
    return torch.from_numpy(np.stack((positions,uncertainties),axis=2))
#
def getInitialCSPDistributionParameters(distances: torch.tensor):
    assert distances.dim() == 2
    cutoff = 5.0
    closestMatches = distances.min(dim=-1)[0]:
    closestMatches = closestMatches[closestMatches > cutoff]

    assigmentParameters = torch.zeros(distances+(2,),dtype=torch.float32)
    #assignmentParameters are logits
    assigmentParameters[...,1] = (distances > 5)*3
    assigmentParameters[...,1] = ((3 < distances) & (distances < 5))*1
    assigmentParameters[...,1] = (distances < 3)* -2

    concentration,loc,scale = scipy.stats.weibull_min.fit(closestMatches.detach().numpy(),loc=0)
    #scale is the inverse of rate (i.e. rate = 1.0/scale)
    #loc is ignored as it is fixed at zero

    return concentration,scale,assigmentParameters
#

def verifyTensorConvergence(torchPreviousParameter: torch.tensor,
                               torchNewParameter: torch.tensor,
                               averageDeviation: torch.tensor,
                               maxDeviation: torch.tensor) -> bool:
    difference = torchNewParameter - torchPreviousParameter
    return difference.abs().mean() < averageDeviation and (difference.abs() < maxDeviation).all()
#
def calculatePositionProb(sample, weights, shape):
    positionProbs = torch.zeros(shape, dtype=torch.float32)
    rows = torch.arange(shape[0]).unsqueeze(0).expand(sample.shape[0], shape[0])
    weights_expanded = weights.unsqueeze(1).expand(sample.shape[0], shape[0]).flatten()
    positionProbs.index_put_((rows.flatten(), sample.flatten()), weights_expanded, accumulate=True)
    return positionProbs
#
def validateSufficentSampling(samples: tuple, shape: tuple) -> bool:
    matchings, weights = samples
    firstHalf = calculatePositionProb(matchings[::2,...],weights[::2,...], shape)
    secondHalf = calculatePositionProb(matchings[1::2,...],weights[1::2,...], shape)
    return verifyTensorConvergence(firstHalf,secondHalf,
                            torch.tensor([0.05],dtype=torch.float32),
                            torch.tensor([0.1],dtype=torch.float32))
#
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference_peak_list', type=str, help='reference peak list filename')
    parser.add_argument( 'reference_cs_column_names', type=str, nargs='+', help='reference cs column names')
    parser.add_argument('target_peak_list', type=str, help='target peak list filename')
    parser.add_argument('target_cs_column_names', type=str, nargs='+', help='target cs column names')


    return parser.parse_args()
if __name__ == "__main__":
    peakList_1 = sys.argv[0]
    args = parseArguments()
    reference_peaks = getPeakPositionsFromFile(args.reference_peak_list,
                                               args.reference_cs_column_names,
                                               [0.03,0.003])
    target_peaks = getPeakPositionsFromFile(args.reference_peak_list,
                                            args.target_peak_list,
                              [0.03,0.003])

    components_distances_squared = torch.pow(reference_peaks[:,:,0] - target_peaks[:,:,0].transpose(0,1),2)
    components_distances_squared_normalized = (
            components_distances_squared/(torch.pow(reference_peaks[:,:,1],2) +
                                          torch.pow(target_peaks[:,:,1].transpose(0,1),2))
    )
    distances_squared_normalized = components_distances_squared_normalized.sum(dim=-1)

    nonMatch_shape,nonMatch_scale = getInitialWeibullParameters(distances_squared_normalized)
    cspMatch_concentration, cspMatch_scale, initial_assignment_parameters = (
        getInitialCSPDistributionParameters(distances_squared_normalized))



