from src.PeakMatcher.ProposalNet import ProposalNet
from src.PeakMatcher.PeakMatcher import (initalizeAllComponents,
                                         getPeakPositionsFromFile,
                                         calculateDistancesSquaredNormalized,
                                         calculateBetaParametersFromMeanAndVariance)
from src.PeakMatcher.CSPDetectionDistribution import CSPDetectionDistribution
import importlib_resources
from pathlib import Path
import torch

def generateCSPSampler(reference_list: Path,
                       target_list: Path):

    reference_positions, reference_peak_list = getPeakPositionsFromFile(reference_list,
                                                                          ['w1', 'w2'],
                                                                          fixedError=[0.03, 0.003])
    target_positions, target_peak_list = getPeakPositionsFromFile(target_list,
                                                                          ['w1', 'w2'],
                                                                          fixedError=[0.03, 0.003])
    distances_squared_normalized = calculateDistancesSquaredNormalized(reference_positions,target_positions,offset=torch.zeros((2,)))

    (csp_distribution,
     non_matching_distribution,
     csp_conditional_mixture_weights,
     matching_mixture_weights) = initalizeAllComponents(distances_squared_normalized,2,2000,29)

    CSPSampler = CSPDetectionDistribution(distances_squared_normalized,
                                          csp_conditional_mixture_weights,
                                          matching_mixture_weights,
                                          calculateBetaParametersFromMeanAndVariance(mean=0.001,variance=0.001**2).log(),
                                          csp_distribution,
                                          non_matching_distribution)
    return CSPSampler
#

data_directory =  importlib_resources.files('Tests.TestData')

#case1
reference_list = data_directory/'case1'/'IL1B_FragmentScreen_Reference_1.list'
target_list = data_directory/'case1'/'IL1B_FragmentScreen_Fragment_232.list'
output_directory = data_directory/'case1'/"test_output"

CSPSampler = generateCSPSampler(reference_list,target_list)
CSPSampler.sample((100,))
print("Done")
