from test_peakMatchingDistribution import sampleDistribution, getSampleStats
from PeakMatchingDistribution import PeakMatchingDistribution
import torch

parameters = torch.eye(5)*20
'''distances = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [2.0, 1.0, 3.0, 4.0, 5.0],
                          [3.0, 2.0, 1.0, 4.0, 5.0],
                          [4.0, 3.0, 2.0, 1.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 1.0]])'''
distances = torch.ones((5,5))*5 - torch.eye(5)*4
#distances[4,4]=5
torch.set_printoptions(precision=3, sci_mode=False)
dist = PeakMatchingDistribution(distances)


