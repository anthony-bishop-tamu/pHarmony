import dis

from PeakMatchingDistribution import PeakMatchingDistribution
import torch
from torch.autograd import gradcheck
import scipy.optimize as opt
import numpy as np
import math


#dist = PeakMatchingDistribution(distances,parameters)

def sampleDistribution(distances,size):
    dist = PeakMatchingDistribution(distances)
    samples = []
    for i in range(size):
        print(f"Sample:{i}")
        order,match,likelihood = dist.sample()
        samples.append((order,match,likelihood))
        print(likelihood)
        r_likelihood = dist.calculateLogLikelihood((order,match,likelihood))
    #
    return samples
#
def getSampleStats(distances,sampleSize, rescale=True):
    samples = sampleDistribution(distances, sampleSize)

    sum = torch.zeros(distances.shape)
    factors = torch.ones(sampleSize)/sampleSize
    if rescale:
        for i in range(sampleSize):
            factors[i] = samples[i][2]
        #
        factors -= factors.mean()
        factors = factors.exp()
        factors /= factors.sum()

    for idx in range(sampleSize):
        order, match, likelihood = samples[idx]

        for i in range(0, len(order)):
            if (order[i] < len(order)):
                sum[order[i], match[i] - len(order)] += factors[idx]
            else:
                sum[match[i], order[i] - len(order)] += factors[idx]
        #
    #
    return sum
#
if __name__ == '__main__':
    size = 50
    distances = torch.ones(size,size) *5 - torch.eye(size,size)*3
    torch.set_printoptions(precision=3, sci_mode=False)

    sum = getSampleStats(distances, 1000)
    print(sum)

