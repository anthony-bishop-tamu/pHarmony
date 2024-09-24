from CSPDetectionDistribution import CSPDetectionDistribution
import torch
import time
def calculatePositionProb(samples,logProbs,shape):
    probs = torch.zeros(shape)

    probVector = (logProbs - logProbs.logsumexp(dim=0)).exp()
    rowMissing = torch.zeros(shape[0])
    colMissing = torch.zeros(shape[1])
    idx = 0
    for sample in samples:
        for i in range(sample.shape[0]):
            order = sample[i,0]
            match = sample[i,1]
            if order < shape[0]:
                if match == - 1:
                    rowMissing[order] = +1*probVector[idx]
                else:
                    probs[order, match - shape[0]] += 1*probVector[idx]

            else:
                if match == - 1:
                    colMissing[order-shape[0]] += 1 * probVector[idx]
                else:
                    probs[match, order - shape[0]] += 1 * probVector[idx]
        #
        idx+=1
    #
    return probs, rowMissing, colMissing
#
def calculateDistanceMatrix(dim: int, noCSP: int):
    matchingDistances = torch.zeros(dim)

    scaling = 10
    torch.set_printoptions(3, sci_mode=False)
    CSP = dim-noCSP
    chi2Dist = torch.distributions.Chi2(1)
    weibullDist = torch.distributions.Weibull(390.44, 1.87)
    gaussianDist = torch.distributions.Normal(loc=10, scale=3)

    distances = weibullDist.sample((dim,dim))

    matchingDistances[:noCSP] = chi2Dist.sample((noCSP,))
    if CSP > 0:
        matchingDistances[noCSP:dim] = gaussianDist.sample((dim-noCSP,))

    distances[torch.arange(dim), torch.arange(dim)] = matchingDistances
    return distances
#

if __name__ == '__main__':

    torch.manual_seed(42)
    nsamples = 1000
    distances = calculateDistanceMatrix(500, 450)
    print(distances)

    assignment_parameters= torch.ones(distances.shape[0],distances.shape[1],2)
    CSP_parameters= torch.tensor([10, 20], dtype=torch.float)
    non_matching_parameters = torch.tensor([1.87, 390],dtype=torch.float32)

    CSPDist = CSPDetectionDistribution(distances, assignment_parameters, CSP_parameters, non_matching_parameters)


for sampleSize in [ 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    start = time.time()
    samples = CSPDist.sample((sampleSize,))
    end = time.time()

    elapsed_time = end - start
    print(f"Sample Size: {sampleSize} Elapsed time: {elapsed_time:.6f} seconds")

