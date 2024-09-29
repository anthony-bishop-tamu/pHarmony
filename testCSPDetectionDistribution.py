from CSPDetectionDistribution import CSPDetectionDistribution
import torch
import time
import cProfile
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
    nsamples = 100
    distances = calculateDistanceMatrix(200, 180)
    print(distances)

    assignment_parameters= torch.ones(distances.shape[0],distances.shape[1],2)
    CSP_parameters= torch.tensor([10, 20], dtype=torch.float)
    non_matching_parameters = torch.tensor([1.87, 390],dtype=torch.float32)

    CSPDist = CSPDetectionDistribution(distances, assignment_parameters, CSP_parameters, non_matching_parameters)

    cProfile.run('CSPDist.sample((nsamples,))','output.prof')
'''for i in [100, 200, 500, 1000, 2000, 5000, 10000]:
    sample_start = time.time()
    samples = CSPDist.sample((i,))
    sample_end = time.time()
    logProb_start = time.time()
    logProbs = CSPDist.log_prob(samples)
    logProb_end = time.time()
    print(f"numSamples:  {i}  Sample Time (s): {sample_end-sample_start}  Time per sample (s) {(sample_end-sample_start)/i} LogProbTime (s): {logProb_end-logProb_start} LogProbPerSample(s): {(logProb_end-logProb_start)/i}")
#'''


