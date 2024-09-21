from CSPDetectionDistribution import CSPDetectionDistribution
import torch

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
def calculateDistanceMatrix(dim: int, noCSP: int, minNoMatchDistance: float, maxNonMatchDistance: float):
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
    distances = calculateDistanceMatrix(5, 3, 3, 10)
    print(distances)

    CSPparameters = torch.ones(distances.shape[0],distances.shape[1],2)*0.99
    CSPparameters[:,:,1] = 1.0 - CSPparameters[:,:,0]
    gumbel_parameters = [10, 20]

    CSPDist = CSPDetectionDistribution(distances, CSPparameters, gumbel_parameters)

    samples=[]
    logProbs = torch.zeros(nsamples,dtype=torch.float32)
    for i in range(nsamples):
        samples.append(CSPDist.sample())
        l = CSPDist.log_prob(samples[i])
        l.backward()
    #
    print(calculatePositionProb(samples,logProbs))
