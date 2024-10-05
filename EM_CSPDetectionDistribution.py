from CSPDetectionDistribution import CSPDetectionDistribution
from testCSPDetectionDistribution import calculateDistanceMatrix, calculatePositionProb
import torch
from torchviz import make_dot
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import sys
def getInitialWeibullParameters(distances):
    distances = distances.flatten().numpy()
    shape,loc,scale = stats.weibull_min.fit(distances,loc=0)
    return shape,scale
#
def EM_minimization_function(samples, dist: CSPDetectionDistribution, csp_probability_prior_params: torch.tensor,
                             csp_distribution_prior_params: torch):

    logLikelihoodTerm = dist.log_prob(samples[0],samples[1]).sum()

    log_probability_params = dist.csp_probability_parameters() - dist.csp_probability_parameters().logsumexp(dim=2,keepdim=True)
    csp_assignment_individual_regularization = log_probability_params * (csp_probability_prior_params.unsqueeze(0).unsqueeze(0)-1.0)
    csp_assignment_regularization = csp_assignment_individual_regularization.sum()

    csp_distribution_individual_regularization = (csp_distribution_parameters - csp_distribution_prior_params[:,0])**2/(2*csp_distribution_prior_params[:,1]**2)
    csp_distribution_regularization = -1*csp_distribution_individual_regularization.sum()
    value = logLikelihoodTerm #csp_assignment_regularization + csp_distribution_regularization
    print("Loss: ", logLikelihoodTerm, csp_assignment_regularization, csp_distribution_regularization)
    #print(csp_distribution_params)
    #print(dist.csp_assignment_parameters())
    return -1 * value

def optimizer_closure(optimizer,samples, distances,csp_assignment_params, csp_distribution_params,
                      csp_assignment_prior_params: torch.tensor,csp_distribution_prior_params: torch.tensor,
                      no_match_distribution_parameters):
    optimizer.zero_grad()
    dist = CSPDetectionDistribution(distances,csp_assignment_params,csp_distribution_params,no_match_distribution_parameters)
    loss = EM_minimization_function(samples,dist, csp_assignment_prior_params, csp_distribution_prior_params)
    loss.backward(retain_graph=True)
    return loss
#
def maximization(samples: tuple,
                 distances: torch.tensor,
                 csp_assignment_params: torch.tensor,
                 csp_distribution_params: torch.tensor,
                 csp_assignment_prior_params: torch.tensor,
                 csp_distribution_prior_params: torch.tensor,
                 no_match_distribution_parameters: torch.tensor):


    optimizer = torch.optim.LBFGS([csp_assignment_params,csp_distribution_params],lr=1e-3,max_iter=100)

    optimizer.step(lambda : optimizer_closure(optimizer, samples, distances, csp_assignment_params,csp_distribution_params,
                                              csp_assignment_prior_params, csp_distribution_prior_params,no_match_distribution_parameters))
#
def runEMStep(distances: torch.tensor,
              csp_probability_params: torch.tensor,
              csp_distribution_params: torch.tensor,
              csp_probability_prior_params: torch.tensor,
              csp_distribution_prior_params: torch.tensor,
              no_match_distribution_params: torch.tensor,
              sampleSize: int):

        dist = CSPDetectionDistribution(distances,csp_probability_params,
                                        csp_distribution_params,no_match_distribution_params)
        samples = dist.sample((sampleSize,))
        maximization(samples,distances,csp_probability_params,csp_distribution_params,
                     csp_probability_prior_params,csp_distribution_prior_params,no_match_distribution_params)
        return samples
#
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(42)
    dim = 200
    noCSP = 180
    distances = calculateDistanceMatrix(dim,noCSP)

    csp_distribution_prior_gaussian_parameters = torch.tensor([[20, 5],[8,5]], dtype=torch.float32) #weak gaussian prior on the location, and scale params

    csp_distribution_initial_params = torch.tensor([20,2], dtype=torch.float32,requires_grad=True)

    csp_probability_prior_dirichlet_parameters = torch.tensor([0.05,0.01])
    csp_probability_initial_params = torch.ones(distances.shape[0],distances.shape[1],2)
    csp_probability_initial_params[:,:,:] = torch.tensor([0,-2.197]).unsqueeze(0).unsqueeze(0)
    csp_probability_initial_params = csp_probability_initial_params.clone().detach().requires_grad_(True)


    shape,scale = getInitialWeibullParameters(distances)
    non_matching_distribution_initial_parameters = torch.tensor([shape, scale],dtype=torch.float32)

    csp_probability_parameters = csp_probability_initial_params.clone().detach().requires_grad_(True)
    csp_distribution_parameters = csp_distribution_initial_params.clone().detach().requires_grad_(True)
    no_match_distribution_parameters = non_matching_distribution_initial_parameters.clone().detach().requires_grad_(True)

    print("Distance Matrix: ")
    print(distances)
    for i in range(100):

        samples,logProbs = runEMStep(distances,csp_probability_parameters,csp_distribution_parameters,
                  csp_probability_prior_dirichlet_parameters,csp_distribution_prior_gaussian_parameters,
                  no_match_distribution_parameters,1000)

        print(f"iteration {i} \n starting matching probabilities: ")
        matchingProb = calculatePositionProb(samples,(logProbs - logProbs.logsumexp(dim=-1,keepdim=True)).exp(), distances.shape)
        print(matchingProb)

        print("cspAssignmentProbabilities: ")
        print( (csp_probability_parameters[:,:,1]-csp_probability_parameters.logsumexp(dim=2)).exp())

        print("CSP distribution parameters: ")
        print(csp_distribution_parameters)
#













