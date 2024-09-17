import torch
import math
class PeakMatchingDistribution(torch.distributions.Distribution):
    def __init__(self, distances): #distances and priors are mxn matricies
        super(PeakMatchingDistribution, self).__init__()

        self._distances = distances
    def sample(self, sample_shape=torch.Size()):
        availableRows = torch.ones(self._distances.shape[0], dtype=torch.bool)
        availableColumns = torch.ones(self._distances.shape[1], dtype=torch.bool)
        orderArray = []
        matchArray = []
        i = 0
        log_likelihood = 0
        sample_matrix = torch.zeros(self._distances.shape)
        while(len(torch.where(availableRows == True)[0]) > 0 or len(torch.where(availableColumns == True)[0]) > 0):
            assert(availableRows.shape == availableColumns.shape)
            order, match, likelihood = self._getNextInSequence(availableRows, availableColumns, sample_matrix)
            orderArray.append(order)
            matchArray.append(match)
            log_likelihood += math.log(likelihood)
            #print(log_likelihood)
            i+=1
        #
        return orderArray, matchArray, log_likelihood
    #
    def calculateLogLikelihood(self, sample):
        log_likelihood = torch.zeros(1,1)
        orderArray, matchArray, x = sample
        availableRows = torch.ones(self._distances.shape[0], dtype=torch.bool)
        availableColumns = torch.ones(self._distances.shape[1], dtype=torch.bool)
        eps_float32 = torch.finfo(torch.float32).eps
        max_float32 = torch.finfo(torch.float32).max
        negEnergy = torch.zeros(1,1)
        for i in range(len(orderArray)):
            index = orderArray[i]
            availableRowIndicies = torch.where(availableRows == True)[0]
            availableColumnIndicies = torch.where(availableColumns == True)[0]
            total_available = len(availableRowIndicies)+len(availableColumnIndicies)
            if index < self._distances.shape[0]:
                assert(availableRows[index])
                likelihood = -1*self._distances[index,availableColumnIndicies]*self._distances[index,availableColumnIndicies]
                numerator = likelihood.flatten().exp().clamp(min=eps_float32,max=max_float32)
                probability = numerator/numerator.sum()
                ind = torch.where(availableColumnIndicies == matchArray[i]-self._distances.shape[0])[0]
                l = probability[ind]/total_available
                log_likelihood += l.log()
                negEnergy += l.log()
               # print(log_likelihood)
                availableRows[index] = False
                availableColumns[matchArray[i]-self._distances.shape[0]] = False
            else:
                index -= self._distances.shape[0]
                assert(availableColumns[index])
                likelihood = -1*self._distances[availableRowIndicies,index]
                numerator = likelihood.flatten().exp().clamp(min=eps_float32, max=max_float32)
                probability = numerator / numerator.sum()
                ind = torch.where(availableRowIndicies == matchArray[i])[0]
               # print(log_likelihood)
                l = probability[ind]/total_available
                log_likelihood += l.log()
                negEnergy += l.log()
                availableColumns[index] = False
                availableRows[matchArray[i]] = False


        return log_likelihood

    def _getNextInSequence(self,availableRows, availableColumns,sample_matrix):
        availableRowIndicies = torch.where(availableRows == True)[0]
        availableColumnIndicies = torch.where(availableColumns == True)[0]

        total_available = len(availableRowIndicies)+len(availableColumnIndicies)
        eps_float32 = torch.finfo(torch.float32).eps
        max_float32 = torch.finfo(torch.float32).max
        index = torch.randint(0,total_available,(1,)).item()

        if index < len(availableRowIndicies):
            random_row_index = availableRowIndicies[index].item()
            likelihood = -1*self._distances[random_row_index,availableColumnIndicies]
            likelihoodExp = likelihood.flatten().exp().clamp(min=eps_float32,max=max_float32)
            probabilities = likelihoodExp/likelihoodExp.sum()
            i, prob = self._getMatch(probabilities)
            match_index = availableColumnIndicies[i].item()
            prob = prob*1.0/total_available
            availableRows[random_row_index] = False
            availableColumns[match_index] = False
            return random_row_index,match_index+self._distances.shape[0], prob
        else:
            index -= len(availableRowIndicies)
            random_column_index = availableColumnIndicies[index].item()
            likelihood = -1*self._distances[availableRowIndicies,random_column_index]
            likelihoodExp = likelihood.flatten().exp().clamp(min=eps_float32,max=max_float32)
            probabilities = likelihoodExp / likelihoodExp.sum()
            i, prob = self._getMatch(probabilities)
            match_index = availableRowIndicies[i].item()
            prob = prob * 1.0 / total_available
            availableRows[match_index] = False
            availableColumns[random_column_index] = False
            return random_column_index+self._distances.shape[0], match_index, prob
    #
    def _getMatch(self, probabilities):
        dist = torch.distributions.Categorical(probs=probabilities)
        idx = dist.sample().item()
        return idx, probabilities[idx].item()





