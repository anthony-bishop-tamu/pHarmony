import torch
import pyro.distributions as dist
import torch.nn.functional as F
class CSPDetectionDistribution(torch.distributions.Distribution):
    def __init__(self, distances: torch.tensor, csp_prob_parameters: torch.tensor, csp_distribution_parameters: torch.tensor, non_matching_distribution_parameters: torch.tensor):
        super().__init__()

        assert(distances.shape + (2,) == csp_prob_parameters.shape)
        assert(csp_distribution_parameters.shape == (2,)) #Gaussian parameters mean, std
        assert(non_matching_distribution_parameters.shape == (2,)) #Weibull distribution parameters scale, concentration

        self._distances = distances
        self._csp_probability_parameters = csp_prob_parameters # logits
        self._csp_distribution_parameters = csp_distribution_parameters
        self._csp_distribution = dist.Normal(self._csp_distribution_parameters[0],self._csp_distribution_parameters[1])
        self._non_matching_distribution_parameters = non_matching_distribution_parameters
        self._non_matching_distribution = dist.Weibull(concentration=self._non_matching_distribution_parameters[0],
                                                       scale=self._non_matching_distribution_parameters[1]) #location scale
        self._chi2_distribution = dist.torch.Chi2(1) #chi2 distribution for

        self.eps_float32 = torch.finfo(torch.float32).eps
        self.max_float32 = torch.finfo(torch.float32).max
        self.min_float32 = torch.finfo(torch.float32).min
        self._event_shape = torch.Size([self._distances.shape[0],3])

        self._loglikelihoodMatrix = torch.stack((self._chi2_distribution.log_prob(self._distances).clamp(min=self.min_float32),
                                                self._csp_distribution.log_prob(self._distances).clamp(min=self.min_float32),
                                                self._non_matching_distribution.log_prob(self._distances).clamp(min=self.min_float32)),dim=2)

        self._calculateBaseMatchingLikelihood()

        assert not self._loglikelihoodMatrix.isnan().any()
    #
    def _calculateBaseMatchingLikelihood(self):
       # nonCSPlikelihoods = self._loglikelihoodMatrix[:,:, 0]
       # CSPLikelihoods = self._loglikelihoodMatrix[:,:, 1]
        cspProbLogits = self._csp_probability_parameters
        #log_weights = torch.zeros(10,10,2)
        log_weights = cspProbLogits - cspProbLogits.logsumexp(dim=2, keepdim=True)
        self._baseMatchLikelihood = torch.logsumexp(self._loglikelihoodMatrix[:,:,:2] + log_weights, dim=2)

    def _calculateDecisionLikelihoods(self, rowIndicies, columnIndicies):
        nonMatchingCSPLikelihoods = self._loglikelihoodMatrix[rowIndicies,columnIndicies,2]
        assert(len(nonMatchingCSPLikelihoods.shape) == 1)
        nonMatchingLogSum = nonMatchingCSPLikelihoods.sum()
        withNoMatchLikeliHoodsAtOtherSites = self._baseMatchLikelihood[rowIndicies,columnIndicies] + (nonMatchingLogSum - nonMatchingCSPLikelihoods)
        withNoMatchDecision = torch.cat((withNoMatchLikeliHoodsAtOtherSites,nonMatchingLogSum.unsqueeze(0)),dim=0)
        return withNoMatchDecision
    #
    def _makeDecision(self, logits):
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return idx, torch.exp(logits[idx] - logits.logsumexp(dim=0))

    def _getNextInSequence(self, availableRows, availableColumns):
        availableRowIndicies = torch.where(availableRows == True)[0]
        availableColumnIndicies = torch.where(availableColumns == True)[0]

        total_available = len(availableRowIndicies) + len(availableColumnIndicies)

        index = torch.randint(0, total_available, (1,)).item()

        if index < len(availableRowIndicies):
            random_row_index = availableRowIndicies[index]

            decisionLikelihoods = self._calculateDecisionLikelihoods(random_row_index,availableColumnIndicies)
            decision_idx, decision_probability = self._makeDecision(decisionLikelihoods)
            availableRows[random_row_index] = False

            assert (decision_idx <= self._distances.shape[1])
            if decision_idx == availableColumnIndicies.shape[0]:
                match_index = -1
            else:
                match_index = availableColumnIndicies[decision_idx]
                availableColumns[match_index] = False

            decision_probability = decision_probability * 1.0 / total_available

            if match_index == -1:
                linear_col_idx = -1
            else:
                linear_col_idx = match_index + self._distances.shape[0]

            particle = torch.tensor([random_row_index, linear_col_idx ],dtype=torch.int32)
            return particle, decision_probability
        else:
            index -= len(availableRowIndicies)
            random_column_index = availableColumnIndicies[index]
            decisionLikelihoods = self._calculateDecisionLikelihoods(availableRowIndicies, random_column_index)

            decision_idx, decision_probability = self._makeDecision(decisionLikelihoods)

            availableColumns[random_column_index] = False
            assert(decision_idx <= self._distances.shape[0])
            if decision_idx == availableRowIndicies.shape[0]:
                match_index = -1
            else:
                match_index = availableRowIndicies[decision_idx]
                availableRows[match_index] = False

            decision_probability = decision_probability * 1.0 / total_available

            particle = torch.tensor([random_column_index + self._distances.shape[0], match_index], dtype=torch.int32)
            return particle, decision_probability
    def sample(self, sample_shape=torch.Size()):
        availableRows = torch.ones(self._distances.shape[0], dtype=torch.bool)
        availableColumns = torch.ones(self._distances.shape[1], dtype=torch.bool)
        sample = torch.zeros((self._distances.shape[0]+self._distances.shape[1],2), dtype=torch.int32)
        i = 0
        log_likelihood = 0
        while (len(torch.where(availableRows == True)[0]) > 0 or len(torch.where(availableColumns == True)[0]) > 0):
            assert (availableRows.shape == availableColumns.shape)
            sample[i,:], logProb = self._getNextInSequence(availableRows, availableColumns)
            log_likelihood += logProb
            # print(log_likelihood)
            i += 1
        #

        return sample[:i,:]
    #
    def log_prob(self, sample):
        log_likelihood = torch.zeros(1, 1)
        order = sample[:, 0]
        match = sample[:, 1]
        availableRows = torch.ones(self._distances.shape[0], dtype=torch.bool)
        availableColumns = torch.ones(self._distances.shape[1], dtype=torch.bool)

        for i in range(order.shape[0]):
            index = order[i].item()
            availableRowIndicies = torch.where(availableRows == True)[0]
            availableColumnIndicies = torch.where(availableColumns == True)[0]

            total_available = len(availableRowIndicies) + len(availableColumnIndicies)
            if index < self._distances.shape[0]:
                assert (availableRows[index])
                decision_likelihoods = self._calculateDecisionLikelihoods(index,availableColumnIndicies)

                availableRows[index] = False
                if match[i] != -1:
                    ind = torch.where(availableColumnIndicies == match[i] - self._distances.shape[0])[0]
                    availableColumns[match[i] - self._distances.shape[0]] = False
                else:
                    ind = -1
                l = decision_likelihoods[ind] - decision_likelihoods.logsumexp(dim=0) + torch.log(torch.tensor([1.0/total_available],dtype=torch.float32))
                log_likelihood = log_likelihood + l
            else:
                index -= self._distances.shape[0]
                assert (availableColumns[index])
                decision_likelihoods = self._calculateDecisionLikelihoods(availableRowIndicies,index)
                availableColumns[index] = False
                if match[i] != -1:
                    ind = torch.where(availableRowIndicies == match[i])[0]
                    availableRows[match[i]] = False
                else:
                    ind = -1

                l = decision_likelihoods[ind] - decision_likelihoods.logsumexp(dim=0) + torch.log(torch.tensor([1.0/total_available],dtype=torch.float32))
                log_likelihood = log_likelihood + l

        assert(log_likelihood.requires_grad)
        assert(torch.all(~availableRows))
        assert(torch.all(~availableColumns))
        return log_likelihood

    def csp_distribution_parameters(self):
        return self._csp_distribution_parameters

    def csp_probability_parameters(self):
        return self._csp_probability_parameters



#
