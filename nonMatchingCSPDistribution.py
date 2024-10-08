import matplotlib.axes
import torch
import pandas as pd
DataDir="/Users/anthonybishop/LaboratoryFiles/Programs/PeakMatcher/TestData/Manually_Picked_Peaks"
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats
import scipy
from Frechet import Frechet
def readData(referenceListFile: str, fragmentListFile: str):
    #list_files = glob.glob(data_dir + "/*.list")

    referenceFiles = open(referenceListFile, 'r').readlines()
    fragmentFiles = open(fragmentListFile, 'r').readlines()

    reference_dfs = {f.strip(): pd.read_csv(f.strip(),sep="\s+") for f in referenceFiles}
    fragment_dfs = {f.strip(): pd.read_csv(f.strip(),sep="\s+") for f in fragmentFiles}

    return reference_dfs, fragment_dfs
#
def calculateDistances(reference_dfs, fragment_dfs):
    all_identical_assignments=[]
    all_non_identical_assignments=[]
    for df_name1, df1 in reference_dfs.items():
        for df_name2, df2 in fragment_dfs.items():
                merged_df = pd.merge(df1, df2, suffixes=('_1', '_2'),how="cross")
                merged_df['distance'] = (((merged_df['w1_1'] - merged_df['w1_2'])*0.1014) ** 2 +
                                                (merged_df['w2_1'] - merged_df['w2_2']) ** 2)/(0.002**2)

                identical = merged_df[merged_df["User_1"] == merged_df["User_2"]]['distance'].values
                all_identical_assignments += identical.tolist()
                nonidentical = merged_df[merged_df["User_1"] != merged_df["User_2"]]['distance'].values
                all_non_identical_assignments += nonidentical.tolist()
            #
        #
    #
    return np.array(all_identical_assignments), np.array(all_non_identical_assignments)
#
def calculateLogLikelihood(matchingDistances, assignmentParameters: torch.tensor, CSPParameters,dof):
       non_csp_dist = torch.distributions.Chi2(dof)
       csp_dist = torch.distributions.Gamma(concentration=2,rate=1.0/CSPParameters[0])

       log_norm_ap = assignmentParameters - assignmentParameters.logsumexp(dim=1,keepdim=True)
       matchingDistances[matchingDistances == 0] = np.finfo(float).eps
       non_csp_dist = non_csp_dist.log_prob(matchingDistances)+log_norm_ap[:,0]
       csp_likelihood = csp_dist.log_prob(matchingDistances)+log_norm_ap[:,1]
       matching_likelihood = torch.logsumexp(torch.stack([non_csp_dist,csp_likelihood], dim=1), dim=1)
       return matching_likelihood

#
def optimizeCSPParameters(matchingDistances,dof):
    CSPParameters = scipy.stats.gamma.fit(matchingDistances[matchingDistances > 3 ],floc=0,fa=2.0)
    CSPParameters = torch.tensor([CSPParameters[2]],dtype=torch.float32, requires_grad=True)
    assignmentParameters = torch.zeros(matchingDistances.shape+(2,),dtype=torch.float32)
    assignmentParameters[:,0][matchingDistances<2] = 2
    assignmentParameters[:,1][matchingDistances>5] = 2
    assignmentParameters = torch.tensor(assignmentParameters,dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([assignmentParameters,CSPParameters],lr=0.01)
    matchingDistances = torch.from_numpy(matchingDistances)
    loss_prev = torch.finfo(torch.float32).max
    for i in range(100000):
        optimizer.zero_grad()
        likelihoods = calculateLogLikelihood(matchingDistances, assignmentParameters, CSPParameters,dof)
        loss = -1 * likelihoods.sum()
        loss.backward()
        print(loss)
        if abs(loss - loss_prev)< 1e-5:
            break
        else:
            loss_prev = loss
        optimizer.step()
    return assignmentParameters.detach().numpy(), CSPParameters.detach().numpy()
#

def optimizeWeibull(distances):
    p1 = stats.weibull_min.fit(distances)
    return p1
#
def createMatchingFocusedPlot(ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes,matching, nonMatching,dof):

    orgSize = nonMatching.size

    matching =  matching[matching < 400] #filter extreme values
    nonMatching = nonMatching[nonMatching < 400]
    assignmentParameters, CSPParameters = optimizeCSPParameters(matching,dof)
    bin_width = 1
    weights = np.sum(
       np.exp(assignmentParameters - scipy.special.logsumexp(assignmentParameters, axis=1, keepdims=True)), axis=0)
    weights = weights / np.sum(weights)
    bins = np.arange(0, np.max(nonMatching) + bin_width, bin_width)

    matching_weights = np.ones_like(matching) / (matching.size)
    nonMatching_weights = np.ones_like(nonMatching) / (orgSize)
    ax2.hist(matching, bins=bins, weights=matching_weights, label="matching", color="blue", alpha=0.5)
    ax2.hist(nonMatching, bins=bins, weights=nonMatching_weights, label="nonmatching", color="red", alpha=0.5)
    ax1.hist(matching, bins=bins, weights=matching_weights, label="matching", color="blue", alpha=0.5)
    ax1.hist(nonMatching, bins=bins, weights=nonMatching_weights, label="nonmatching", color="red", alpha=0.5)
    x = np.arange(0, np.max(matching) + bin_width, bin_width)

    mix_pdf = stats.chi2(dof).pdf(x) * weights[0] + stats.gamma(dof,CSPParameters[0]).pdf(x)*weights[1]
    #mix_pdf = stats.halfnorm(1).pdf(x)
    #nonmatch = stats.weibull_min(p1[0],loc=p1[1],scale=p1[2]).pdf(x)

    ax2.plot(x, mix_pdf, 'r-', label=f'fittedDistribution')
    ax1.plot(x, mix_pdf, 'r-', label=f'fittedDistribution')
    #ax2.plot(x, nonmatch, 'b-', label=f'nonMatchingDistribution')

    print(bins[1] - bins[0])

    ax2.set_xlabel('Distance (error normalized 1H 0.003 square ppm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Square Distances')
    ax2.set_yscale('log')
    ax2.legend()

    ax1.set_ylabel('Frequency')
    ax1.legend()

def createNonMatchingFocusedPlot(ax, nonMatching):
    p1 = optimizeWeibull(nonMatching)
    bins = np.logspace(-0.5,7,100)
    ax.hist(nonMatching,bins=bins, label="nonmatching", color="blue", alpha=0.5,density=True)
    pdf = stats.weibull_min(p1[0],loc=p1[1],scale=p1[2]).pdf(bins)
    ax.plot(bins, pdf, 'r-', label=f'fittedDistribution')
    ax.set_xscale('log')
    return p1

if __name__ == "__main__":
    reference_dfs, fragment_dfs = readData(DataDir+'/referenceSpectraLists.txt',DataDir+'/fragmentSpectraLists.txt')
    matching, nonMatching = calculateDistances(reference_dfs, fragment_dfs)
    dof=2
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1)
    fig.set_size_inches(8,12)
   # p1 = createNonMatchingFocusedPlot(ax1, nonMatching)
    createMatchingFocusedPlot(ax2, ax3, matching, nonMatching,dof)
    fig.tight_layout()
    fig.show()
    fig.savefig('NonMatchingFocusedPlot.svg')

