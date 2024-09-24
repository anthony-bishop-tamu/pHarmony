import torch
import pandas as pd
DataDir="/Volumes/homes/Anthony_Bishop/LaboratoryFiles/Data/FBDD_RM/Interleukin1beta/NMR-Spectra/BrukerDirectories/800/SAR_Screening/Feb06-2020-nmrsu"
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats
def readData(data_dir):
    #list_files = glob.glob(data_dir + "/*.list")
    list_files = ["/Volumes/homes/Anthony_Bishop/LaboratoryFiles/Data/FBDD_RM/Interleukin1beta/NMR-Spectra/BrukerDirectories/800/SAR_Screening/Feb06-2020-nmrsu/ABFBDD_IL_00001_HSQC_Feb06-2020-nmrsu_12.list",
                  "/Volumes/homes/Anthony_Bishop/LaboratoryFiles/Data/FBDD_RM/Interleukin1beta/NMR-Spectra/BrukerDirectories/800/SAR_Screening/Feb06-2020-nmrsu/ABFBDD_IL_00002_HSQC_Feb06-2020-nmrsu_22.list"]
    dfs = {f: pd.read_csv(f,sep="\s+") for f in list_files}

    return dfs
#
def calculateDistances(dfs):
    all_identical_assignments=[]
    all_non_identical_assignments=[]
    for df_name1, df1 in dfs.items():
        for df_name2, df2 in dfs.items():
            if df_name1 != df_name2:
                merged_df = pd.merge(df1, df2, suffixes=('_1', '_2'),how="cross")
                merged_df['distance'] = (((merged_df['w1_1'] - merged_df['w1_2'])*0.1014) ** 2 +
                                                (merged_df['w2_1'] - merged_df['w2_2']) ** 2)/(0.005**2)

                distances = np.array(merged_df["distance"].tolist())
                distances = distances.reshape(int(np.sqrt(distances.size)),int(np.sqrt(distances.size)))
            #
        #
    #
    return torch.from_numpy(distances)
#
def calculateLogLikelihood(matchingDistances, assignmentParameters: torch.tensor, CSPParameters):
       chi2_dist = torch.distributions.Chi2(2)
       csp_dist = torch.distributions.Normal(loc=CSPParameters[0],scale=CSPParameters[1])

       log_norm_ap = assignmentParameters - assignmentParameters.logsumexp(dim=1,keepdim=True)

       chi2_likelihood = chi2_dist.log_prob(matchingDistances)+log_norm_ap[:,0]
       csp_likelihood = csp_dist.log_prob(matchingDistances)+log_norm_ap[:,1]
       matching_likelihood = torch.logsumexp(torch.stack((chi2_likelihood,csp_likelihood),dim=1),dim=1)
       return matching_likelihood

#
def optimization_closure(optimizer,matchingDistances,assignmentParameters,CSPParameters):
    optimizer.zero_grad()
    likelihoods = calculateLogLikelihood(matchingDistances, assignmentParameters, CSPParameters)
    loss = -1*likelihoods.sum()
    loss.backward(retain_graph=True)
    print(loss)
    return loss
#
def optimizeParameters(matchingDistances,assignmentParameters,CSPParameters):
    optimizer = torch.optim.LBFGS([assignmentParameters, CSPParameters], lr=1e-1, max_iter=10000)
    optimizer.step(lambda : optimization_closure(optimizer,matchingDistances, assignmentParameters, CSPParameters))
#

def optimizeExponential(distances):
    distances = distances.numpy()
    location, scale = stats.expon.fit(distances)
    return scale
#

if __name__ == "__main__":
    parsedFiles = readData(DataDir)
    distances = calculateDistances(parsedFiles)

    distances[distances == 0] = 1E-6


    matching = distances.diagonal()
    nonMatching = distances[~torch.eye(distances.size(0), dtype=torch.bool)]
    print("Read in distances: ")

    scale = optimizeExponential(nonMatching)
    print("Exponential parameters:", scale)

    assignmentParameters = torch.zeros(matching.shape+(2,),dtype=torch.float32)
    assignmentParameters[:,0 ] = 0
    assignmentParameters[:,1 ] = -2.0
    assignmentParameters = torch.tensor(assignmentParameters - assignmentParameters.logsumexp(dim=1,keepdim=True),
                                         dtype=torch.float32,
                                         requires_grad=True)
    csp_parameters = torch.tensor(torch.tensor([25,10], dtype=torch.float32), requires_grad=True)
    nonMatching_parameters = torch.tensor([scale], dtype=torch.float32, requires_grad=True)

    optimizeParameters(matching,assignmentParameters,csp_parameters)

    print("Optimization complete")

    norm_assignmentParameters = (assignmentParameters - assignmentParameters.logsumexp(dim=1,keepdim=True)).exp()

    distributionWeights = np.array([0,float(matching.numel())/distances.numel(),float(nonMatching.numel())/distances.numel()])
    matching_weights = norm_assignmentParameters.mean(dim=0).detach().numpy()

    distributionWeights[0] = distributionWeights[1]*matching_weights[0]
    distributionWeights[1] = distributionWeights[1]*matching_weights[1]

    print("Distribution Weights: ", distributionWeights)

    distances = distances.detach().numpy()
    matching = matching.detach().numpy()
    nonMatching = nonMatching.detach().numpy()
    csp_parameters = csp_parameters.detach().numpy()
    print(csp_parameters)
    assignmentParameters = (assignmentParameters - assignmentParameters.logsumexp(dim=1,keepdim=True)).exp()
    nonMatching_parameters = nonMatching_parameters.detach().numpy()

    bin_width = 0.4
    bins = np.arange(0,  np.max(matching) + bin_width, bin_width)

    plt.hist(matching,bins=bins,label="matching",color="blue",alpha=0.5,density=True)
    #plt.hist(nonMatching,bins=np.arange(0,np.max(nonMatching), 1000),label="NonMatchingDistances",color="red",alpha=0.5,density=True)
    #plt.hist(distances.flatten(), bins=np.arange(0,np.max(nonMatching),1000), label="AllDistances", color="blue", alpha=0.5, density=True)
    x = np.linspace(0, np.max(matching), 10000)

    #mix_dist = stats.chi2.pdf(x,1)
    mix_dist = (stats.chi2.pdf(x, 2) * matching_weights[0] +
                stats.norm.pdf(x, csp_parameters[0], csp_parameters[1]) * matching_weights[1])
    nonMatch_dist = stats.expon.pdf(x, scale=nonMatching_parameters[0])
    plt.plot(x, mix_dist, 'r-', label=f'fittedDistribution')
    plt.plot(x, nonMatch_dist, 'b-', label=f'nonMatchingDistribution')
    print(bins[1]-bins[0])

    plt.xlabel('Distance^2 (error normalized 1H 0.003 square ppm)')
    plt.ylabel('Frequency')
    plt.ylim([0,0.05])
    plt.title('Histogram of Square Distances')
    plt.legend()
    plt.show()

