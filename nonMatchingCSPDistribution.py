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
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI, autoguide
from pyro.optim import Adam
import pyro.poutine as poutine
import pyro.distributions.constraints as constraints
from Frechet import Frechet
class NonCentralChi2(dist.Chi2):
    arg_constraints = {'loc': constraints.real}
    def __init__(self, dof, loc, validate_args=False):
        assert dof.shape == loc.shape
        super(NonCentralChi2, self).__init__(dof,validate_args=validate_args)
        self._loc = loc
    #
    def sample(self,sample_shape=torch.Size()):
        return self._loc + super(NonCentralChi2, self).sample(sample_shape)

    def log_prob(self,sample):
        sub = sample - self._loc
        lp = super(NonCentralChi2, self).log_prob(sub)
        lp[lp.isnan()]=-np.inf
        return lp
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NonCentralChi2, _instance)
        new._loc = self._loc
        return super().expand(batch_shape, new)


#
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
                merged_df['distance'] = (((merged_df['w1_1'] - merged_df['w1_2'])*0.1014/0.005) ** 2 +
                                         ((merged_df['w2_1'] - merged_df['w2_2'])/0.005) ** 2)

                identical = merged_df[merged_df["User_1"] == merged_df["User_2"]]['distance'].values
                all_identical_assignments += identical.tolist()
                nonidentical = merged_df[merged_df["User_1"] != merged_df["User_2"]]['distance'].values
                all_non_identical_assignments += nonidentical.tolist()
            #
        #
    #
    return np.array(all_identical_assignments), np.array(all_non_identical_assignments)
#
def model_mle(data,dof):
        # Mixing probability (between the two components)
        mix_prob = pyro.param('mix_prob', torch.full(data.shape,1.0,dtype=torch.float32), constraint=dist.constraints.unit_interval)

        alpha = pyro.param('alpha', torch.tensor([1.50]), constraint=dist.constraints.positive)
        scale = pyro.param('scale', torch.tensor([1.0]), constraint=dist.constraints.positive)
        #chi2_scale = pyro.param('chi2_scale', torch.tensor([0.00009]), constraint=dist.constraints.positive)
        chi2_scale = 1

        csp_mag_dist = Frechet(alpha, scale)
        no_csp_dist = dist.Chi2(dof)

        weighted_no_csp_log_prob = no_csp_dist.log_prob(data/chi2_scale)*mix_prob
        weighted_csp_log_prob = csp_mag_dist.log_prob(data/chi2_scale)*(1.0-mix_prob)
        total_log_prob = weighted_no_csp_log_prob + weighted_csp_log_prob

        pyro.factor("log_likelihood",total_log_prob.sum())
#
def guide(data,dof):
    pass
def optimizeMiture(data,k):

    pyro.clear_param_store()
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model_mle, guide=guide, optim=optimizer, loss=Trace_ELBO())

    # Training loop
    num_iterations = 100000
    loss_previous = torch.finfo(torch.float32).max
    for step in range(num_iterations):
        loss = svi.step(data,dof)
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss}")
        if abs(loss - loss_previous) < 1e-5:
            break
        loss_previous = loss

    # After training, we can inspect the learned parameters (MLE estimates)
    alpha_mle = pyro.param("alpha").item()
    scale_mle = pyro.param("scale").item()
    mix_prob= pyro.param("mix_prob").detach().numpy()

    print("MLE estimates:")
    print(f"alpha (Frechet shape): {alpha_mle}")
    print(f"beta (Frechet scale): {scale_mle}")
    print(f"mix_prob: {mix_prob[:10]}")  # Show first 10 logits for the mixture components
    return alpha_mle, scale_mle, mix_prob
#

def optimizeWeibull(distances):
    p1 = stats.weibull_min.fit(distances)
    return p1
#
def createMatchingFocusedPlot(ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes,matching, nonMatching,dof):

    orgSize = nonMatching.size

    low_nonMatching = nonMatching[nonMatching < matching.max()]
    alpha_mle, scale_mle, mix_prob= optimizeMiture(torch.from_numpy(matching),dof)
    bin_width = 1
    complement = 1.0 - mix_prob
    chi2_sum = mix_prob.sum().item()
    complement_sum = complement.sum().item()
    total = chi2_sum + complement_sum
    weights = np.array([chi2_sum/total,complement_sum/total])
    bins = np.arange(0, np.max(matching.max()) + bin_width, bin_width)
    log_bins = np.logspace(-0.5, np.log10(matching.max()), 200)

    matching_weights = np.ones_like(matching) / (matching.size)
    low_nonMatching_weights = np.ones_like(low_nonMatching) / (orgSize)
    ax2.hist(matching, bins=bins, weights=matching_weights, label="matching", color="blue", alpha=0.5)
    ax2.hist(nonMatching, bins=bins, weights=nonMatching/nonMatching.size, label="nonmatching", color="red", alpha=0.5)

    ax1.hist(matching, bins=bins, weights=matching_weights, label="matching", color="blue", alpha=0.5)
    ax1.hist(low_nonMatching, bins=bins, weights=low_nonMatching_weights, label="nonmatching", color="red", alpha=0.5)

    chi2_pdf = stats.chi2(dof).pdf(bins)* weights[0]
    csp_pdf = np.zeros_like(bins)
    csp_pdf[1:] = Frechet(torch.tensor([alpha_mle]),torch.tensor([scale_mle])).log_prob(torch.from_numpy(bins[1:])).exp().detach().numpy()*weights[1]
    #mix_pdf = stats.halfnorm(1).pdf(x)
    #nonmatch = stats.weibull_min(p1[0],loc=p1[1],scale=p1[2]).pdf(x)


    ax1.plot(bins, csp_pdf, 'b-', label=f'csp_only')
    ax1.plot(bins,chi2_pdf, label=f'chi2_only')
    ax1.plot(bins, chi2_pdf+csp_pdf, 'r-', label=f'fittedDistribution')

    chi2_pdf = stats.chi2(dof).pdf(log_bins) * weights[0]
    csp_pdf = Frechet(torch.tensor([alpha_mle]), torch.tensor([scale_mle])).log_prob(
        torch.from_numpy(log_bins)).exp().detach().numpy() * weights[1]

    ax2.plot(log_bins, chi2_pdf + csp_pdf, 'r-', label=f'mix_distribution')
    #ax2.plot(x, nonmatch, 'b-', label=f'nonMatchingDistribution')

    print(bins[1] - bins[0])

    ax2.set_xlabel('Distance (error normalized 1H 0.003 square ppm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Square Distances')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.legend()

    ax1.set_ylabel('Frequency')
    ax1.set_xlim([0,100])
    ax1.legend()

def createNonMatchingFocusedPlot(ax, nonMatching):
    p1 = optimizeWeibull(nonMatching)
    bins = np.logspace(-0.5,np.log10(nonMatching.max()),100)
    ax.hist(nonMatching,bins=bins, label="nonmatching", color="blue", alpha=0.5,density=True)
    pdf = stats.weibull_min(p1[0],loc=p1[1],scale=p1[2]).pdf(bins)
    ax.plot(bins, pdf, 'r-', label=f'fittedDistribution')
    ax.set_xscale('log')
    return p1

if __name__ == "__main__":
    reference_dfs, fragment_dfs = readData(DataDir+'/referenceSpectraLists.txt',DataDir+'/fragmentSpectraLists.txt')
    matching, nonMatching = calculateDistances(reference_dfs, fragment_dfs)
    matching[matching == 0] = 0.001
    dof=2
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1)
    fig.set_size_inches(8,12)
    p1 = createNonMatchingFocusedPlot(ax1, nonMatching)
    createMatchingFocusedPlot(ax2, ax3, matching, nonMatching,dof)
    fig.tight_layout()
    fig.show()
    fig.savefig('NonMatchingFocusedPlot.svg')

