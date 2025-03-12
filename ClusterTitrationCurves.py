from scipy import optimize as opt
import scipy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import argparse
from pathlib import Path
from Bio import PDB
import re
from matplotlib import pyplot as plt
def CSPBindingEquation(Kd: np.array, CSPsat: np.array, P: float, X: np.array):
    CSP_calc = CSPsat[:,np.newaxis] * ((X[np.newaxis,:] + P + Kd[:,np.newaxis]) - np.sqrt((X[np.newaxis,:] + P + Kd[:,np.newaxis])**2 - 4 * X[np.newaxis,:] * P)) / (2 * P)
    assert (CSP_calc != np.NaN).all()
    return CSP_calc
#
def minimization(params: np.array, X: np.array, CSP: np.array, P: float,error: float):
    Kd = params[:len(params)//2]
    CSPsat = params[len(params)//2:]
    residuals = (CSP - CSPBindingEquation(Kd, CSPsat, P, X))**2/error**2
    return np.nansum(residuals)
#
def minimization_labels(params: np.array, labels: np.array, X: np.array, CSP: np.array, P: float,error: float):
    Kd = np.zeros(labels.shape)
    n_clusters = len(np.unique(labels))

    baseKd = params[:n_clusters]
    CSPsat = params[n_clusters:]

    for i in range(np.size(baseKd)):
        mask = labels == i
        Kd[mask] = baseKd[i]
    #
    all_params = np.concatenate((Kd,CSPsat),axis=0)
    return minimization(all_params, X, CSP, P, error)

#
def extractResIndex(s):
    match = re.fullmatch(r'[A-Za-z]+(\d+)N-H', s)
    return int(match.group(1)) if match else None

def generateScaledFits(CSPs: np.array, concentrations: np.array, protein_concentration: float, residueIndexes: np.array,  labels: np.array, mean_params: np.array, std_params: np.array):
    n_clusters = len(np.unique(labels))
    figsize = [ 6.4, 4.8]
    figsize[1] = figsize[1]/2 * n_clusters
    fig, axs = plt.subplots(nrows=n_clusters, ncols=2, figsize=figsize)
    for i in range(0,n_clusters):
        generateScaledFitFigure(CSPs,concentrations,protein_concentration,residueIndexes,labels,mean_params,std_params,i,axs[i,0],axs[i,1])
    #
    fig.tight_layout()
    fig.show()
    return fig
#
def generateScaledFitFigure(CSPs: np.array, concentrations: np.array, protein_concentration: float, residueIndexes: np.array, labels: np.array,
                            mean_params: float, std_params: float,  clusterIndex: int, ax_actual: plt.Axes, ax_scaled: plt.Axes):
    n_clusters = len(np.unique(labels))
    cluster_mask = labels == clusterIndex
    cluster_Kd = mean_params[clusterIndex:clusterIndex+1]
    cluster_CSPsats = mean_params[n_clusters:][cluster_mask]
    cluster_titrations = CSPs[cluster_mask,:]
    cluster_residue_indexes = residueIndexes[cluster_mask]

    range_c = np.linspace(0, np.max(concentrations), 100)
    CSP_calc = CSPBindingEquation(cluster_Kd,cluster_CSPsats,protein_concentration,range_c.transpose())
    error = np.ones_like(cluster_titrations)*0.003

    ax_actual.plot(range_c,CSP_calc.transpose(),linestyle='-',marker='')

    colors = [ line.get_color() for line in ax_actual.get_lines() ]
    ax_actual.set_xlabel('Ligand Concentration (mM)')
    ax_actual.set_ylabel("CSP (ppm)")
    ax_actual.set_ylim([0,cluster_titrations.max()*1.05])
    ax_actual.set_xlim([0,ax_actual.get_xlim()[1]])
    kd_textbox_coord = [ ax_actual.get_xlim()[0]*1.15, ax_actual.get_ylim()[1]*0.85]
    ax_actual.text(kd_textbox_coord[0],kd_textbox_coord[1],f"Kd: {cluster_Kd[0]:.0f} ± {std_params[clusterIndex]:0.3f} mM ", fontsize=12, color='black')
    for i in range(0,cluster_titrations.shape[0]):
        ax_actual.errorbar(concentrations,cluster_titrations[i,:],color=colors[i],yerr=error[i,:],ecolor=colors[i],fmt='o')


    # generate scaled plots
    n_series = len(CSP_calc)
    interval = 1.0/(n_series+1)
    finalPoints = np.array([i*interval for i in range(1,n_series+1)])
    scaling_factors = (finalPoints / np.max(cluster_titrations,axis=1))[:,np.newaxis]

    scaled_cluster_titrations = cluster_titrations * scaling_factors
    scaled_CSP_calc = CSP_calc * scaling_factors
    scaled_error = error * scaling_factors

    ax_scaled.plot(range_c,scaled_CSP_calc.transpose(),linestyle='-',marker='')
    ax_scaled.set_xlabel('Ligand Concentration (mM)')
    ax_scaled.set_ylabel("Scaled CSP")
    ax_scaled.set_ylim([0,ax_scaled.get_ylim()[1]])
    for i in range(0,cluster_titrations.shape[0]):
        ax_scaled.errorbar(concentrations,scaled_cluster_titrations[i,:],color=colors[i],yerr=scaled_error[i,:],ecolor=colors[i],fmt='o')
        ax_scaled.set_ylim([0.0,1.0])
        residue_textbox_x = ax_scaled.get_xlim()[1]*1.05
        ax_scaled.text(residue_textbox_x, scaled_cluster_titrations[i,-1],f"{cluster_residue_indexes[i]}N-H",va='center',fontsize=12,color=colors[i])






def calculateBIC(coordinates: np.array, centroids: np.array, labels: np.array):
    n_clusters = len(np.unique(labels))
    n_coords = len(coordinates)
    centroids_expanded = centroids[labels,:]
    sumResiduals_2 = np.sum((coordinates - centroids_expanded)**2)
    sigma_2 = 1.0/(n_coords - n_clusters)*sumResiduals_2
    log_likelihood = -1*n_coords/2.0*np.log(2*np.pi*sigma_2) - 1.0/(2*sigma_2) * sumResiduals_2

    BIC = -2.0*log_likelihood + (n_clusters*3 + 1)*np.log(n_clusters)
    return BIC

#
def MonteCarloKds(CSPs: np.array, concentrations: np.array, protein_concentration: float, labels: np.array, params: np.array, error: float):
    bounds = opt.Bounds(lb=np.zeros_like(params)+0.0001)
    error_adjusted_CSPS= scipy.stats.norm.rvs(loc=CSPs,scale=np.array([error]),size=(100,*CSPs.shape))
    #error_adjusted_CSPS = np.stack([CSPs]*100,axis=0)
    n_clusters = len(np.unique(labels))
    monte_params = np.zeros((len(error_adjusted_CSPS),n_clusters+len(CSPs)))
    for i in range(0,len(error_adjusted_CSPS)):
        error_mod_data = error_adjusted_CSPS[i,:,:]
        result = opt.minimize(minimization_labels,params, args=(labels,concentrations,error_mod_data,protein_concentration,0.003), bounds=bounds)
        monte_params[i,:] = result.x
    #
    return np.mean(monte_params,axis=0), np.std(monte_params,axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--titration_data',required=True,type=Path,help='tiration data')
    parser.add_argument('--pdb_file',required=True,type=Path,help='pdb file')
    parser.add_argument('--chain',required=True,type=str,help='Chain code')
    parser.add_argument('--offset_index',default=0,type=int,help='Value to add to the assigned residue index to match the pdb indexing')
    parser.add_argument("--protein_concentration", required=True, type=float,help="Protein concentration")

    parser = parser.parse_args()
    protein_concentration = parser.protein_concentration
    titration_data = pd.read_csv(parser.titration_data)
    structure = PDB.PDBParser().get_structure("protein", parser.pdb_file)
    if len(structure) != 1:
        raise Exception("There should be exactly one model in the pdb")
    chain = structure[0][parser.chain]
    residueIndexes = titration_data['Assignment'].apply(extractResIndex).to_numpy(dtype=np.int32)
    residueIndexes += parser.offset_index

    coords = [ chain[res_id]['N'].get_coord() for res_id in residueIndexes.tolist() ]
    coords = np.array(coords)
    concentrations = titration_data.columns[1:].to_numpy(dtype=float)
    CSPs = titration_data.iloc[:,1:].to_numpy(dtype=float)

    selected_rows = []
    for i in range(0,len(residueIndexes)):
        params = np.zeros((2,))
        params[0] = 100.0
        params[1] = 0.05
        bounds = opt.Bounds(lb=np.zeros_like(params)+0.0001)
        result = opt.minimize(minimization, params,
                          args=(concentrations, CSPs[i,:], protein_concentration, 0.003), bounds=bounds)
        Kds = result.x[:1]
        CSPsat = result.x[1:]
        tss = np.nansum((CSPs[i,:] - CSPs[i,:].mean())**2)/0.003**2
        sse = result.fun
        r_2 = (1.0 - sse/tss)
        if Kds[0] < 1000 and CSPsat[0] > 0.02 and r_2 > 0.9:
            selected_rows.append(i)
    #
    selected_rows = np.array(selected_rows)
    coords = coords[selected_rows,:]
    CSPs = CSPs[selected_rows,:]
    residueIndexes = residueIndexes[selected_rows]

    labels_dict = {}


    for i in range(1,20):
        if i == 1:
            centroids = np.mean(coords, axis=0)[np.newaxis,:]
            labels = np.zeros(len(coords),dtype=int)
        else:
            clustering = KMeans(n_clusters=i,n_init=1000).fit(coords)
            labels = clustering.labels_
            centroids = clustering.cluster_centers_
        #
        BIC = calculateBIC(coords, centroids, labels)
        print(f"Number of clusters: {i}, BIC: {BIC}")
        labels_dict[i] = (labels, BIC)
    #
    min = 1E100
    min_arg = 0
    for i in labels_dict.keys():
        labels = labels_dict[i][0]
        BIC = labels_dict[i][1]
        if BIC < min:
            min_arg = i
            min = BIC

    labels = labels_dict[min_arg][0]
    n_clusters = len(np.unique(labels))
    print(f"Number of clusters: {n_clusters}")
    params = np.zeros((n_clusters+len(labels),))
    params[:n_clusters] = 100.0
    params[n_clusters:] = 0.05
    bounds = opt.Bounds(lb=np.zeros_like(params))
    result = opt.minimize(minimization_labels, params, args=(labels, concentrations, CSPs, protein_concentration, 0.003),
                          bounds=bounds)
    print(result.x)
    mean_params, std_params = MonteCarloKds(CSPs, concentrations, protein_concentration, labels, result.x, 0.003)
    generateScaledFits(CSPs,concentrations,protein_concentration,residueIndexes,labels,mean_params, std_params)
    print("Done")
