from scipy import optimize as opt
import scipy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
import argparse
from pathlib import Path
from Bio import PDB
import re
from matplotlib import pyplot as plt
import math
from line_profiler import LineProfiler
import concurrent.futures
import pstats
def CSPBindingEquation(Kd: np.array, CSPsat: np.array, P: float, X: np.array):
    X_broadcasted = X[np.newaxis, :]  # Avoid recomputation
    A = X_broadcasted + P + Kd
    B = 4 * X_broadcasted * P
    sqrt_term = np.sqrt(np.square(A) - B)

    CSPs = CSPsat * ((A - sqrt_term) / (2 * P))

    #assert (position_calc != np.NaN).all()
    return CSPs
#
def minimization(params: np.array, X: np.array, CSP: np.array, P: float, error: np.array):

    n_kd = CSP.shape[0]
    log_Kd = params[:n_kd].reshape((CSP.shape[0], 1))
    CSPsat = params[n_kd:].reshape((CSP.shape[0], 1))

    assert log_Kd.base is params
    assert CSPsat.base is params


    residuals = np.square((CSP - CSPBindingEquation(np.exp(log_Kd), CSPsat, P, X))/(math.sqrt(2.0)*error))
    sum = np.nansum(residuals)
    #print(sum)
    return sum
#
def minimization_labels(params: np.array, labels: np.array, X: np.array, CSP: np.array, P: float, error):

    n_csp_params = CSP.shape[0]
    n_clusters = params.shape[0] - n_csp_params
    all_params = np.zeros((CSP.shape[0]+n_csp_params,))

    all_params[:CSP.shape[0]] = params[:n_clusters][labels]
    all_params[CSP.shape[0]:] = params[n_clusters:]

    return minimization(all_params, X, CSP, P, error)

#
def extractResIndex(s):
    match = re.fullmatch(r'[A-Za-z]+(\d+)N-H', s)
    return int(match.group(1)) if match else None

def extractConcentrations(s):
    match = re.fullmatch(r'[-+]?\d*\.?\d+\(mM\)', s)
    return float(re.search(r"[-+]?\d*\.?\d+", s).group()) if match else None

def generateScaledFits(CSPs: np.array, concentrations: np.array, protein_concentration: float, residueIndexes: np.array, labels: np.array, monte_params: np.array, plot_errors: np.array, scaling_factor: np.array):
    n_clusters = len(np.unique(labels))
    figsize = [ 6.4, 4.8]
    figsize[1] = figsize[1]/2 * n_clusters
    median_params = np.median(monte_params, axis=0)
    lb_params=np.quantile(monte_params, 0.05, axis=0)
    ub_params=np.quantile(monte_params, 0.95, axis=0)
    Kds = median_params[:n_clusters]
    CSPsat = median_params[n_clusters:]

    fig, axs = plt.subplots(nrows=n_clusters, ncols=2, figsize=figsize)
    if n_clusters == 1:
        axs = axs[np.newaxis,:]
    for i in range(0,n_clusters):
        cluster_mask = labels == i
        cluster_Kd = Kds[i:i+1]
        cluster_CSPsat = CSPsat[cluster_mask]
        cluster_CSPs = CSPs[cluster_mask]
        cluster_residue_indexes = residueIndexes[cluster_mask]
        cluster_plot_errors = plot_errors[cluster_mask]
        range_c = np.linspace(0, concentrations.max(), 100)
        CSP_calc = CSPBindingEquation(cluster_Kd, cluster_CSPsat[:,np.newaxis], protein_concentration, range_c.transpose())
        generateScaledFitFigure(cluster_CSPs,concentrations, CSP_calc, range_c,cluster_residue_indexes,median_params[i],lb_params[i],ub_params[i],cluster_plot_errors,axs[i,0],axs[i,1])
    #
    fig.tight_layout()
    return fig
#
def calculateCSPS(titration_data: np.array, scaling_factor: np.array):
    return np.sqrt(np.sum(((titration_data-titration_data[...,0:1,:])/scaling_factor)**2, axis=-1))
#
def generateScaledFitFigure(CSPs: np.array, concentrations: np.array, CSP_calc: np.array, range_c: np.array, cluster_residue_indexes: np.array,
                            cluster_Kd: float, cluster_Kd_lb: float, cluster_Kd_ub: float, plot_error: np.array, ax_actual: plt.Axes, ax_scaled: plt.Axes):


    ax_actual.plot(range_c,CSP_calc.transpose(),linestyle='-',marker='')

    colors = [ line.get_color() for line in ax_actual.get_lines() ]
    ax_actual.set_xlabel('Ligand Concentration (mM)')
    ax_actual.set_ylabel("CSP (ppm)")
    ax_actual.set_ylim([0,np.nanmax(CSPs)*1.05])
    ax_actual.set_xlim([0,ax_actual.get_xlim()[1]])
    kd_textbox_coord = [ ax_actual.get_xlim()[0]*1.15, ax_actual.get_ylim()[1]*0.85]
    ax_actual.text(kd_textbox_coord[0],kd_textbox_coord[1],f"Kd: {cluster_Kd:.0f}: {cluster_Kd_lb:.0f}, {cluster_Kd_ub:.0f} mM ", fontsize=12, color='black')
    for i in range(0,CSPs.shape[0]):
        ax_actual.errorbar(concentrations,CSPs[i,:],color=colors[i],yerr=plot_error[i,:],ecolor=colors[i],fmt='o')


    # generate scaled plots
    n_series = len(CSP_calc)
    interval = 1.0/(n_series+1)
    finalPoints = np.array([i*interval for i in range(1,n_series+1)])
    scaling_factors = (finalPoints / np.nanmax(CSP_calc,axis=1))[:,np.newaxis]

    scaled_cluster_titrations = CSPs * scaling_factors
    scaled_CSP_calc = CSP_calc * scaling_factors
    scaled_error = plot_error * scaling_factors[:,:,np.newaxis]

    ax_scaled.plot(range_c,scaled_CSP_calc.transpose(),linestyle='-',marker='')
    ax_scaled.set_xlabel('Ligand Concentration (mM)')
    ax_scaled.set_ylabel("Scaled CSP")
    ax_scaled.set_ylim([0,ax_scaled.get_ylim()[1]])
    for i in range(0,CSPs.shape[0]):
        ax_scaled.errorbar(concentrations,scaled_cluster_titrations[i,:],color=colors[i],yerr=scaled_error[i,:],ecolor=colors[i],fmt='o')
        ax_scaled.set_ylim([0.0,1.0])
        residue_textbox_x = ax_scaled.get_xlim()[1]*1.05
        ax_scaled.text(residue_textbox_x, scaled_CSP_calc[i,-1],f"{cluster_residue_indexes[i]}N-H",va='center',fontsize=12,color=colors[i])

def MonteCarloKds(positions: np.array, concentrations: np.array, protein_concentration: float, labels: np.array, params: np.array, error:np.array):
    sample_size = 1000
    error_adjusted_positions= scipy.stats.norm.rvs(loc=positions,scale=error[np.newaxis,np.newaxis,:],size=(sample_size,*positions.shape))
    scaling_factor = error/error[-1]
    error_adjusted_CSPs = calculateCSPS(error_adjusted_positions,scaling_factor)
    n_clusters = len(np.unique(labels))
    monte_params = np.zeros((len(error_adjusted_positions),n_clusters+positions.shape[0]))

    csp_errors = np.std(error_adjusted_CSPs,axis=0)

    for i in range(sample_size):
        error_mod_data = error_adjusted_CSPs[i]
        result = opt.minimize(minimization_labels, params, method="Powell", options={'maxiter': 1000000},
                              args=(labels, concentrations, error_mod_data, protein_concentration, csp_errors))
        monte_params[i, :] = result.x  # Store result
        print(f"Monte Carlo round: {i} completed: success? {result.success}")
    #
    return monte_params, error_adjusted_CSPs

def outputPML(residue_indexes:np.array, labels: np.array, Kds: np.array, Kd_error: np.array, outFilePath: str):

    outFile = open(outFilePath, "w")
    print("fetch 4DEP", file=outFile)
    print("remove chain A+B+C", file=outFile)
    print("create IL1B_obj, chain D", file=outFile)
    print("fetch 8C3U_A", file=outFile)
    print("color grey, IL1B_obj or 8C3U_A", file=outFile)
    print("select ligand_T9C, resname T9C", file=outFile)
    print("color magenta, ligand_T9C", file=outFile)
    print("align 8C3U_A, IL1B_obj", file=outFile)
    print("center IL1B_obj", file=outFile)
    print("remove 4DEP and chain D", file=outFile)
    print("show surface, IL1B_obj", file=outFile)

    cluster_indicies = np.unique(labels)
    colors = ["red", "blue", "green", "yellow", "orange", "magenta", "cyan", "white", "deeppurple"]
    cluster_names = []
    for cluster_i in cluster_indicies:
        mask = labels == cluster_i
        indexes = residue_indexes[mask].tolist()
        cluster_names.append(f"{cluster_i+1}_kd_{Kds[cluster_i]:.0f}_{Kd_error[cluster_i]:.0f}_mM")
        print(f"select {cluster_names[-1]} , resid " + "+".join(map(str, indexes)) + " and (IL1B_obj or 8C3U_A)", file=outFile)
        print(f"color {colors[cluster_i % len(colors)]}, {cluster_names[-1]}", file=outFile)

    #

    outFile.close()
def calculateError(positions: np.array, position_error: np.array):
    sample_size = 1000
    error_adjusted_positions = scipy.stats.norm.rvs(loc=positions, scale=position_error[np.newaxis, np.newaxis, :],
                                                    size=(sample_size, *positions.shape))
    scaling_factor = position_error / position_error[-1]
    error_adjusted_CSPs = calculateCSPS(error_adjusted_positions, scaling_factor)
    return np.std(error_adjusted_CSPs,axis=0)

    if optimal_k == 1:
        labels = np.zeros((positions.shape[0],),dtype=np.int32)
    else:
        clustering_object = SpectralClustering(n_clusters=optimal_k,affinity='precomputed',n_init=1000,verbose=True)
        labels = clustering_object.fit_predict(affinity_matrix)
    #
    print("Optimal number of clusters: ", optimal_k)
    return labels, optimal_k
#
def ClusterTitrationCurves(titration_data: Path, pdb_file: Path, chain: str, offset_index: int, protein_concentration: float, spectral_dimensions: list, error: list, output_directory:Path,name_stem:str ):

    output_directory.mkdir(exist_ok=True, parents=True)

    protein_concentration = protein_concentration
    titration_sheets = pd.read_excel(titration_data,sheet_name=None)
    concentrations = []

    spectral_dimensions = spectral_dimensions
    error = np.array(error)
    titration_data = []
    for sheet in titration_sheets:
        concentration = extractConcentrations(sheet)
        if concentration is None:
            raise Exception(f"Could not parse concentration from sheet name {sheet}")
        assert spectral_dimensions == titration_sheets[sheet].filter(regex=r'^w\d+$').columns.to_list()
        concentrations.append(concentration)
        df = titration_sheets[sheet]
        titration_data.append(df[spectral_dimensions].to_numpy())
    #

    #Extract Data
    titration_data = np.array(titration_data).transpose([1,0,2])
    concentrations = np.array(concentrations)
    structure = PDB.PDBParser().get_structure("protein", pdb_file)
    if len(structure) != 1:
        raise Exception("There should be exactly one model in the pdb")
    chain = structure[0][chain]
    residueIndexes = titration_sheets[list(titration_sheets.keys())[0]]['Assignment'].apply(extractResIndex).to_numpy(dtype=np.int32)
    residueIndexes += offset_index
    coords = [ chain[res_id]['N'].get_coord() for res_id in residueIndexes.tolist() ]
    coords = np.array(coords)
    CSPs = calculateCSPS(titration_data,error/error[-1])
    selected_rows = []
    Kds = []
    for i in range(0,len(residueIndexes)):
        params = np.zeros((2,))
        params[0] = math.log(1000)
        params[1] = np.nanmax(CSPs[i])


        result = opt.minimize(minimization, params, method='Powell',
                          args=(concentrations, CSPs[i:i+1], protein_concentration,1.4*error[-1:]))

        if not result.success:
            continue
        params = result.x
        Kd = math.exp(params[0])
        CSPsat = params[1]
        sse = result.fun
        residueIndex = residueIndexes[i]
        print(residueIndex, Kd , CSPsat, sse, CSPs[i].max())
        if Kd < 1000 and CSPsat > 0.02 and sse < 3 and np.nanmax(CSPs[i]) > 0.01:
            selected_rows.append(i)
            Kds.append(Kd)

    #
    if len(selected_rows) < 5:
        print("Too few residues selected: ")
        return
    #
    selected_rows = np.array(selected_rows)
    log_Kds = np.log(np.array(Kds)[:,np.newaxis])
    coords = coords[selected_rows,:]
    titration_data = titration_data[selected_rows,:,:]
    residueIndexes = residueIndexes[selected_rows]
    CSPs = CSPs[selected_rows]


    KdStats = []
    for i in range(1,5):
        fitted_mixture = GaussianMixture(n_components=i, covariance_type='full',init_params='k-means++',n_init=1000).fit(log_Kds)
        BIC = fitted_mixture.bic(log_Kds)
        labels = fitted_mixture.predict(log_Kds)
        KdStats.append((i,labels,BIC))
    #
    BIC_min = 9E100
    min_arg = 0
    for i in range(len(KdStats)):
        if KdStats[i][2] < BIC_min:
            min_arg = i
            BIC_min = KdStats[i][2]
    #


    labels = KdStats[min_arg][1]
    n_Kd_clusters = len(np.unique(labels))
    print(f"Number of Kd clusters: {n_Kd_clusters}")

    final_cluster_labels = np.zeros((len(CSPs),),dtype=int)


    n_final_clusters = 0
    for i in range(n_Kd_clusters):
        global_indexes = np.where(labels == i)[0]
        kd_specific_CSPs = CSPs[global_indexes]
        kd_specific_coords = coords[global_indexes,:]
        if len(kd_specific_coords) > 1:
            kd_specific_labels = MeanShift(cluster_all=True,bandwidth=10.0).fit_predict(kd_specific_coords)
            final_cluster_labels[global_indexes] = n_final_clusters+kd_specific_labels
            n_final_clusters += np.max(kd_specific_labels)+1
        else:
            final_cluster_labels[global_indexes] = n_final_clusters+1
            n_final_clusters += 1
    #
    print(f"Total number of spatial clusters: {n_final_clusters}")




    params = np.zeros((n_final_clusters+len(labels),))
    params[:n_final_clusters] = 1000.0
    params[n_final_clusters:] = 0.1

    result = opt.minimize(minimization_labels, params, method="Powell", options={'maxiter': 100000}, args=(final_cluster_labels, concentrations, CSPs, protein_concentration,np.array([0.003])))
    print(result.x)
    assert(result.success)
    '''profiler = LineProfiler()
    profiler.add_function(MonteCarloKds)
    profiler.add_function(minimization_labels)
    profiler.add_function(minimization)
    profiler.add_function(PositionBindingEquation)
    profiler.enable()'''
    monte_params, error_adjusted_CSPs = MonteCarloKds(titration_data, concentrations, protein_concentration,final_cluster_labels,params,error)
    '''profiler.disable()
    profiler.print_stats()'''
    top_percentile = np.quantile(error_adjusted_CSPs,0.95, axis=0) - np.median(error_adjusted_CSPs,axis=0)
    bottom_percentile = np.quantile(error_adjusted_CSPs,0.05, axis=0) - np.median(error_adjusted_CSPs,axis=0)
    plot_errors = np.abs(np.array([bottom_percentile, top_percentile]).transpose(1,0,2))
    #plot_errors = np.array([np.std(error_adjusted_CSPs, axis=0), np.std(error_adjusted_CSPs, axis=0)]).transpose(1,0,2)
    fig= generateScaledFits(CSPs,concentrations,protein_concentration,residueIndexes,final_cluster_labels,monte_params,plot_errors,1.4*error[-1:])

    plot_file = output_directory/f"{name_stem}_ClusterTitrationCurves.png"

    median_Kds = np.median(monte_params, axis=0)[:n_final_clusters]
    lb_Kds = np.quantile(monte_params, 0.05, axis=0)[:n_final_clusters]
    ub_Kds = np.quantile(monte_params, 0.95, axis=0)[:n_final_clusters]

    fig.savefig(plot_file)
    plt.close(fig)
    outputPML(residueIndexes,final_cluster_labels,median_Kds,(ub_Kds-lb_Kds)/2,output_directory/f"{name_stem}_clusters.pml")
    unique_clusters = np.unique(final_cluster_labels)
    cluster_data = [ ]
    for unique_cluster in unique_clusters:
        cluster_index = unique_cluster+1
        cluster_mask = final_cluster_labels == unique_cluster
        d = {}
        d["cluster_index"] = cluster_index
        d["Kd (mM)"] = median_Kds[unique_cluster]
        d["Kd_lower (mM)"] = lb_Kds[unique_cluster]
        d["Kd_upper (mM)"] = ub_Kds[unique_cluster]
        d["indicies"] = residueIndexes[cluster_mask].tolist()
        cluster_data.append(d)
    #
    df = pd.DataFrame(cluster_data)
    df.to_excel(output_directory/f"{name_stem}clusters_data.xlsx",index=False)
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--titration_data',required=True,type=Path,help='tiration data, an excel file of peak positions')
    parser.add_argument('--pdb_file',required=True,type=Path,help='pdb file')
    parser.add_argument('--chain',required=True,type=str,help='Chain code')
    parser.add_argument('--offset_index',default=0,type=int,help='Value to add to the assigned residue index to match the pdb indexing')
    parser.add_argument("--protein_concentration", required=True, type=float,help="Protein concentration")
    parser.add_argument("--spectral_dimensions", required=True, nargs='+', type=str,help="Spectral dimensions present e.g. w1 w2")
    parser.add_argument( "--error", required=True, type=float,nargs='+',help="error (ppm) for each indicated spectral dimension")
    parser.add_argument("--output_directory", required=True, type=Path,help="Output directory")
    parser.add_argument("--name_stem", required=True, type=str, help="Name stem")
    parser = parser.parse_args()


    ClusterTitrationCurves(parser.titration_data,parser.pdb_file,parser.chain,parser.offset_index,
                           parser.protein_concentration,parser.spectral_dimensions,parser.error,parser.output_directory,parser.name_stem)


    print("Done")
