
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
def buildPlot(matchingProbabilities: torch.tensor,
              csp_mixture_weights: np.array,
              no_csp_match_distribution: torch.distributions.Distribution,
              csp_distribution: torch.distributions.Distribution,
              distances: torch.tensor,
              confidence_cutoff: float = 0.90):

        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(15,15))
        matching_mask = matchingProbabilities > confidence_cutoff
        non_match_mask = matchingProbabilities < (1.0-confidence_cutoff)

        matches = distances[matching_mask].flatten()
        non_matches = distances[non_match_mask].flatten()

        low_max = distances.max()

        if matches.numel() > 0:
            low_max = matches.max()

        low_nonMatches = non_matches[non_matches < low_max]

        bins = np.arange(0.0005,low_max,1.0)
        ax1.hist(matches, bins=bins,color='blue',label='Matches',alpha=0.5,weights=np.ones_like(matches)/(non_matches.size()[0]+matches.size()[0]))
        ax1.hist(low_nonMatches, bins=bins,weights=np.ones_like(low_nonMatches)/(non_matches.size()[0]+matches.size()[0]),color='red',label='nonMatches',alpha=0.5)
        bins = torch.from_numpy(bins)

        matching_mixture_weights = np.array([matchingProbabilities.sum(), (1.0 - matchingProbabilities).sum()])/matchingProbabilities.numel()

        nocsp_pdf = no_csp_match_distribution.log_prob(bins).exp().detach().numpy()
        csp_pdf = csp_distribution.log_prob(bins).exp().detach().numpy()
        ax1.plot(bins,nocsp_pdf*csp_mixture_weights[0]*matching_mixture_weights[0],color='red',label='No CSP')
        ax1.plot(bins,csp_pdf*csp_mixture_weights[1]*matching_mixture_weights[0],color='blue',label='CSP Distribution')
        ax1.plot(bins,(nocsp_pdf*csp_mixture_weights[0]+csp_pdf*csp_mixture_weights[1])*matching_mixture_weights[0],color='green',label='All matches',linestyle='--')
       # ax1.plot(bins[1:],non_match_distribution.log_prob(bins[1:]).exp().detach().numpy()*matching_mixture_weights[1],color='orange',label='NonMatchDistribution')

        #log scale plot
        log_bins = np.logspace(-0.5,np.log10(non_matches.max()),100)

        matches_hist, edges = np.histogram(matches, bins=log_bins)
        nonmatches_hist, edges = np.histogram(non_matches, bins=log_bins)

        bin_widths = np.diff(edges)

        total = np.sum(matches_hist)+np.sum(nonmatches_hist)

        ax2.bar(edges[:-1], matches_hist/(bin_widths*total), width=bin_widths, align='edge', color='blue', label='Matches', alpha=0.5)
        ax2.bar(edges[:-1], nonmatches_hist/(bin_widths*total), width=bin_widths, align='edge', color='red', label='nonMatches', alpha=0.5)

        ax2.set_yscale('log')
        ax2_ylim = ax2.get_ylim()
       # ax2.plot(log_bins, non_match_distribution.log_prob(torch.from_numpy(log_bins)).exp().detach().numpy()*matching_mixture_weights[1], color='orange',
       #          label='NonMatchDistribution')
        ax2.plot(log_bins,no_csp_match_distribution.log_prob(torch.from_numpy(log_bins)).exp().detach().numpy()*csp_mixture_weights[0]*matching_mixture_weights[0],color='red',label='No CSP')
        ax2.plot(log_bins,csp_distribution.log_prob(torch.from_numpy(log_bins)).exp().detach().numpy()*csp_mixture_weights[1]*matching_mixture_weights[0],color='blue',label='CSP Distribution')

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(ax2_ylim)

        ax1.set_xlabel('Normalized Squared Distances')
        ax2.set_xlabel('Normalized Squared Distances')
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Frequency')

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')


        #ax3.hist(non_matches, bins=log_bins, color='red', label='NonMatches', alpha=0.5, density=True)
        #ax3.plot(log_bins, non_match_distribution.log_prob(torch.from_numpy(log_bins)).exp().detach().numpy(), color='red', label='Non matching distances')

        #ax3.legend(loc='upper right')
        #ax3.set_xscale('log')
        #ax3.set_ylabel('Frequency')
        #ax3.set_xlabel('Normalized Squared Distances')

        fig.tight_layout()
        return fig
#
def outputResults(matchingProbabilities: np.array,
                                state_probability: np.array,
                                referencePeakList: tuple, #tuple of a pandas dataframe and the dimension (0 or 1) in the representation, and a list of the resonance columns
                                targetPeakList: tuple, #tuple of a pandas dataframe and the dimension (0 or 1) in the representation
                                transferedPeaks: str,
                                highConfidenceTransferedPeaks: str,
                                highConfidenceTransferedPeakList: str,
                                probabilityTable: str,
                                chemicalShiftProbabilityTable: str,
                                confidenceCutoff: float = 0.90):


    row_peakList = referencePeakList
    col_peakList = targetPeakList
    #Get column labels!
    column_labels = pd.MultiIndex.from_frame(col_peakList[0][["Assignment"] + col_peakList[1]])
    row_labels = pd.MultiIndex.from_frame(row_peakList[0][["Assignment"] + row_peakList[1]])

    #output matching probability matrix
    probability_df = pd.DataFrame(matchingProbabilities)
    probability_df.columns = column_labels
    probability_df.index = row_labels
    probability_df.to_csv(probabilityTable)

    #output csp_probability matches
    csp_corrected_probability_df = pd.DataFrame(state_probability[:,:,1])
    csp_corrected_probability_df.columns = column_labels
    csp_corrected_probability_df.index = row_labels
    csp_corrected_probability_df.to_csv(chemicalShiftProbabilityTable)

    #build transferred peak lists

    match_probs = matchingProbabilities.max(axis=1)
    column_indexes = matchingProbabilities.argmax(axis=1)
    row_indexes = np.arange(matchingProbabilities.shape[0])
    csp_confidences = state_probability[row_indexes,column_indexes,1]
    referencePeaks = referencePeakList[0].iloc[row_indexes][["Assignment"]]
    targetPeaks = targetPeakList[0].iloc[column_indexes][["Assignment"]]
    targetPositions = targetPeakList[0].iloc[column_indexes][targetPeakList[1]]
    referencePositions = referencePeakList[0].iloc[row_indexes][referencePeakList[1]]
    match_probs = pd.DataFrame(match_probs, columns=["MatchingProbability"])
    csp_probs = pd.DataFrame(csp_confidences, columns=["CSPProbability"])

    transfer_df = pd.concat([referencePeaks.reset_index(drop=True), referencePositions,
                             targetPeaks.reset_index(drop=True), targetPositions.reset_index(drop=True) ,
                             match_probs,csp_probs], axis=1)
    transfer_df.columns = ([ "Assignment_ref"]+["ref_"+label for label in referencePeakList[1]]+
                           ["Assignment_target"]+[label for label in targetPeakList[1] ] +
                               [ "MatchingProbability", "CSPProbability"])
    transfer_df.to_csv(transferedPeaks,index=False)
    transfer_df[transfer_df['MatchingProbability'] > confidenceCutoff].to_csv(highConfidenceTransferedPeaks,index=False)

    transfer_df.rename(columns={"Assignment_ref": "Assignment"}, inplace=True)

    transfer_df[transfer_df['MatchingProbability'] > confidenceCutoff][
        ["Assignment"]+[label for label in targetPeakList[1]]
        ].to_csv(highConfidenceTransferedPeakList,index=False,sep='\t')


#

