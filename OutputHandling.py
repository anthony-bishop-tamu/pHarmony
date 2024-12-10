
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
def buildPlot(matchingProbabilities: np.array,
              csp_probabilities: np.array,
              no_csp_match_distribution: torch.distributions.Distribution,
              csp_distribution: torch.distributions.Distribution,
              non_match_distribution: torch.distributions.Distribution,
              distributionPlot: str,
              distances: torch.tensor,
              confidence_cutoff: float = 0.90):

        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,15))
        matching_mask = matchingProbabilities > confidence_cutoff
        non_match_mask = matchingProbabilities < (1.0-confidence_cutoff)

        matches = distances[matching_mask].flatten()
        non_matches = distances[non_match_mask].flatten()
        low_nonMatches = non_matches[non_matches < matches.max()]

        bins = np.arange(0,matches.max(),1.0)
        ax1.hist(matches, bins=bins,color='blue',label='Matches',alpha=0.5,density=True)
        ax1.hist(low_nonMatches, bins=bins,weights=np.ones_like(low_nonMatches)/non_matches.size(),color='red',label='nonMatches',alpha=0.5)
        bins = torch.from_numpy(bins)

        csp_weight = csp_probabilities[matching_mask].sum()/matching_mask.sum()

        nocsp_pdf = no_csp_match_distribution.log_prob(bins).exp().detach().numpy()*(1-csp_weight)
        csp_pdf = csp_distribution.log_prob(bins).exp().detach().numpy()*csp_weight
        ax1.plot(bins,nocsp_pdf,color='red',label='No CSP')
        ax1.plot(bins,csp_pdf,color='blue',label='CSP Distribution')
        ax1.plot(bins,nocsp_pdf+csp_pdf,color='green',label='All matches',linestyle='--')
        ax1.plot(bins[1:],non_match_distribution.log_prob(bins[1:]).exp().detach().numpy(),color='orange',label='NonMatchDistribution')

        #log scale plot
        ax2.hist(matches, bins=bins, color='blue', label='Matches', alpha=0.5, density=True)
        ax2.hist(low_nonMatches, bins=bins, weights=np.ones_like(low_nonMatches) / non_matches.size(), color='red',
                 label='nonMatches', alpha=0.5)
        ax2.plot(bins, nocsp_pdf + csp_pdf, color='green', label='All matches', linestyle='--')
        ax2.plot(bins[1:], non_match_distribution.log_prob(bins[1:]).exp().detach().numpy(), color='orange',
                 label='NonMatchDistribution')

        ax2.set_xscale('log')
        ax2.set_yscale('log')

        ax1.set_xlabel('Normalized Squared Distances')
        ax2.set_xlabel('Normalized Squared Distances')
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Frequency')

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')

        log_bins = np.logspace(-0.5,np.log10(non_matches.max()),100)
        ax3.hist(non_matches, bins=log_bins, color='red', label='NonMatches', alpha=0.5, density=True)
        ax3.plot(log_bins, non_match_distribution.log_prob(torch.from_numpy(log_bins)).exp().detach().numpy(), color='red', label='Non matching distances')

        ax3.legend(loc='upper right')
        ax3.set_xscale('log')
        ax3.set_ylabel('Frequency')
        ax3.set_xlabel('Normalized Squared Distances')

        fig.tight_layout()
        fig.show()
        fig.savefig(distributionPlot)
#
def outputResults(matchingProbabilities: np.array,
                                csp_probabilities: np.array,
                                referencePeakList: tuple, #tuple of a pandas dataframe and the dimension (0 or 1) in the representation, and a list of the resonance columns
                                targetPeakList: tuple, #tuple of a pandas dataframe and the dimension (0 or 1) in the representation
                                transferedPeaklist: str,
                                highConfidenceTransferredPeakList: str,
                                probabilityTable: str,
                                chemicalShiftProbabilityTable: str,
                                confidenceCutoff: float = 0.90):

    if referencePeakList[1] == targetPeakList[1]:
        raise ValueError("Reference peak list and target peak list must be specified at different dimensions")
    if referencePeakList[1] == 0:
        rowIsReference = True
        row_peakList = referencePeakList
        col_peakList = targetPeakList
        assert(targetPeakList[1] == 1)
    #
    else:
        rowIsReference = False
        row_peakList = targetPeakList
        col_peakList = referencePeakList
        assert(referencePeakList[1] == 1)
    #
    #Get column labels!
    column_labels = pd.MultiIndex.from_frame(col_peakList[0][["Assignment"] + col_peakList[2]])
    row_labels = pd.MultiIndex.from_frame(row_peakList[0][["Assignment"] + row_peakList[2]])

    #output matching probability matrix
    probability_df = pd.DataFrame(matchingProbabilities)
    probability_df.columns = column_labels
    probability_df.index = row_labels
    probability_df.to_csv(probabilityTable)

    #output csp_probability matches
    matching_corrected_probability = csp_probabilities*matchingProbabilities
    csp_corrected_probability_df = pd.DataFrame(matching_corrected_probability)
    csp_corrected_probability_df.columns = column_labels
    csp_corrected_probability_df.index = row_labels
    csp_corrected_probability_df.to_csv(chemicalShiftProbabilityTable)

    #build transferred peak lists
    if rowIsReference:
        match_probs = matchingProbabilities.max(axis=1)
        column_indexes = matchingProbabilities.argmax(axis=1)
        row_indexes = np.arange(matchingProbabilities.shape[0])
        csp_confidences = matching_corrected_probability[row_indexes,column_indexes]
        referencePeaks = referencePeakList[0].iloc[row_indexes][["Assignment"]]
        targetPeaks = targetPeakList[0].iloc[column_indexes][targetPeakList[2]]
        referencePositions = referencePeakList[0].iloc[row_indexes][referencePeakList[2]]
        match_probs = pd.DataFrame(match_probs, columns=["MatchingProbability"])
        csp_probs = pd.DataFrame(csp_confidences, columns=["CSPProbability"])

    else:
        match_probs = matchingProbabilities.max(axis=0)
        row_indexes = matchingProbabilities.argmax(axis=0)
        column_indexes = np.arange(matchingProbabilities.shape[1])
        csp_confidences = matching_corrected_probability[row_indexes,column_indexes]
        referencePeaks = referencePeakList[0].iloc[column_indexes][["Assignment"]]
        targetPeaks = targetPeakList[0].iloc[row_indexes][targetPeakList[2]]
        referencePositions = referencePeakList[0].iloc[column_indexes][referencePeakList[2]]
        match_probs = pd.DataFrame(match_probs, columns=["MatchingProbability"])
        csp_probs = pd.DataFrame(csp_confidences, columns=["CSPProbability"])

    transfer_df = pd.concat([referencePeaks.reset_index(drop=True), targetPeaks.reset_index(drop=True), referencePositions, match_probs,csp_probs], axis=1)
    transfer_df.columns = ([ "Assignment"]+[label for label in targetPeakList[2] ] +
                               ["ref_"+label for label in referencePeakList[2]] + [ "MatchingProbability", "CSPProbability"])
    transfer_df.to_csv(transferedPeaklist)
    transfer_df[transfer_df['MatchingProbability'] > confidenceCutoff].to_csv(highConfidenceTransferredPeakList)

#

