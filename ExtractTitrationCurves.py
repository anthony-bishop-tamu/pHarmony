import pandas as pd

from PeakMatcher import MatchPeaks, getPeakPositionsFromFile
import argparse
from pathlib import Path
import torch
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list', required=True, help='reference peak list', type=Path)
    parser.add_argument('--titration_concentrations',required=True, nargs="+",type=float,help='titration concentrations')
    parser.add_argument('--titration_peak_lists', required=True, nargs='+', type=Path, help='list of titration peak lists (in order)')
    parser.add_argument('--w1_uncertainty_ppm', default=0.015, type=float, help='w1 uncertainty in ppm (15N)')
    parser.add_argument('--w2_uncertainty_ppm', default=0.0015, type=float, help='w2 uncertainty in ppm (1H)')
    parser.add_argument('--R2Cutoff', default=0.9, type=float, help='R2Cutoff value')
    parser.add_argument("--output_directory", required=True, type=Path, help="Output directory")
    parser = parser.parse_args()
    parser.output_directory = parser.output_directory.resolve()
    if len(parser.titration_peak_lists) != len(parser.titration_concentrations):
        print("There must be a titration peak list for every provided concentration")
        assert False
    if parser.titration_concentrations[0] != 0.0:
        print("First concentration must the control (0.0 mM)")
        assert False

    peakLists = []
    fixedError = [ parser.w1_uncertainty_ppm, parser.w2_uncertainty_ppm ]
    for i in range(len(parser.titration_concentrations)):
        if i == 0:
            assigned_peak_positions, assigned_peaks = getPeakPositionsFromFile(parser.reference_peak_list,['w1', 'w2'],fixedError=fixedError)
            reference_peak_positions, reference_peaks = getPeakPositionsFromFile(parser.titration_peak_lists[i],
                                                                                 ['w1', 'w2'],fixedError=fixedError)
            matchingDistribution, matchingProbabilities, distances_normalized_squared, reference_offset = MatchPeaks(
               assigned_peak_positions, reference_peak_positions, expected_fraction_missing=0.02,
                expected_fraction_csp=0.05, fixedOffset=torch.tensor([0, 0], dtype=torch.float))
            value, index = torch.max(matchingProbabilities, dim=1)

            reference_peaks.iloc[index[value > 0.9].numpy(),0 ] = assigned_peaks.iloc[torch.arange(len(assigned_peaks))[value > 0.9].numpy(),0].astype(str).reset_index(drop=True)
            peak_mask = reference_peaks['Assignment'] != "?-?"
            reference_peaks = reference_peaks[peak_mask].copy()
            reference_peak_positions = reference_peak_positions[peak_mask.to_numpy()]

            peakLists.append((0.0,reference_peaks))
            name = parser.reference_peak_list.stem + "_transferredPeaks.list"
            reference_peaks.to_csv(parser.output_directory/name, index=False,sep='\t')

        #
        else:
            target_peak_positions, target_peaks = getPeakPositionsFromFile(parser.titration_peak_lists[i], ['w1', 'w2'],fixedError=fixedError)
            matchingDistribution, matchingProbabilities, distances_normalized_squared, reference_offset = MatchPeaks(reference_peak_positions,target_peak_positions, expected_fraction_missing=0.02, expected_fraction_csp=0.05,fixedOffset=torch.tensor([0,0],dtype=torch.float))

            value, index = torch.max(matchingProbabilities,dim=1)
            target_peaks.iloc[index[value > 0.9].numpy(),0 ] = reference_peaks.iloc[torch.arange(len(reference_peaks))[value > 0.9].numpy(),0].astype(str).to_numpy()
            peak_mask = target_peaks['Assignment'] != "?-?"
            target_peaks = target_peaks[peak_mask].copy()
            target_peak_positions = target_peak_positions[peak_mask.to_numpy()]
            peakLists.append((parser.titration_concentrations[i],target_peaks))
            reference_peaks = target_peaks
            reference_peak_positions = target_peak_positions

            name = parser.titration_peak_lists[i].stem + "_transferredPeaks.list"
            target_peaks.to_csv(parser.output_directory/name, index=False,sep='\t')
    #
    parser.output_directory.mkdir(exist_ok=True,parents=True)
    peakPositionFile = parser.output_directory / "peakPositions.xlsx"
    with pd.ExcelWriter(peakPositionFile,engine="openpyxl") as writer:
        for peakList in peakLists:
            # add NaNs for missing peaks

            merged_peakList = assigned_peaks[['Assignment']].merge(peakList[1], on='Assignment', how='left' )
            merged_peakList.to_excel(writer, sheet_name=f"{peakList[0]}(mM)", index=False)
        #





