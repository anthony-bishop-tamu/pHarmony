import pandas as pd

from PeakMatcher import MatchPeaks, getPeakPositionsFromFile, NoPeaksFoundError
import argparse
from pathlib import Path
import torch
import numpy as np
import os

def ExtractTitrationCurves(assignment_peak_list: Path, titration_peak_lists: list, titration_concentrations: list, spectral_dimensions: list, error: list, output_directory: Path, name_stem: str):
    output_directory = output_directory.resolve()
    output_directory.mkdir(exist_ok=True, parents=True)
    i = 0
    while i < len(titration_peak_lists):
        if not Path(titration_peak_lists[i]).is_file():
            del titration_peak_lists[i]
            del titration_concentrations[i]
        else:
            try:
                target_peak_positions, target_peaks = getPeakPositionsFromFile(titration_peak_lists[i],
                                                                               spectral_dimensions,
                                                                               fixedError=error)
                i+=1
            except NoPeaksFoundError:
                del titration_peak_lists[i]
                del titration_concentrations[i]
        #
    #
    if len(titration_peak_lists) != len(titration_concentrations):
        print("There must be a titration peak list for every provided concentration")
        assert False
    if titration_concentrations[0] != 0.0:
        print("First concentration must the control (0.0 mM)")
        assert False

    peakLists = []
    fixedError = error
    for i in range(len(titration_concentrations)):
        if i == 0:
            assigned_peak_positions, assigned_peaks = getPeakPositionsFromFile(assignment_peak_list, spectral_dimensions,
                                                                               fixedError=fixedError)
            reference_peak_positions, reference_peaks = getPeakPositionsFromFile(titration_peak_lists[i],
                                                                                 spectral_dimensions, fixedError=fixedError)
            matchingDistribution, matchingProbabilities, distances_normalized_squared, reference_offset = MatchPeaks(
                assigned_peak_positions, reference_peak_positions, expected_fraction_missing=0.02,
                expected_fraction_csp=0.05, fixedOffset=torch.tensor([0, 0], dtype=torch.float))
            value, index = torch.max(matchingProbabilities, dim=1)

            reference_peaks.iloc[index[value > 0.9].numpy(), 0] = assigned_peaks.iloc[
                torch.arange(len(assigned_peaks))[value > 0.9].numpy(), 0].astype(str).reset_index(drop=True)
            peak_mask = reference_peaks['Assignment'] != "?-?"
            reference_peaks = reference_peaks[peak_mask].copy()
            reference_peak_positions = reference_peak_positions[peak_mask.to_numpy()]

            peakLists.append((0.0, reference_peaks))
            name = assignment_peak_list.stem + "_transferredPeaks.list"
            reference_peaks.to_csv(output_directory / name, index=False, sep='\t')

        #
        else:
            target_peak_positions, target_peaks = getPeakPositionsFromFile(titration_peak_lists[i], spectral_dimensions,
                                                                           fixedError=fixedError)
            matchingDistribution, matchingProbabilities, distances_normalized_squared, reference_offset = MatchPeaks(
                reference_peak_positions, target_peak_positions, expected_fraction_missing=0.02,
                expected_fraction_csp=0.05, fixedOffset=torch.tensor([ 0 for i in range(len(spectral_dimensions))], dtype=torch.float))

            value, index = torch.max(matchingProbabilities, dim=1)
            target_peaks.iloc[index[value > 0.9].numpy(), 0] = reference_peaks.iloc[
                torch.arange(len(reference_peaks))[value > 0.9].numpy(), 0].astype(str).to_numpy()
            peak_mask = target_peaks['Assignment'] != "?-?"
            target_peaks = target_peaks[peak_mask].copy()
            target_peak_positions = target_peak_positions[peak_mask.to_numpy()]
            peakLists.append((titration_concentrations[i], target_peaks))
            reference_peaks = target_peaks
            reference_peak_positions = target_peak_positions

            name = titration_peak_lists[i].stem + "_transferredPeaks.list"
            target_peaks.to_csv(output_directory / name, index=False, sep='\t')
    #
    peakPositionFile = output_directory / f"{name_stem}_peakPositions.xlsx"
    with pd.ExcelWriter(peakPositionFile, engine="openpyxl") as writer:
        for peakList in peakLists:
            # add NaNs for missing peaks

            merged_peakList = assigned_peaks[['Assignment']].merge(peakList[1], on='Assignment', how='left')
            merged_peakList.to_excel(writer, sheet_name=f"{peakList[0]}(mM)", index=False)
        #
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment_peak_list', required=True, help='reference peak list', type=Path)
    parser.add_argument('--titration_concentrations',required=True, nargs="+",type=float,help='titration concentrations')
    parser.add_argument('--titration_peak_lists', required=True, nargs='+', type=Path, help='list of titration peak lists (in order)')
    parser.add_argument('--error', required=True, type=float, nargs='+', help='error in each dimension (approx 0.0015 for 1H and 0.015 for N')
    parser.add_argument("--output_directory", required=True, type=Path, help="Output directory")
    parser.add_argument("--spectral_dimensions", required=True, type=str, nargs="+", help="Spectral dimensions")
    parser.add_argument("--name_stem", required=True, type=str, help="Name stem")
    parser = parser.parse_args()

    ExtractTitrationCurves(parser.assignment_peak_list,parser.titration_peak_lists, parser.titration_concentrations, parser.spectral_dimensions, parser.error, parser.output_directory,parser.name_stem)
    print("Done")
#

