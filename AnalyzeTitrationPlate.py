from ClusterTitrationCurves import ClusterTitrationCurves
from ExtractTitrationCurves import ExtractTitrationCurves
import pandas as pd
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_spreadsheet', type=Path, required=True, help='Experiment spreadsheet path')
    parser.add_argument( '--assigned_peak_list', type=Path, required=True, help='assigned peak list path')
    parser.add_argument('--pdb_file', type=Path, required=True, help='PDB file path')
    parser.add_argument('--chain', type=str, required=True, help='pdb chain')
    parser.add_argument( '--offset_index', type=int, default=0, help='number to add to assignment resindex to match pdb')
    parser.add_argument('--protein_concentration', type=float, required=True, help="Protein concentration (mM)")
    parser.add_argument('--spectral_dimensions', type=str, nargs='+', required=True, help='Spectral dimensions (e.g. w1 w2)')
    parser.add_argument('--position_error', type=float, nargs='+', required=True, help='Position error (e.g. 0.015 0.0015)')
    parser.add_argument('--output_directory', type=Path, required=True, help='Output directory')
    parser.add_argument( '--skip_if_done',action='store_true', required=False, default=False, help='Skip if already done')
    parser.add_argument( '--skip_peak_matching',action='store_true', required=False, default=False, help='Skip if peak matching is already done')
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(exist_ok=True, parents=True)

    experiment_spreadsheet = pd.read_excel(args.experiment_spreadsheet,sheet_name='Sheet1')

    experiment_spreadsheet['PeakList_path'] = experiment_spreadsheet['Experiment Directory'] + '/' + experiment_spreadsheet['ExpNo'].astype(str) + '/' + experiment_spreadsheet['PeakList']

    chemIDs = experiment_spreadsheet['ChemID'].unique()
    controlEntries = experiment_spreadsheet[experiment_spreadsheet['ChemID'] == 'Control']
    for chemID in chemIDs:
        if chemID == 'Control':
            continue

        chemID_output_dir = output_directory / chemID
        chemID_output_dir.mkdir(exist_ok=True, parents=True)
        peakPositionsFile = chemID_output_dir/"ExtractedTitrationCurves"/f"{chemID}_peakPositions.xlsx"
        clusterPymolFile = chemID_output_dir/"ClusterTitrationCurves"/f"{chemID}_clusters.pml"
        clusterTitrationCurves = chemID_output_dir/"ClusterTitrationCurves"/f"{chemID}_ClusterTitrationCurves.png"

        if args.skip_if_done and peakPositionsFile.is_file() and clusterPymolFile.is_file() and clusterTitrationCurves.is_file():
            continue

        entries = experiment_spreadsheet[experiment_spreadsheet['ChemID'] == chemID]
        control_entry = controlEntries.iloc[[0]]
        assert(control_entry['CompoundFinalAqueousConcentration (mM)'].iloc[0] == 0)

        titration_peak_lists = control_entry['PeakList_path'].apply(Path).to_list() + entries['PeakList_path'].apply(Path).to_list()
        if not args.skip_peak_matching or not peakPositionsFile.is_file():
            ExtractTitrationCurves(args.assigned_peak_list,titration_peak_lists,
                               [0]+entries['CompoundFinalAqueousConcentration (mM)'].to_list(),
                               args.spectral_dimensions,
                               args.position_error,
                               chemID_output_dir/"ExtractedTitrationCurves",chemID)
        #

        ClusterTitrationCurves(chemID_output_dir/"ExtractedTitrationCurves"/f"{chemID}_peakPositions.xlsx",
                               args.pdb_file,args.chain,args.offset_index,
                               args.protein_concentration,
                               args.spectral_dimensions,
                               args.position_error,
                               chemID_output_dir/"ClusterTitrationCurves",chemID)
        #
    #

