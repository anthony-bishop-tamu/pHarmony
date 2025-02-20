import torch
import sys
import pandas as pd
from pathlib import Path
import argparse
import re

def extractTableFromFile(fileName: Path, ref_dim: int, target_dim: int) -> tuple:
    df = pd.read_csv(fileName, header=None)

    ten = df.iloc[target_dim+2:,ref_dim+1:]
    ten = torch.from_numpy(ten.astype(float).to_numpy())

    ref = df.iloc[target_dim+1:,0:ref_dim+1]
    names = ref.iloc[0]
    ref = ref[1:]
    ref.columns = names
    ref.reset_index(drop=True, inplace=True)

    targ = df.iloc[0:target_dim+1,ref_dim+1:].transpose()

    return ten, ref, targ

#
def pymol_out(fileName: Path, resid_matches: list, resid_csps: list):
    outFile = open(fileName, 'w')
    csp_selection_string = "select CSPs, resid \"" + "+".join(map(str, resid_csps)) + "\" and name N"
    matches_selection_string = "select matches, resid \"" + "+".join(map(str, resid_matches)) + "\" and name N"
    print("fetch 9ilb",file=outFile)
    print(f"color grey, all", file=outFile)
    print(f"show spheres, name N", file=outFile)
    print(csp_selection_string, file=outFile)
    print(matches_selection_string, file=outFile)
    print(f"color blue, matches", file=outFile)
    print(f"color red, CSPs", file=outFile)
    outFile.close()
#
def trackCSPs(reference_transfer_directory: Path, csp_transfer_directory: Path,
              reference_dims: int, csp_transfer_dims: int):
    reference_match_path = reference_transfer_directory/"Match_probabilities.csv"

    target_match_path = csp_transfer_directory/"Match_probabilities.csv"
    target_csv_path = csp_transfer_directory/"CSP_probabilities.csv"
    pymolOutput = csp_transfer_directory/"HeatMap.pml"

    reference_match = extractTableFromFile(reference_match_path, parser.reference_dim, parser.reference_dim)

    target_match = extractTableFromFile(target_match_path, parser.target_dim, parser.target_dim)
    target_csv = extractTableFromFile(target_csv_path, parser.target_dim, parser.target_dim)

    csp_probabilitity = target_match[0] * target_csv[0]
    reference_match_matrix = reference_match[0]
    target_match_matrix = target_match[0]
    chained_match_matrix = reference_match_matrix.unsqueeze(-1) * target_match_matrix
    chained_csp_matrix = chained_match_matrix * csp_probabilitity

    csp_probabilitities = chained_csp_matrix.sum(dim=(-1, -2))
    match_probabilities = chained_match_matrix.sum(dim=(-1, -2))
    reference_list = reference_match[1]

    reference_list['Match_Probability'] = match_probabilities
    reference_list['CSP_Probability'] = csp_probabilitities

    parser.output.parent.mkdir(exist_ok=True, parents=True)
    parser.pymol_output.parent.mkdir(exist_ok=True, parents=True)

    resid_pattern = re.compile(r'^[A-Za-z](\d+)N-H$')
    csp_list = reference_list[reference_list['CSP_Probability'] > 0.90]["Assignment"]
    match_list = reference_list[reference_list['Match_Probability'] > 0.90]["Assignment"]
    csp_list = [str(int(match.group(1)) - 1) for s in csp_list if (match := resid_pattern.match(s))]
    match_list = [str(int(match.group(1)) - 1) for s in match_list if (match := resid_pattern.match(s))]
    pymol_out(pymolOutput, match_list, csp_list)
    reference_list.to_csv(parser.output, index=False)
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_directory", type=Path, required=True, default=1, help="Reference Matching Directory")
    parser.add_argument("--reference_dim", type=int, required=True, help="Reference dimensionality")
    parser.add_argument("--target_directory", type=Path, required=True, default=1,
                        help="Target Matching Directory")
    parser.add_argument("--target_dim", type=int, required=True, help="Target dimensionality")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file of CSP and Matching Probs")
    parser.add_argument("--pymol_output", type=Path, required=True, help="Pymol_output of CSPs")

    parser = parser.parse_args()
    reference_match_path = parser.reference_directory/"Match_probabilities.csv"



    target_match_path = parser.target_directory/"Match_probabilities.csv"
    target_csv_path = parser.target_directory/"CSP_probabilities.csv"


    exit()
#