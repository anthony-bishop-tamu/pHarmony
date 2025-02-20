import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
from collections import Counter

def pymol_out(fileName: Path, resid_matches: list, resid_csps: list):
    outFile = open(fileName, 'w')
    csp_selection_string = "select CSPs, resid \"" + "+".join(map(str, resid_csps))
    matches_selection_string = "select matches, resid \"" + "+".join(map(str, resid_matches))
    print("fetch 9ilb",file=outFile)
    print(f"color grey, all", file=outFile)
    print("show surface, all", file=outFile)
    print(csp_selection_string, file=outFile)
    print(matches_selection_string, file=outFile)
    print(f"color red, CSPs", file=outFile)
    outFile.close()
#

def ExtractCSPs(inputFile: Path, decoration: str):
    data = pd.read_csv(inputFile)

    data['distances'] = np.sqrt(((data['w1'] - data['ref_w1'])/10)**2 + (data['w2'] - data['ref_w2'])**2)
    resid_pattern = re.compile(r'^[A-Za-z](\d+)N-H$')
    resIndexes =  np.array([int(match.group(1))-1 for s in data['Assignment_ref'] if (match := resid_pattern.match(s))])
    data['ResIndex'] = resIndexes.astype(str)

    matches = data[data['MatchingProbability'] > 0.90]
    CSPs = matches[matches['distances'] > 0.010 ]

    fileName = Path(inputFile).stem+f"{decoration}_csps.pml"
    pymolPath = Path(inputFile).parent/fileName
    HeatMap(Path(pymolPath), Counter(CSPs['ResIndex'].tolist()))

    return CSPs['ResIndex']
#
def HeatMap(outFile: Path, IndexDict: Counter):
    outFile = open(outFile, 'w')
    print("fetch 4DEP", file=outFile)
    print("remove chain A+B+C", file=outFile)
    print("fetch 8C3U_A", file=outFile)
    print("create IL1B_obj, chain D", file=outFile)
    print("color grey, IL1B_obj or 8C3U_A", file=outFile)
    print("select ligand_T9C, resname T9C", file=outFile)
    print("color magenta, ligand_T9C", file=outFile)
    print("align 8C3U_A, IL1B_obj", file=outFile)
    print("center IL1B_obj", file=outFile)
    print("remove 4DEP and chain D", file=outFile)
    print("alter all, b=0", file=outFile)
    print("rebuild", file=outFile)
    print("select CSPs, resid \"" + "+".join(map(str, IndexDict.keys())) + " and (IL1B_obj or 8C3U_A)",file=outFile)
    for resi, b_value in IndexDict.items():
        outFile.write("alter resi %s and (IL1B_obj or 8C3U_A), b=%f\n" % (resi, b_value))
    print("select pH_Sensitive, (IL1B_obj or 8C3U_A) and resid 119+75+32+49+76+136+138+139+142", file=outFile)

    print("spectrum b, blue_red, CSPs",file=outFile)
    print(f"color green, 4DEP", file=outFile)
    outFile.close()
#
if __name__ == "__main__":
    index_list = []
    decoration = sys.argv[1]
    hitCounter = 0
    spectra_counter = 0
    for arg in sys.argv[2:]:
        resIndexes = ExtractCSPs(Path(arg),decoration).tolist()
        if len(resIndexes) > 4:
            index_list += resIndexes
            experiment_no_re = r'/(\d+)/'
            plate_no_re = r'plate_(\d+)'
            ExpNo = re.search(experiment_no_re, arg).group(1)
            plate_no = re.search(plate_no_re, arg).group(1)
            print(arg, plate_no, ExpNo, ":"+",".join(map(str, resIndexes)),sep=",")
            hitCounter += 1
        #
        spectra_counter += 1
    #
    print(f"{hitCounter} hits, {spectra_counter} spectra")
    IndexDict = Counter(index_list)
    HeatMap(f"All_hits_{decoration}.pml",IndexDict)
    print(IndexDict)
    exit()


