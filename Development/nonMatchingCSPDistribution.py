import torch
import pandas as pd
DataDir="/Volumes/homes/Anthony_Bishop/LaboratoryFiles/Data/FBDD_RM/Interleukin1beta/NMR-Spectra/BrukerDirectories/800/SAR_Screening/Feb06-2020-nmrsu"
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
def readData(data_dir):
    list_files = glob.glob(data_dir + "/*.list")
    parsedFiles = []
    dfs = {f: pd.read_csv(f,sep="\s+") for f in list_files}

    return dfs
#
def calculateDistances(dfs):
    all_identical_assignments=[]
    all_non_identical_assignments=[]
    for df_name1, df1 in dfs.items():
        for df_name2, df2 in dfs.items():
            if df_name1 != df_name2:
                merged_df = pd.merge(df1, df2, suffixes=('_1', '_2'),how="cross")
                merged_df['distance'] = np.sqrt(((merged_df['w1_1'] - merged_df['w1_2'])*0.1014) ** 2 +
                                                (merged_df['w2_1'] - merged_df['w2_2']) ** 2)/0.003
                matching = merged_df[merged_df["Assignment_1"] == merged_df["Assignment_2"]]
                nonmatching = merged_df[merged_df["Assignment_1"] != merged_df["Assignment_2"]]
                all_non_identical_assignments += nonmatching["distance"].tolist()
                all_identical_assignments += matching["distance"].tolist()
                #print(matching)
            #
        #
    #
    return all_identical_assignments, all_non_identical_assignments
#

parsedFiles = readData(DataDir)
matching,nonMatching = calculateDistances(parsedFiles)
matching = np.array(matching)
print("Minimum: non Matching: ")
#min_val = min(min(matching), min(nonMatching))
#max_val = max(max(matching), max(nonMatching))
bin_width = 0.3
bins = np.arange(0,  max(nonMatching) + bin_width, bin_width)

#plt.hist(matching,bins=bins,label="Matching",color="blue",alpha=0.5,density=True)
n,bins,patches = plt.hist(nonMatching,bins=bins,label="NonMatching",color="red",alpha=0.5,density=True)
x = np.linspace(min(bins), max(bins), 1000)
chi2_pdf = stats.chi2.pdf(x, 1)/2
shape, loc, scale = stats.weibull_min.fit(nonMatching)
weilbull_pdf = stats.weibull_min.pdf(x, shape, loc=loc, scale=scale)
plt.plot(x, weilbull_pdf, 'r-', label=f'WeilbullDistribution')
#plt.plot(x, chi2_pdf, 'g-', label=f'Chi2 Dof={1}')
print(bins[1]-bins[0])
plt.xlabel('Distance (error normalized 1H 0.003 ppm)')
plt.ylabel('Frequency')
plt.title('Histogram of Distances')

plt.legend()
plt.show()

