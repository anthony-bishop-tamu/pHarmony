import scipy.stats as stats
import numpy as np
import pandas as pd
import torch
import argparse
from matplotlib import pyplot as plt
def getPeakPositionsFromFile(filename, cs_cols, uncertaintycols=None, fixedError=None):
    df = pd.read_csv(filename,sep="\s+")
    positions = df[cs_cols].to_numpy(dtype=np.float32)
    if uncertaintycols is not None:
        uncertainties = df[uncertaintycols].to_numpy(dtype=np.float32)
    elif fixedError is not None:
        uncertainties = np.zeros_like(positions)
        uncertainties[:,:] = np.array(fixedError,dtype=np.float32)[np.newaxis,:]
    else:
        raise ValueError("Must specify either uncertaintycols or fixedError")
    #
    return torch.from_numpy(np.stack((positions,uncertainties),axis=2))
#
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_peak_list1', type=str, help='reference peak list 1 filename')
    parser.add_argument( '--reference1_cs_column_names', type=str, nargs='+', help='reference 1 cs column names')
    parser.add_argument('--reference_peak_list2', type=str, help='reference peak list 2 list filename')
    parser.add_argument('--reference2_cs_column_names', type=str, nargs='+', help='reference 2 cs column names')

    return parser.parse_args()

#
if __name__ == '__main__':
   referenceFile = '/Users/anthonybishop/LaboratoryFiles/Programs/PeakMatcher/TestData/Manually_Picked_Peaks/referenceSpectraLists.txt'
   referenceFile = open(referenceFile, 'r').readlines()
   total_distances = np.array([])
   for f1 in referenceFile:
       for f2 in referenceFile:
           if f1 == f2:
               continue
           else:
               f1 = f1.strip()
               f2 = f2.strip()
               s = 0.0003
               reference_peaks = getPeakPositionsFromFile(f1,['w1','w2'],fixedError=[10*s,1*s])
               target_peaks = getPeakPositionsFromFile(f2,['w1','w2'],fixedError=[10*s,1*s])
               component_distances = target_peaks[:, :, 0] - reference_peaks[:, :, 0]
               component_distances_normalized = component_distances[:, :] / np.sqrt(
                   reference_peaks[:, :, 1] ** 2 + target_peaks[:, :, 1] ** 2)
               component_distances_squared_normalized = component_distances_normalized ** 2

               distances = np.sqrt(component_distances_squared_normalized.sum(dim=-1).flatten().numpy())
               total_distances = np.append(total_distances,distances)
        #
    #
   distances = total_distances.flatten()
   distances = distances**2
   distances = distances[distances < 20]
   loc,my_scale = stats.halfnorm.fit(distances,floc=0)

   print(my_scale)
   bins = np.linspace(0, distances.max(), 50)
   plt.hist(distances, bins=bins,alpha=0.5, density=True)
   plt.plot(bins, stats.chi2(2).pdf(bins))
   plt.show()
