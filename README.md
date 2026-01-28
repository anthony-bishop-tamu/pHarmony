# pHarmony

## Requirements

- **Python**: `>= 3.9`

### Python Dependencies

The following packages are required:

- `numpy>=1.24.0,<2 #Issue with dependencies on some platforms requiring 1`
- `pandas>=2.0`
- `torch>=2.0`
- `matplotlib>=3.2`
- `scipy>=1.10`
- `scikit-learn>=1.6.1`
- `tqdm>=4.67.1`

---

## Installation (Three Easy Steps)

### 1) Create a Conda environment

```bash
conda create --name pHarmony python=3.9
```

### 2) Activate the environment

```bash
conda activate pHarmony
```

### 3) Install `pHarmony`

Install via HTTPS 

```bash
pip install git+https://github.com/anthony-bishop-tamu/pHarmony.git@main  
# pip install git+https://github.com/anthony-bishop-tamu/pHarmony.git@v1.3.6 is the version at the time of publishing
# pip install git+https://github.com/anthony-bishop-tamu/pHarmony.git@main for the latest stable release
```

---

## Verify Installation

To test that installation worked, run in your terminal:

```bash
pHarmony
```
This command is the standalone peak matching algorithm
You should see usage/help information printed to the terminal.
Be sure the conda environment is active to run pHarmony

## pHarmony Usage
### Here is an example usage for pHarmony (2D matching example)
```bash
pHarmony    --reference_peak_list reference.list \ 
            --reference_cs_column_names w2 w3 \
            --reference_peak_list_error 0.015 0.0015 \
            --target_peak_list target.list \
            --target_cs_column_names w2 w3 \
            --target_peak_list_error 0.015 0.0015 \
            --output_directory output_directory
            
            
### --reference_peak_list : A sparky formatted peak list
### --reference_cs_column_names : The column names in the reference peak list that contain the dimensions to be matched
### --reference_peak_list_error : The uncertainty in the reference peak positions (dimensions corresponding to column names)
### --target_peak_list : A sparky formatted peak list
### --target_cs_column_names : The column names in the reference peak list that contain the dimensions to be matched 
### each dimension specified here will be matched to the corresponding dimension in the reference_cs_column_names
### --target_peak_list_error 0.03 0.003 : uncertainty in the target peak position
### --output_directory : The directory to contain the output files
### --CSP_scaling_factors: factors by which to scale the chemical shift dimensions when calculating CSPs in the output files (e.g. 0.101 1.0 for 15N-1H HSQC)

### OPTIONAL ARGUMENTS
### --confidence_cutoff : Default 0.95 - confidence cutoff used gor generating final peak lists
### --expected_fraction_csp : default 0.1 - Prior estimate of the fraction of matched peaks that undergo a CSP (Keep under 0.5)
### --variance_scale_fraction_csp: default 2 - Scaling factor to create Beta prior variance: variance = variance_scale_fraction_csp*expected_fraction_csp**2
### --expected_max_csp: Expected maximum chemical shift: default 0.1 ppm (good for proton-detected experiments) (in ppm of the dimension with the lowest uncertainty)
###  if this is set too large, sampling might become extremely slow
### --log_file: flag for including log file
### --gradient_convergence : Threshold of parameter gradient norms to determine convergence during optimization

```

### Output File Description
Each run generates the following output files

- #### CSP_probabilities.csv - a table that describes the posterior probability that the distance between two peaks is due to a chemical shift (given that its between matching peaks)

- #### Match_probabilities.csv - a table that describes the posterior probability that any two peak matches (as calculated from the final converged sample in the EM loop)

    - Note rows and columns may not sum to 1.0 (reference and target peaks might have sampled no match states)

- #### *_fittedDistributions.png - a plot showing the fit of the matching likelihoods and the distribution of the normalized square distances
- #### *_transferred.csv - a list that indicates the highest probability match for each reference peak, the probability that the reference peak is missing and other statistics
- #### *_transferred_HC.csv - same as *_transferred.csv, but filtered, rejecting any matches below the specified confidence cutoff
- #### *_transferred.list - the inputted target_list which only contains peaks that were matched above the specified confidence cutoff to the reference spectrum peaks are labelled with the labels of their corresponding reference peak
- #### log.txt - Optional log file that catches terminal output




## Integration into Your Own Python Scripts

In your script you can import a function that performs a peak matching operation with

```python
from pHarmony import MatchPeaks
```

### This function takes the following arguments

#### reference_peak_positions: torch.Tensor 
- this should be of shape (number of reference peaks, number of dims, 2)
- data placed in slice [:,:,0] are the chemical shifts for the corresponding peak and dimension

- data placed in slice [:,:,1] are the uncertainty in chemical shift for the corresponding peak and dimension

#### target_peak_positions: torch.Tensor 
- this should be of shape (number of target peaks, number of dims, 2) 
- data placed in slice [:,:,0] are the chemical shifts for the corresponding peak and dimension 
- data placed in slice [:,:,1] are the uncertainty in chemical shift for the corresponding peak and dimension

#### expected_fraction_csp: float 
- this value should be in the range (0,1) and is the prior belief of what fraction of matched peak pairs were perturbed by a CSP. (0.1 is a reasonable value, values near 0.5 or more could cause problems)

variance_scale_fraction_csp: float - this value should be positive and reflects how strongly the fraction csp prior
influences the final result (2.0 is a reasonable value, larger values decrease the prior's influence, larger values increase it)

#### max_predicted_dnm: float 
- this value is the point at which the matching and non-matching likelihood functions should be set as equal 
- It is generally calculated by taking a reasonable maximum expected chemical shit (say 0.1 ppm in proton) 
dividing by the average or smallest chemical shift uncertainty (in proton say 0.0015-0.003) and squaring the result

#### gradient_convergence: float 
- this is the value that determines whether parameters converged during the maximization 
gradient norms must be below this number (1E-6 is a reasonable value)

### Returns - tuple: (sampler: MMSampler, matching_probabilities: torch.tensor, normalized_distance_squared_matrix: torch.tensor)

#### sampler: MMSampler
- This is an MMSampler (custom matching matrix sampler) object, parameterized with the EM optimized distributions
- You can draw your own sample which is a a torch tensor of type int of shape (number of samples, number of reference peaks) 
- each entry corresponds to the index of the matched target column (-1 indicates no match)
- sample by calling
```python
my_sample = sampler.sample(number_of_samples) #where number_of_samples is an int
```


#### matching_probabilities: torch.tensor
- This is a tensor of type float of shape (number_of_reference_peaks, number_of_target_peaks)
- Each entry is the frequency of the corresponding match occuring in the final sample

#### normalized_distance_squared_matrix: torch.tensor
- This is a tensor of type float of shape (number_of_reference_peaks, number_of_target_peaks) 
that contains the normalized distances squared between all reference-target peak pairs
