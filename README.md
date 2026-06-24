# pHarmony

pHarmony matches multidimensional NMR peak lists and estimates posterior match probabilities. It provides:

- a `pHarmony` command-line tool for matching Sparky-style peak lists
- a Python API centered on `pHarmony.MatchPeaks`

## Requirements

- Python `>=3.9`
- `numpy>=1.24.0,<2`
- `pandas>=2.0`
- `torch>=2.0`
- `matplotlib>=3.2`
- `scipy>=1.10`
- `scikit-learn>=1.6.1`
- `tqdm>=4.67.1`

## Installation

Create and activate an environment, then install the package.

```bash
conda create --name pHarmony python=3.9
conda activate pHarmony
pip install git+https://github.com/anthony-bishop-tamu/pHarmony.git@main
```

For development from a local checkout:

```bash
pip install -e .
```

## Verify Installation

With the environment active, run:

```bash
pHarmony --help
```

You should see the command-line options for the standalone peak matching workflow.

## Command-Line Usage

Example 2D matching run:

```bash
pHarmony \
  --reference_peak_list reference.list \
  --reference_cs_column_names w1 w2 \
  --reference_peak_list_error 0.015 0.0015 \
  --target_peak_list target.list \
  --target_cs_column_names w1 w2 \
  --target_peak_list_error 0.015 0.0015 \
  --CSP_scaling_factors 0.101 1.0 \
  --output_directory peak_matcher_output
```

The reference and target peak lists are read as whitespace-delimited tables. Output transfer files expect an `Assignment` column, and the chemical shift columns are selected with `--reference_cs_column_names` and `--target_cs_column_names`.

### Required Arguments

- `--reference_peak_list`: reference peak list filename
- `--reference_cs_column_names`: chemical shift column names in the reference list
- `--reference_peak_list_error`: uncertainty for each reference dimension
- `--target_peak_list`: target peak list filename
- `--target_cs_column_names`: chemical shift column names in the target list
- `--target_peak_list_error`: uncertainty for each target dimension
- `--CSP_scaling_factors`: scaling factors for CSP calculation, one per matched dimension

The number and order of reference dimensions, target dimensions, errors, and CSP scaling factors must match.

### Optional Arguments

- `--output_directory`: directory for output files; default `./peak_matcher_output`
- `--confidence_cutoff`: minimum posterior matching probability for high-confidence outputs; default `0.95`
- `--expected_fraction_csp`: prior estimate for the fraction of matched peaks with a chemical shift perturbation; default `0.1`
- `--variance_scale_fraction_csp`: scale factor used to set the Beta prior variance for CSP mixture weights; default `3.0`
- `--expected_max_csp`: expected maximum CSP in ppm, in units of the lowest-uncertainty/scaled dimension; default `0.1`
- `--log_file`: write terminal logging to `log.txt` in the output directory

## Output Files

Each run writes these files to the output directory:

- `Match_probabilities.csv`: posterior matching probability matrix for reference-target peak pairs
- `CSP_probabilities.csv`: posterior probability that a matched pair has a chemical shift perturbation
- `<reference>_<target>_fittedDistributions.png`: fitted matching, non-matching, and CSP distance distributions
- `<reference>_<target>_transferred.csv`: best target match for each reference peak, including match probability, missing probability, normalized squared distance, and CSP
- `<reference>_<target>_transferred_HC.csv`: high-confidence rows from the transferred table using `--confidence_cutoff`
- `<reference>_<target>_transferred.list`: target peak list with high-confidence assignments transferred from the reference list
- `<reference>_<target>_transferred_matching_only.list`: high-confidence matched peak list containing transferred assignments and target coordinates only
- `log.txt`: optional log file when `--log_file` is supplied

Rows and columns in `Match_probabilities.csv` may not sum to `1.0` because reference and target peaks can be sampled as unmatched.

## Python API

Import the matcher:

```python
from pHarmony import MatchPeaks
```

Call signature:

```python
sampler, matching_probabilities, distances_squared_normalized = MatchPeaks(
    reference_peak_positions,
    target_peak_positions,
    expected_fraction_csp,
    variance_scale_fraction_csp,
    max_predicted_dnm,
)
```

### Inputs

`reference_peak_positions` and `target_peak_positions` are `torch.Tensor` objects with shape:

```text
(number_of_peaks, number_of_dimensions, 2)
```

For each peak and dimension:

- `[..., 0]` is the chemical shift
- `[..., 1]` is the chemical shift uncertainty

Other parameters:

- `expected_fraction_csp`: prior estimate in the range `(0, 1)` for the fraction of matched peak pairs with a CSP
- `variance_scale_fraction_csp`: positive scale factor for the CSP mixture-weight prior variance
- `max_predicted_dnm`: normalized squared distance where matching and non-matching likelihoods are set to be comparable

You can calculate `max_predicted_dnm` from an expected CSP with:

```python
from pHarmony import calculateMaxD2FromCSP
```

### Returns

- `sampler`: an `MMSampler` parameterized with the optimized posterior distributions
- `matching_probabilities`: tensor of shape `(number_of_reference_peaks, number_of_target_peaks)`
- `distances_squared_normalized`: tensor of normalized squared distances for every reference-target pair

Draw assignment samples from the sampler with:

```python
sample = sampler.sample(number_of_samples)
```

The returned sample has shape `(number_of_samples, number_of_reference_peaks)`. Each value is the matched target peak index, and `-1` indicates no match.
