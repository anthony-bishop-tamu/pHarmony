# pHarmony

## Requirements

- **Python**: `>= 3.9`

### Python Dependencies

The following packages are required:

- `numpy>=1.24.0`
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

#### Option A: Install via SSH (recommended)

```bash
pip install git+ssh://git@github.com/anthony-bishop-tamu/pHarmony.git
```

#### Option B: Install via HTTPS (no SSH keys required)

```bash
pip install git+https://github.com/anthony-bishop-tamu/pHarmony.git
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

## Integration into Your Own Python Scripts

In your script you can import a function that performs a peak matching operation with

```python
from pHarmony import MatchPeaks
```

This function takes the following arguments

reference_peak_positions: torch.Tensor - this should be of shape (number of reference peaks, number of dims, 2)

data placed in slice [:,:,0] are the chemical shifts for the corresponding peak and dimension

data placed in slice [:,:,1] are the uncertainty in chemical shift for the corresponding peak and dimension

target_peak_positions: torch.Tensor - this should be of shape (number of target peaks, number of dims, 2)

data placed in slice [:,:,0] are the chemical shifts for the corresponding peak and dimension

data placed in slice [:,:,1] are the uncertainty in chemical shift for the corresponding peak and dimension

expected_fraction_csp: float - this value should be in the range (0,1) and is the prior belief of what fraction of matched peak 
pairs were perturbed by a CSP. (0.1 is a reasonable value, values near 0.5 or more could cause problems)

variance_scale_fraction_csp: float - this value should be positive and reflects how strongly the fraction csp prior
influences the final result (2.0 is a reasonable value, larger values decrease the prior's influence, larger values increase it)

max_predicted_dnm: float - this value is the point at which the matching and non-matching likelihood functions should be set as equal
It is generally calculated by taking a reasonable maximum expected chemical shit (say 0.1 ppm in proton) 
dividing by the average or smallest chemical shift uncertainty (in proton say 0.0015-0.003) and squaring the result

gradient_convergence: float - this is the value that determines whether parameters converged during the maximization g
radient norms must be below this number (1E-6 is a reasonable value)

