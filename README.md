# DSA4212: Bayesian optimisation versus evolutionary strategies

A research study of **fixed-budget optimisation under noisy evaluations**. We compare a Gaussian-process Bayesian optimiser (BO) with CMA-ES, investigating how to allocate a fixed evaluation budget between exploration and repeated measurements for final candidate selection.

## Research question

When the total number of noisy objective evaluations is fixed, how should evaluations be divided between searching new points and re-evaluating shortlisted candidates? Does the answer depend on the optimiser, objective landscape, and noise level?

## Main experimental design

- Four 5-dimensional synthetic benchmarks: Sphere, Ackley, Rosenbrock, Rastrigin, bounded to [-5, 5]^5.
- Experiment 7: 600 objective evaluations per policy, 20 seeds, noise at 0% and 5% of each objective's global IQR, validation allocation 0%, 10%, 25%; shortlist sizes 5 and 10 (primary analysis K=5).
- Regret is computed using the noise-free benchmark, not the smallest noisy observation. Evaluation budgets include initial design, search, and repeated measurements. CPU wall-clock runtime is a separate metric.
- Follow-ups: corrected CMA-ES population sensitivity (8A-R), corrected spatial-diversity shortlist (8B-R), and intermediate 1%-IQR noise (8C, 10 seeds on Rosenbrock and Rastrigin).
- Noise scale means **global IQR**, not equal local signal-to-noise ratio. The noise standard deviation is supplied to BO; this is an assumption of the controlled benchmark.

## Repository guide

- `notebooks/`: original exploratory and main research notebooks. **Do not Run All:** the complete experiments can take hours and some cells require mounted Google Drive. The notebooks include superseded exploratory 8A and 8B; corrected 8A-R and 8B-R are authoritative.
- `src/experiment.py`: standalone reference implementation extracted from the original framework and Experiment 7 cells; no Google Drive dependency for its core functions.
- `results/`: authoritative CSV summaries and per-policy results for Experiments 7, 8A-R, 8B-R, 8C.
- `figures/`: final visualisations.
- `configs/experiment7.json`: original main-study configuration including library versions and noise scales.
- `scripts/check_results.py`: verifies saved-data consistency and alignment of corrected baselines.
- `tests/test_smoke.py`: quick CPU reproducibility/budget tests.
- `report/`: final report source and bibliography **to be added**.
## Installation and reproducibility

### Environment

The project was developed and tested using Python 3.12 in a Linux-based GitHub Codespaces environment. The optimisation experiments were originally executed on Google Colab using CPU resources.

Python 3.12 is recommended for reproducing the project. Python 3.14 was not used for the successful verification because some pinned scientific dependencies were incompatible with that version.

### Installation

From the repository root, create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

If the environment does not provide `ensurepip`, create the environment without pip and install pip separately:

```bash
python3.12 -m venv --without-pip .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
python -m pip install -r requirements.txt
```

### Verification

Run the saved-result consistency checks:

```bash
python scripts/check_results.py
```

Run the reproducibility smoke tests:

```bash
python -m pytest -q tests/
```

Both verification commands completed successfully in GitHub Codespaces using Python 3.12. The automated test suite reported **3 passed**.

These checks validate saved-result consistency and test a small part of the experimental implementation. They do not rerun the full optimisation study or constitute an independent reproduction of every saved trajectory.

The original experiment notebooks and saved results are provided in `notebooks/` and `results/`, respectively.

## Quick setup and audit (no optimisation rerun)

Python 3.13 was used in the main study. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/check_results.py
python -m pytest -q tests/
```

The checks use saved CSVs and a tiny CMA-ES smoke test; they should complete quickly. They **do not reproduce the full 320 search trajectories**. Run from the repository root so Python can find `src`.

## Re-running a small optimiser search

```python
from src.experiment import E6Evaluator, e6_run_cma, sphere
import numpy as np

ev = E6Evaluator(sphere, sigma=0.1, seed=100042, budget=18)
e6_run_cma(ev, seed=42, search_budget=18)
assert ev.calls == 18
print(min(ev.y))
```

To regenerate the full main study, consult Experiment 7's setup/execution/analysis cells in `notebooks/main_experiments.ipynb`. Those cells save search trajectories and per-policy CSVs to Google Drive. The expensive raw `.npz` search trajectories are **not included in this Git repository**; the saved result tables are. The manifest lists the original trajectories but does not itself provide their bytes. Exact full-trajectory reproduction also depends on the environment and pinned library versions.

## Findings (interpret cautiously)

- On noisy Rosenbrock, allocating evaluations to validation frequently reduced final recommendation regret, revealing substantial candidate-selection error.
- On Rastrigin, substantial validation often reduced performance because it displaced exploration.
- Corrected population-size sensitivity found landscape-dependent effects; default CMA-ES population 8 was not uniformly optimal.
- The corrected diversity-shortlist comparison did not establish a consistent advantage for spatial diversity; paired uncertainty intervals crossed zero.
- The 1%-IQR follow-up suggests Rosenbrock selection problems can persist even with reduced global noise.

These are benchmark- and implementation-specific findings, not universal algorithm rankings. Experiment 8C uses different seeds from Experiment 7, so cross-noise comparisons are not paired. BO and CMA-ES use the common initial design differently; BO's noise variance is specified rather than estimated. Configuration choices were explored unevenly. Wall-clock costs differ dramatically despite equal objective-call budgets. Bootstrap intervals are exploratory and not adjusted for multiple testing.


