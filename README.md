# DSA4212: Bayesian Optimisation versus Evolutionary Strategies

A research study of **fixed-budget optimisation under noisy evaluations**, comparing Gaussian-process Bayesian optimisation (BO) with the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

The project investigates how a limited evaluation budget should be allocated between exploring new candidate solutions and re-evaluating promising candidates before making a final recommendation.

## Research Question

When the total number of noisy objective evaluations is fixed, how should evaluations be divided between searching new points and re-evaluating shortlisted candidates? Does the optimal allocation depend on the optimisation method, objective landscape, and noise level?

## Experimental Design

Data preparation: No external dataset is required. Objective values are generated directly from the four synthetic benchmark functions. Initial candidate locations are generated using seeded Latin hypercube sampling, and independent Gaussian observation noise is added during evaluation. The benchmark definitions, fixed noise scales, seeds and experimental settings are provided in the source code and configuration files.

The main study (Experiment 7) uses:

- **Benchmark functions:** Sphere, Ackley, Rosenbrock and Rastrigin.
- **Dimension:** Five dimensions, with search domain \([-5,5]^5\).
- **Evaluation budget:** 600 noisy objective evaluations per policy.
- **Noise levels:** 0% and 5% of each objective's global interquartile range (IQR).
- **Validation allocations:** 0%, 10% and 25% of the total evaluation budget.
- **Random seeds:** 20 independent seeds per experimental condition.
- **Candidate shortlists:** Sizes 5 and 10, with \(K=5\) used for the primary analysis.
- **Performance metric:** Noise-free simple regret of the final recommended candidate, evaluated against the known optimum.

The evaluation budget includes initial design evaluations, search evaluations and repeated measurements used for final validation. Computational overhead is measured separately using CPU wall-clock time.

Noise is additive Gaussian, scaled using each benchmark's global objective IQR. This scaling does not imply identical local signal-to-noise ratios across functions.

The BO implementation is supplied with the observation-noise variance, representing a controlled modelling assumption rather than an estimated quantity.

### Follow-up Experiments

The main study was followed by targeted investigations motivated by the observed results:

| Experiment | Purpose |
|---|---|
| 8A-R | Corrected CMA-ES population-size sensitivity, testing whether algorithm configuration changes the comparison |
| 8B-R | Corrected shortlist-diversity comparison, investigating whether spatial diversity improves final candidate selection |
| 8C | Intermediate noise sensitivity at 1% IQR, using 10 seeds on Rosenbrock and Rastrigin |

Earlier exploratory experiments investigated evaluation-budget scaling, BO configuration, noisy candidate selection, validation allocation and shortlist size.

These preliminary studies informed the final experimental design. The corrected 8A-R and 8B-R results supersede their original exploratory versions.

## Repository Structure

```text
.
├── README.md
├── authors.md
├── requirements.txt
├── configs/
│   └── experiment7.json
├── src/
│   └── experiment.py
├── notebooks/
│   ├── preliminary_experiments.ipynb
│   └── main_experiments.ipynb
├── scripts/
│   └── check_results.py
├── tests/
│   └── test_smoke.py
├── results/
│   ├── experiment_7/
│   ├── experiment_8a_corrected/
│   ├── experiment_8b_corrected/
│   └── experiment_8c/
├── figures/
└── report/
    ├── report.tex
    ├── report.pdf
    ├── references.bib
    └── figures/
```

### Directory Guide

- **`notebooks/`** — Original exploratory and main research notebooks, including experimental development and follow-up investigations. Some cells depend on mounted Google Drive storage. Avoid using *Run All*, as the complete experiments can take several hours.
- **`src/experiment.py`** — Standalone reference implementation of the optimisation framework, extracted from the original experimental code.
- **`configs/`** — Main-study configuration, including experimental parameters, library versions and noise scales.
- **`results/`** — Saved numerical results and per-policy summaries from the authoritative main and corrected follow-up experiments.
- **`figures/`** — Visualisations generated from the experimental results.
- **`scripts/check_results.py`** — Saved-result consistency checks, including evaluation accounting and corrected cross-experiment comparisons.
- **`tests/test_smoke.py`** — Lightweight CPU tests for reproducibility and budget enforcement.
- **`report/`** — Final submitted PDF report, complete LaTeX source, bibliography and required figures.
- **`authors.md`** — Author information and student number.

## Installation and Reproducibility

### Environment

The optimisation experiments were originally executed using **Google Colab with CPU resources**.

Repository verification was successfully performed using **Python 3.12 in a Linux-based GitHub Codespaces environment**.

Python 3.12 is recommended for reproducing the repository's verification checks because some pinned scientific dependencies were incompatible with Python 3.14.

The original experimental library versions and configuration details are recorded in `configs/experiment7.json`.

### Installation

From the repository root, create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

If the environment does not provide `ensurepip`, create the environment without pip and install it separately:

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

Run the automated smoke tests:

```bash
python -m pytest -q tests/
```

Both commands completed successfully in GitHub Codespaces using Python 3.12.

The smoke-test suite reported:

```text
3 passed in 11.50s
```

These checks verify saved-result consistency and selected properties of the optimisation implementation. They do **not** independently reproduce every optimisation trajectory or rerun the full experimental study.

### Re-running a Small Optimisation Test

A small CMA-ES optimisation can be executed without Google Drive:

```python
from src.experiment import E6Evaluator, e6_run_cma, sphere

ev = E6Evaluator(
    sphere,
    sigma=0.1,
    seed=100042,
    budget=18
)

e6_run_cma(ev, seed=42, search_budget=18)

assert ev.calls == 18
print(min(ev.y))
```

Run this example from the repository root.

### Reproducing the Full Study

The original Experiment 7 setup, execution and analysis cells are available in `notebooks/main_experiments.ipynb`.

The notebooks contain the experimental framework used to generate search trajectories and policy-level results, with outputs saved to Google Drive during execution.

### Google Colab Setup for Full Reproduction

To regenerate the full experimental trajectories, open `notebooks/main_experiments.ipynb` in Google Colab using a CPU runtime.

Mount Google Drive before running the experiment cells:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create the output directory if it does not already exist:

```python
from pathlib import Path
Path('/content/drive/MyDrive/DSA4212_Final_Project').mkdir(
    parents=True, exist_ok=True
)
```

Install the dependencies and run the notebook's setup cells before executing the required experiments in order. The notebook uses Google Drive to save intermediate trajectories and results; ensure that its output paths match the mounted directory.

The full study is computationally expensive and may take several hours on a CPU runtime. The saved CSV results and verification scripts allow the reported findings to be inspected without rerunning every trajectory.

The complete main study consists of **320 search trajectories**, followed by policy-level analysis.

The expensive raw `.npz` search trajectories are not included in this repository. Instead, the repository provides saved result tables, experimental configurations, source code and consistency checks.

The trajectory manifest documents the original runs but does not contain the raw trajectory data.

Exact full-study reproduction requires executing the original experimental framework with the documented configuration and compatible library versions. Runtime and numerical behaviour may also depend on the computing environment.

## Main Findings

### 1. Validation is not universally beneficial

On noisy Rosenbrock, allocating part of the evaluation budget to repeated measurements substantially reduced final recommendation regret.

However, on Rastrigin, larger validation allocations often worsened performance because fewer evaluations remained available for exploration.

This suggests that validation is most useful when noisy candidate selection is a major source of error, rather than when discovering promising candidates remains the primary difficulty.

### 2. Search quality and final recommendation quality are different

An optimiser may discover a strong candidate but fail to recommend it because noisy observations distort the estimated ranking of solutions.

The study distinguishes between:

- **Oracle search regret:** Quality of the best candidate encountered during search.
- **Shortlist loss:** Additional regret caused by excluding strong candidates from validation.
- **Ranking loss:** Additional regret caused by selecting a worse candidate from the shortlist.

This decomposition helps explain why repeated validation can improve final recommendations without improving the underlying search trajectory.

### 3. Algorithm configuration affects performance

The corrected CMA-ES population-size experiment found that larger populations substantially improved performance on noisy Rosenbrock, but the same configuration was not uniformly best across all benchmarks.

The comparison therefore depends partly on algorithm configuration, not only on the underlying optimisation method.

### 4. Shortlist diversity did not consistently improve selection

The corrected diversity-aware shortlist experiment did not establish a consistent advantage over selecting candidates based on noisy observations.

The paired uncertainty intervals crossed zero, indicating that the observed differences were not sufficiently conclusive to support a general improvement claim.

This negative result suggests that greater spatial diversity alone does not guarantee better final recommendations.

### 5. Intermediate noise revealed further selection difficulties

The 1%-IQR noise experiment showed that substantial final-selection error can persist on Rosenbrock even when global observation noise is relatively small.

The effectiveness of validation depends on the interaction between noise, objective geometry, search quality and recommendation policy.

## Interpretation and Limitations

The experiments suggest that **the appropriate allocation between exploration and validation depends on the dominant source of final regret**.

When the optimiser encounters promising solutions but cannot reliably distinguish them under noise, repeated validation can be valuable.

When the main challenge is locating good solutions, allocating more evaluations to exploration may be preferable.

The conclusions are specific to the tested benchmarks, implementations and experimental conditions.

Important limitations include:

- The experiments use four five-dimensional synthetic benchmarks rather than real-world objective functions.
- The noise model is additive, independent Gaussian noise with a specified standard deviation. Heteroskedastic and correlated noise were not investigated.
- Noise is scaled using global objective IQR, which does not ensure comparable local signal-to-noise ratios.
- BO is supplied with the observation-noise variance rather than estimating it.
- BO and CMA-ES use the common initial design differently, and configuration choices received unequal exploratory tuning effort.
- Evaluation budgets are equal, but CPU wall-clock costs differ substantially.
- Experiment 8C uses different random seeds from Experiment 7, so cross-noise comparisons are not paired.
- Bootstrap confidence intervals are exploratory and are not adjusted for multiple testing.
- The tested validation policies use fixed allocations rather than adaptively deciding when to stop searching.

The study does not establish a universally superior optimisation algorithm or validation allocation. Instead, it provides evidence about the conditions under which different choices can succeed or fail.

## Report and Submission

The final scientific report is available in:

- `report/report.pdf` — Submitted PDF report.
- `report/report.tex` — Complete LaTeX source.
- `report/references.bib` — Bibliography.
- `report/figures/` — Figures required to compile the report.

The report contains the research motivation, methodological justification, experimental development, results, follow-up investigations, interpretation, limitations, conclusions and AI-assistance declaration.

The PDF and repository should be used together to assess the study's findings and reproducibility. The report includes a more detailed AI-assistance declaration distinguishing the author's contributions from AI-supported activities.
