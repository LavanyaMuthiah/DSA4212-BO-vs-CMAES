"""Reference implementation extracted from the experiment notebooks.

The BO/CMA-ES routines and validation policy are preserved verbatim from
notebook cells 6 and 16; no tuning or scientific algorithm changes.
"""
from dataclasses import dataclass
import numpy as np
import cma
import time
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

@dataclass(frozen=True)
class Config:
    d: int = 5
    budget: int = 200
    n_initial: int = 10
    low: float = -5.0
    high: float = 5.0
    acquisition_candidates: int = 512

CFG = Config()


def sphere(x):
    return float(np.sum(x**2))

def ackley(x):
    return float(
        -20*np.exp(-0.2*np.sqrt(np.mean(x**2)))
        -np.exp(np.mean(np.cos(2*np.pi*x)))
        +20+np.e
    )

def rosenbrock(x):
    return float(np.sum(
        100*(x[1:]-x[:-1]**2)**2
        +(1-x[:-1])**2
    ))

def rastrigin(x):
    return float(
        10*len(x)+np.sum(
            x**2-10*np.cos(2*np.pi*x)
        )
    )

def initial_design(seed):
    sampler = qmc.LatinHypercube(
        d=CFG.d, seed=seed
    )
    return qmc.scale(
        sampler.random(CFG.n_initial),
        CFG.low, CFG.high
    )

FUNCTIONS = {"Sphere": sphere, "Ackley": ackley, "Rosenbrock": rosenbrock, "Rastrigin": rastrigin}

SCALES = {'Sphere': 23.121333413258117, 'Ackley': 1.9780023830366442, 'Rosenbrock': 50776.29120599353, 'Rastrigin': 31.85032255297243}

class E6Evaluator:

    def __init__(self, func, sigma, seed, budget):
        self._func = func
        self.sigma = float(sigma)
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)

        self.X = []
        self.y = []
        self.phase = []
        self.calls = 0

    def evaluate(self, x, phase="search"):

        if self.calls >= self.budget:
            raise RuntimeError("Budget exceeded")

        x = np.asarray(x, dtype=float).copy()

        assert x.shape == (CFG.d,)
        assert np.all(np.isfinite(x))
        assert np.all(x >= CFG.low - 1e-8)
        assert np.all(x <= CFG.high + 1e-8)
        assert phase in ["search", "validation"]

        # True objective is used only inside evaluator.
        true_y = self._func(x)

        observed_y = (
            true_y + self.rng.normal(0, self.sigma)
        )

        self.X.append(x)
        self.y.append(float(observed_y))
        self.phase.append(phase)
        self.calls += 1

        return float(observed_y)

def e6_initialise(ev, seed):
    X0 = initial_design(seed)

    y0 = [
        ev.evaluate(x, "search")
        for x in X0
    ]

    return np.asarray(X0), np.asarray(y0)

def e6_gp(X, y, sigma, kernel=None, tune=False):

    Xs = (X - CFG.low) / (CFG.high - CFG.low)

    ym = float(np.mean(y))
    ys = max(float(np.std(y)), 1e-8)

    yn = (y - ym) / ys

    if kernel is None:
        kernel = (
            ConstantKernel(1.0, (0.01, 100.0))
            * Matern(
                length_scale=np.ones(CFG.d) * 0.3,
                length_scale_bounds=(0.03, 3.0),
                nu=2.5
            )
        )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=max((sigma / ys)**2, 1e-8),
        normalize_y=False,
        optimizer="fmin_l_bfgs_b" if tune else None,
        n_restarts_optimizer=1 if tune else 0,
        random_state=42
    )

    gp.fit(Xs, yn)

    return gp

def e6_run_bo(ev, seed, search_budget):

    rng = np.random.default_rng(seed + 7000)

    e6_initialise(ev, seed)

    learned_kernel = None

    while ev.calls < search_budget:

        X = np.asarray(ev.X)
        y = np.asarray(ev.y)

        tune = (
            learned_kernel is None
            or ev.calls % 25 == 0
        )

        gp = e6_gp(
            X, y, ev.sigma,
            kernel=None if tune else learned_kernel,
            tune=tune
        )

        if tune:
            learned_kernel = gp.kernel_

        candidates = rng.uniform(
            0, 1, (1024, CFG.d)
        )

        mu, std = gp.predict(
            candidates, return_std=True
        )

        scores = mu - 2.0 * std

        z_next = candidates[np.argmin(scores)]

        # Local acquisition refinement
        def acquisition(z):
            m, s = gp.predict(
                np.asarray(z).reshape(1, -1),
                return_std=True
            )
            return float(m[0] - 2.0*s[0])

        result = minimize(
            acquisition,
            z_next,
            method="L-BFGS-B",
            bounds=[(0, 1)] * CFG.d,
            options={"maxiter": 30}
        )

        if (
            result.success
            and result.fun < np.min(scores)
        ):
            z_next = result.x

        x_next = (
            CFG.low +
            z_next * (CFG.high - CFG.low)
        )

        ev.evaluate(x_next, "search")

    assert ev.calls == search_budget

def e6_run_cma(ev, seed, search_budget):

    X0, y0 = e6_initialise(ev, seed)

    best_x = X0[np.argmin(y0)].copy()
    best_y = float(np.min(y0))

    def make_es(center, es_seed):
        return cma.CMAEvolutionStrategy(
            np.clip(center, CFG.low, CFG.high),
            2.0,
            {
                "seed": int(es_seed),
                "bounds": [CFG.low, CFG.high],
                "popsize": 8,
                "verbose": -9,
                "verb_disp": 0,
                "tolfun": 0,
                "tolx": 1e-14,
                "tolstagnation": 100000
            }
        )

    es = make_es(best_x, seed)

    while ev.calls < search_budget:

        candidates = es.ask()
        remaining = search_budget - ev.calls
        batch = candidates[:remaining]

        values = []

        for x in batch:
            value = ev.evaluate(x, "search")
            values.append(value)

            if value < best_y:
                best_y = value
                best_x = np.asarray(x).copy()

        if len(batch) == len(candidates):
            es.tell(candidates, values)
        else:
            # Final incomplete generation is counted.
            break

        if es.stop():
            es = make_es(
                best_x,
                seed + ev.calls + 1
            )

    assert ev.calls == search_budget

B7 = 600
F7 = ["Sphere", "Ackley", "Rosenbrock", "Rastrigin"]
M7 = ["BO", "CMA-ES"]
N7 = [0.05, 0.0]
SEEDS7 = [42, 123] + list(range(1000, 1018))
FRACS7 = [0.0, 0.10, 0.25]
K_VALUES7 = [5, 10]
CONFIG_HASH7 = '3b6a63d8eeb194fa'

def run_search7(fname, method, noise, seed):

    sigma = float(noise * SCALES[fname])

    ev = E6Evaluator(
        FUNCTIONS[fname],
        sigma,
        seed + 100000,
        B7
    )

    start = time.perf_counter()

    if method == "BO":
        e6_run_bo(ev, seed, B7)
    else:
        e6_run_cma(ev, seed, B7)

    elapsed = time.perf_counter() - start

    X = np.asarray(ev.X)
    y = np.asarray(ev.y)

    assert ev.calls == B7
    assert X.shape == (B7, CFG.d)
    assert y.shape == (B7,)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.all(X >= CFG.low-1e-8)
    assert np.all(X <= CFG.high+1e-8)

    return X, y, elapsed

def noise7(fname, method, noise, seed, idx, rep):

    sequence = np.random.SeedSequence([
        7007,
        F7.index(fname),
        M7.index(method),
        int(round(noise * 100000)),
        int(seed),
        int(idx),
        int(rep)
    ])

    rng = np.random.default_rng(sequence)

    return float(
        rng.normal(0, noise * SCALES[fname])
    )

def validate7(X, y, fname, method, noise,
              seed, fraction, K):

    n_val = int(round(B7 * fraction))
    n_search = B7 - n_val

    Xs = X[:n_search]
    ys = y[:n_search]

    # All shortlist decisions use noisy data only.
    shortlist = np.argsort(
        ys, kind="stable"
    )[:K]

    sums = ys[shortlist].copy()
    counts = np.ones(K, dtype=int)

    func = FUNCTIONS[fname]

    for j in range(n_val):

        slot = j % K
        idx = int(shortlist[slot])
        repeat_number = int(counts[slot])

        # Fresh independent noisy evaluation.
        measurement = (
            func(Xs[idx])
            + noise7(
                fname, method, noise,
                seed, idx, repeat_number
            )
        )

        sums[slot] += measurement
        counts[slot] += 1

    selected_slot = int(np.argmin(sums/counts))
    selected_idx = int(shortlist[selected_slot])

    # Independent offline scoring.
    true_values = np.array([
        func(x) for x in Xs
    ])

    oracle = float(np.min(true_values))
    best_shortlist = float(
        np.min(true_values[shortlist])
    )
    selected = float(true_values[selected_idx])

    shortlist_loss = best_shortlist - oracle
    ranking_loss = selected - best_shortlist

    assert np.isclose(
        selected,
        oracle + shortlist_loss + ranking_loss,
        atol=1e-6
    )
    assert shortlist_loss >= -1e-6
    assert ranking_loss >= -1e-6
    assert n_search + n_val == B7

    return {
        "function": fname,
        "method": method,
        "noise": noise,
        "sigma": noise * SCALES[fname],
        "seed": seed,
        "budget": B7,
        "fraction": fraction,
        "K": K,
        "search_evals": n_search,
        "validation_evals": n_val,
        "total_evals": B7,
        "selected_idx": selected_idx,
        "selected_regret": selected,
        "oracle_regret": oracle,
        "shortlist_loss": shortlist_loss,
        "ranking_loss": ranking_loss,
        "selection_gap": selected - oracle,
        "shortlist_coverage": int(
            np.argmin(true_values) in shortlist
        ),
        "config_hash": CONFIG_HASH7
    }
