"""Verify final saved result tables without rerunning expensive optimisers."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'

def check_table(folder, filename, expected):
    p = R / folder / filename
    df = pd.read_csv(p)
    assert len(df) == expected, (p, len(df), expected)
    assert not df.isna().any().any(), p
    return df

def main():
    a = check_table('experiment_7','all_results.csv',1600)
    assert (a.total_evals == 600).all()
    assert np.allclose(a.search_evals + a.validation_evals,600)
    assert np.allclose(a.selected_regret,a.oracle_regret+a.shortlist_loss+a.ranking_loss,atol=1e-6)
    assert len(a[a.K == 5]) == 960
    b = check_table('experiment_8a_corrected','results.csv',1440)
    assert (b.total_evals == 600).all()
    c = check_table('experiment_8b_corrected','results.csv',320)
    assert (c.total_evals == 600).all()
    d = check_table('experiment_8c','all_results.csv',120)
    assert (d.total_evals == 600).all()
    # Recheck corrected 8A population=8 matches all 480 corresponding E7 policies.
    ref = a[(a.K == 5) & (a.method == "CMA-ES")].copy()
    p8 = b[b.popsize == 8].copy()
    merged = p8.merge(ref,on=['function','noise','seed','fraction'],suffixes=('_8a','_7'),validate='one_to_one')
    assert len(merged) == 480
    assert np.allclose(merged.selected_regret_8a,merged.selected_regret_7,atol=1e-9)
    # Corrected 8B top_noisy matches E7 for matching function/noise/K/fraction.
    c0 = c[c.strategy == 'top_noisy']
    ref = a[(a.K == 5)&(np.isclose(a.noise,0.05))&a.fraction.isin([0.10,0.25])&a.function.isin(['Rosenbrock','Rastrigin'])]
    merged = c0.merge(ref,on=['function','method','seed','fraction'],suffixes=('_8b','_7'),validate='one_to_one')
    assert len(merged) == 160
    assert np.allclose(merged.selected_regret_8b,merged.selected_regret_7,atol=1e-9)
    print('PASS: counts, budgets, regret arithmetic, 8A-R/8B-R cross-experiment checks')

if __name__ == '__main__': main()
