"""Short deterministic CPU smoke test; no Colab/Drive required."""
import numpy as np
from src import experiment as ex

def test_objectives():
    x=np.zeros(ex.CFG.d)
    for name in ['Sphere','Ackley','Rastrigin']:
        assert abs(ex.FUNCTIONS[name](x)) < 1e-10
    assert ex.FUNCTIONS['Rosenbrock'](np.ones(ex.CFG.d)) == 0.0

def test_cma_budget_and_reproducibility():
    def run():
        ev=ex.E6Evaluator(ex.sphere,0.1,100042,18)
        ex.e6_run_cma(ev,42,18)
        return np.asarray(ev.X),np.asarray(ev.y),ev.calls
    a,b,n=run(); c,d,m=run()
    assert n==m==18
    np.testing.assert_array_equal(a,c)
    np.testing.assert_array_equal(b,d)

def test_noise_independent_of_call_order():
    a=ex.noise7('Sphere','BO',0.05,42,11,1)
    b=ex.noise7('Sphere','BO',0.05,42,11,1)
    c=ex.noise7('Sphere','BO',0.05,42,11,2)
    assert a==b and a!=c
