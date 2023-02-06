import numpy as np
import bayes_kit as bk
import pytest as pt

def sample_ar1(rho, N):
    z = np.random.normal(size = N)
    for n in range(1, N):
        z[n] += rho * z[n - 1]
    return z
        
def integrated_autocorr_time_ar1(rho):
    last_sum = 0
    sum = 1
    t = 1
    while sum != last_sum:
        last_sum = sum
        sum += 2 * rho**t
        t += 1
    return sum

def expected_ess_ar1(rho, N):
    return N / integrated_autocorr_time_ar1(rho)
    
def run_ess_test_ar1(rho, N):
    v = sample_ar1(rho, N)
    E_ess = expected_ess_ar1(rho, N)

    hat_ess1 = bk.ess(v)
    np.testing.assert_allclose(E_ess, hat_ess1, atol=N, rtol=0.1)

    hat_ess2 = bk.ess_imse(v)
    np.testing.assert_allclose(E_ess, hat_ess2, atol=N, rtol=0.1)

    hat_ess3 = bk.ess_ipse(v)
    np.testing.assert_allclose(E_ess, hat_ess3, atol=N, rtol=0.1)

def test_ess_ar1():
    for rho in np.arange(-0.1, 0.6, step = 0.1):
        run_ess_test_ar1(rho, 10_000)

def test_ess_independent():
    N = 10000
    y = np.random.normal(size=N)
    hat_ess = bk.ess(y)
    E_ess = N
    np.testing.assert_allclose(E_ess, hat_ess, atol=1, rtol=0.1)
    
def test_ess_exceptions():
    for n in range(4):
        v = sample_ar1(0.5, n)
        with pt.raises(ValueError):
            bk.ess(v)
        with pt.raises(ValueError):
            bk.ess_imse(v)
        with pt.raises(ValueError):
            bk.ess_ipse(v)
