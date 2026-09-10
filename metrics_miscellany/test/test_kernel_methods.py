import numpy as np
import pandas as pd
import pytest

from metrics_miscellany import kernel_methods as km


@pytest.fixture
def smooth_data():
    rng = np.random.default_rng(0)
    n = 200
    x = pd.Series(np.linspace(-3, 3, n), name='x')
    sigma = 0.1
    y = pd.Series(np.sin(x) + sigma * rng.standard_normal(n), name='y')
    return x, y, sigma


def test_kernel_regression_recovers_smooth_target(smooth_data):
    """Kernel regression on y = sin(x) + small noise should be close
    to sin at interior points for a moderate bandwidth."""
    x, y, _ = smooth_data
    mhat = km.kernel_regression(x, y, bw=0.3)
    for x0 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        assert abs(float(mhat(x0)) - np.sin(x0)) < 0.1


def test_kernel_regression_variance_recovers_homoskedastic_noise(smooth_data):
    """E[(y - m(x))^2 | x] should track the constant noise variance
    sigma^2 at interior points for a moderate bandwidth.  The
    previous implementation -- which used the inclusive K for the
    residuals and a constant denominator -- gave estimates that
    looked like the *training-set mean residual* scaled by an
    irrelevant constant, which is what this test catches."""
    x, y, sigma = smooth_data
    sigmahat = km.kernel_regression_variance(x, y, bw=0.3)
    for x0 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        est = float(sigmahat(x0))
        # Match sigma^2 to within a factor of ~3 -- generous to allow
        # for bias from kernel smoothing.
        assert sigma ** 2 / 3 < est < 3 * sigma ** 2, (
            f"sigmahat({x0}) = {est:.4f}, expected ~{sigma**2:.4f}")


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 200
    x = pd.Series(np.linspace(-3, 3, n), name='x')
    sigma = 0.1
    y = pd.Series(np.sin(x) + sigma * rng.standard_normal(n), name='y')
    test_kernel_regression_recovers_smooth_target((x, y, sigma))
    test_kernel_regression_variance_recovers_homoskedastic_noise(
        (x, y, sigma))
