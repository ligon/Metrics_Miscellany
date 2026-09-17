from scipy.stats.distributions import binom


def quantile_confidence_intervals(x, q=1 / 2, minimum_coverage=0.95):
    """Quantile of x, with a distribution-free confidence interval.

    Follows William Huber's construction; see
    https://stats.stackexchange.com/questions/122001/confidence-intervals-for-median
    """
    α = 1 - minimum_coverage
    x = x.sort_values()
    n = x.shape[0]
    p = binom(n, q)
    u, cu = [
        (i, p.cdf(i))
        for i in range(1, n)
        if p.cdf(i) > 1 - α / 2 and p.cdf(i - 1) <= 1 - α / 2
    ][0]
    l, cl = [
        (i - 1, p.cdf(i - 1))
        for i in range(1, n)
        if p.cdf(i) > α / 2 and p.cdf(i - 1) <= α / 2
    ][0]

    coverage = 1 - cl - (1 - cu)
    return (x.iloc[l], x.iloc[u]), coverage
