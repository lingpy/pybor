"""
Utility functions for the package.
"""
import math
from pathlib import Path

def test_data(*comps):
    return Path(__file__).parent.parent.parent.joinpath('tests', 'data',
            *comps).as_posix()


def find_cut_point(native, loan):
    """
    Find cut_point that will give highest accuracy.

    Parameters
    ----------
    native : [float]
        List of entropies from native distribution.
    loan : [float]
        List of entropies from loan distribution.

    Returns
    -------
    cut_point : float
        Critical value to test native versus loan word.
        Native if entropy < cut_point else loan.

    """
    total = sorted([-s for s in native] + loan, key=abs)

    # All values are > 0 are correct.
    cut_point = 0
    correct = len(loan)
    correct_max = correct

    for t in total:
        correct += (1 if t < 0 else -1)
        if correct > correct_max:
            correct_max = correct
            cut_point = t

    cut_point = abs(cut_point)
    return cut_point



def find_ref_limit(entropies=None, frac=0.99):
    """
    Calculate a cut-off point for entropies from the data.
    """
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution rather than Gaussian z value
    # as cut-point for discriminating between native and loan.
    entropies = sorted(entropies)
    idx = min((len(entropies)-1)*frac, len(entropies)-1)
    return (entropies[math.floor(idx)]+entropies[math.ceil(idx)])/2
