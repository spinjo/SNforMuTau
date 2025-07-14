import numpy as np
import scipy.integrate as itg
import scipy.special as sp

daMu_exp = 251e-11
daMu_exp_unc = 59e-11
mmu = 105.658


def daMu(g, mZp):  # not needed
    fac, _ = itg.quad(
        lambda x: mmu**2
        * x
        * (1 - x) ** 2
        / (mmu**2 * (1 - x) ** 2 + mZp**2 * x),
        0,
        1,
    )
    return g**2 / (4 * np.pi**2) * fac


def getCoupling(mZp, daMuVal):
    fac, _ = itg.quad(
        lambda x: mmu**2
        * x
        * (1 - x) ** 2
        / (mmu**2 * (1 - x) ** 2 + mZp**2 * x),
        0,
        1,
    )
    return (4 * np.pi**2 * daMuVal / fac) ** 0.5


def getNsigma(mZp, nSigma):  # checked with M3 bounds (but not perfectly)
    p = sp.erf(nSigma / 2**0.5)
    ddaMu = 2**0.5 * daMu_exp_unc * sp.erfinv(p)
    gLower, gUpper = np.zeros((2, len(mZp)))
    for i in range(len(mZp)):
        gUpper[i] = getCoupling(mZp[i], daMu_exp + ddaMu)
        gLower[i] = getCoupling(mZp[i], daMu_exp - ddaMu)
    return gLower, gUpper


def test():
    dx = 251
    sig = 59
    p = sp.erf(dx / (2**0.5 * sig))
    print(p)
    print(2**0.5 * sp.erfinv(p))  # works


# print(getNsigma(5, 2))
