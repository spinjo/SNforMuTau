import numpy as np
import scipy.integrate as itg

Nnu = 3
TnuD = 2.3  # in MeV
gZp = 3
gChi = 4

NeffCritBound = 3.4
NeffCritH0 = 3.2

prec = 1000


def _F(x, stat):
    def statFac(y, stat):
        if stat == "boson":
            return np.exp(y) - 1
        elif stat == "fermion":
            return np.exp(y) + 1

    res, _ = itg.quad(
        lambda y: (4 * y**2 - x**2) * (y**2 - x**2) ** 0.5 / statFac(y, stat),
        x,
        np.inf,
    )
    return 30 / (7 * np.pi**4) * res


def _getNeff(mZp, mChi):
    return Nnu * (
        1
        + 1 / Nnu * gZp / 2 * _F(mZp / TnuD, "boson")
        + 1 / Nnu * gChi / 2 * _F(mChi / TnuD, "fermion")
    ) ** (4 / 3)


def getNeff(mZpmin, mZpmax, mChiOvermZp):
    mZp = np.exp(np.linspace(np.log(mZpmin), np.log(mZpmax), prec))
    mZpBound, mZpH0 = (0, 0)
    for i in range(prec):
        Neff = _getNeff(mZp[i], mChiOvermZp * mZp[i])
        # print(Neff, NeffCritBound, NeffCritH0)
        if Neff < NeffCritBound and mZpBound == 0:
            mZpBound = mZp[i]
        if Neff < NeffCritH0 and mZpH0 == 0:
            mZpH0 = mZp[i]

    return mZpBound, mZpH0  # always mZpBound < mZpH0


# print(getNeff(1, 100, 1/3))

print(_getNeff(40, 40 * 0.45))
print(_getNeff(20, 20 * 0.45))


def getmChi(mZp):
    prec = 100
    mChi = np.exp(np.linspace(np.log(1e-3), np.log(1e3), prec))
    mChiRet = None
    for i in range(prec):
        Neff = _getNeff(mZp, mChi[i])
        if Neff < 3.2:
            print(Neff)
            mChiRet = mChi[i]
            break
    return mChiRet


"""
Zmin=1
Zmax =1e3
prec=100
mZp = np.exp(np.linspace(np.log(Zmin), np.log(Zmax), prec))
mChi = np.zeros(prec)
for i in range(prec):
    mChi[i] = getmChi(mZp[i])
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax=fig.add_subplot(1,1,1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("mZp [MeV]")
ax.set_ylabel("mChi [MeV]")
ax.plot(mZp, mChi)
plt.show()
"""
