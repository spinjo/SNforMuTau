import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import matplotlib as mpl

import os, sys, math

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../")
import helper
import helper_plot as hp
import calcMuTau as calc

import warnings

warnings.filterwarnings("ignore")

col = mpl.cm.Set1(np.linspace(0, 1, 9))[0:9]
mChi = 0.0


def main():
    iCompton = 1
    scat = 4
    nSim = 1
    nPointsSim = 10
    approx = "inv"

    print(f"data/boundTR_{nSim}_{scat}_{iCompton}_{approx}_{nPointsSim}.txt")
    mZp, gL = hp.loadSorted(
        f"data/boundTR_{nSim}_{scat}_{iCompton}_{approx}_{nPointsSim}.txt"
    )
    prec = len(mZp)
    R, T, mu_e, mu_mu, _, _, mu_nue, mu_numu = helper.unpack(nSim)
    iSphere = helper.getRadiusSphere(mChi, R, T, nSim=nSim, out=False)
    dat = np.zeros((prec, 8))
    dat[:, 0] = mZp
    for j in range(prec):
        if np.isnan(gL[j]):  # keep nans
            dat[j, 1:] = [math.nan for _ in range(7)]
            continue
        (
            dat[j, 1],
            dat[j, 2],
            dat[j, 3],
            dat[j, 4],
            dat[j, 5],
            dat[j, 6],
            dat[j, 7],
        ) = calc.lambdaInvMean(
            helper.mmu,
            mChi,
            mu_mu[iSphere],
            mu_numu[iSphere],
            T[iSphere],
            scat=scat,
            mZp=mZp[j],
            gChi=gL[j],
            gL=gL[j],
            approx=approx,
            oneFermion=False,
            iCompton=iCompton,
            giveRatios=True,
        )
    np.savetxt(f"data/ratios.txt", dat)


def plot():
    fSize = 13
    name = [r"$e$", r"$\mu$", r"$\nu_e = \nu_\mu$"]

    ymin = 1e-3
    ymax = 1.15
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    dat = np.loadtxt(f"data/ratios.txt")
    mZp = dat[:, 0]
    # ax[iL].set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(mZp[0], mZp[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$m_{Z'}$ [MeV]", fontsize=fSize)
    ax.set_ylabel(
        r"$\langle\lambda^{-1}\rangle^{-1}_i / \langle\lambda^{-1}\rangle^{-1}_\mathrm{tot}$",
        fontsize=fSize,
    )
    ax.tick_params(axis="both", which="major", labelsize=fSize)

    axtop = ax.twiny()
    axtop.set_xscale("log")
    axtop.set_xlim(mZp[0], mZp[-1])
    axtop.set_xticklabels([])

    axright = ax.twinx()
    # axright.set_yscale("log")
    axright.set_ylim(ymin, ymax)
    axright.set_yticklabels([])

    ax.plot(mZp, 1.0 + mZp * 0, "k--")
    ax.plot(mZp, dat[:, 1], color=col[0], label=r"$\chi\chi\to\mu\mu$")
    ax.plot(mZp, dat[:, 2], color=col[1], label=r"$\chi\mu\to\chi\mu$")
    ax.plot(mZp, dat[:, 3], color=col[2], label=r"$\chi\chi\to \chi\chi$ (s)")
    ax.plot(mZp, dat[:, 4], color=col[3], label=r"$\chi\chi\to \chi\chi$ (t)")
    ax.plot(mZp, dat[:, 5], color=col[4], label=r"$\chi\chi\mu\to \mu\gamma$")
    ax.plot(mZp, dat[:, 6], color=col[6], label=r"$\chi\chi\to\nu\nu$")
    ax.plot(mZp, dat[:, 7], color=col[7], label=r"$\chi\nu\to\chi\nu$")
    ax.legend()

    plt.savefig("mutau_ratios.pdf", bbox_inches="tight")


# main()
plot()
