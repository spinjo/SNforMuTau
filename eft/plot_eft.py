import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import matplotlib as mpl

import os, sys

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../")
import helper_plot as hp

me = 0.5
mmu = 106

xmin = 1e0
xmax = 1e3
ymin = [2e-2, 8e-4, 2e-2]
ymax = [2e1, 2e1, 2e1]
prec = int(1e3)

nPointsSim = 50

col = mpl.cm.Set1(np.linspace(0, 1, 9))[0:9]

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True


def plot(approx="exact"):
    x = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(prec)))

    fSize = 15
    plt.rcParams["hatch.linewidth"] = 0.2

    name = [r"$e$", r"$\mu$", r"$\nu$"]  # r"$\nu_e$", r"$\nu_\mu$"]

    f, ax = plt.subplots(3, 1, figsize=(6, 12), gridspec_kw={"hspace": 0.25})
    L = [1, 2, 0]
    for i in range(3):
        iL = L[i]
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(ymin[iL], ymax[iL])
        ax[i].set_xlabel(r"$m_\chi$ [MeV]", fontsize=fSize)

        # axtop=ax[i].twiny()
        # axtop.set_xscale("log")
        # axtop.set_xlim(xmin, xmax)
        # axtop.set_xticklabels([])

        axright = ax[i].twinx()
        axright.set_yscale("log")
        axright.set_ylim(ymin[iL], ymax[iL])
        axright.set_yticklabels([])

        if iL == 0:
            ax[i].set_ylabel(r"$\Lambda^\mathrm{eff}_e$ [TeV]", fontsize=fSize)
        elif iL == 1:
            ax[i].set_ylabel(r"$\Lambda^\mathrm{eff}_\mu$ [TeV]", fontsize=fSize)
        elif iL == 2:
            ax[i].set_ylabel(r"$\Lambda^\mathrm{eff}_\nu$ [TeV]", fontsize=fSize)
            # ax[iL].set_ylabel(r"$\Lambda^\mathrm{eff}_{\nu_e}$ [TeV]", fontsize=fSize)
        # elif iL==3:
        #    ax[iL].set_ylabel(r"$\Lambda^\mathrm{eff}_{\nu_\mu}$ [TeV]", fontsize=fSize)
        ax[i].tick_params(axis="both", which="major", labelsize=fSize)

        mSN = np.loadtxt(f"data/boundFS_{iL}_1.txt")[:, 0]
        ySN_FS1 = np.loadtxt(f"data/boundFS_{iL}_1.txt")[:, 1]
        ySN_FS2 = np.loadtxt(f"data/boundFS_{iL}_2.txt")[:, 1]

        ySN_TR1_1 = np.loadtxt(f"data/boundTR_{iL}_1_1_{approx}_{nPointsSim}.txt")[:, 1]
        ySN_TR1_2 = np.loadtxt(f"data/boundTR_{iL}_1_2_{approx}_{nPointsSim}.txt")[:, 1]
        ySN_TR1_4 = np.loadtxt(f"data/boundTR_{iL}_1_4_{approx}_{nPointsSim}.txt")[:, 1]
        ySN_TR2_2 = np.loadtxt(f"data/boundTR_{iL}_2_2_{approx}_{nPointsSim}.txt")[:, 1]
        x = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(prec)))
        ySN_FS1_0 = ySN_FS1
        ySN_FS1, ySN_TR1_2 = hp.improve(
            mSN, x, ySN_FS1_0, ySN_TR1_2, approach=0, interp=True
        )
        ySN_FS1_1, ySN_TR1_1 = hp.improve(
            mSN, x, ySN_FS1_0, ySN_TR1_1, approach=0, interp=True
        )
        ySN_FS1_4, ySN_TR1_4 = hp.improve(
            mSN, x, ySN_FS1_0, ySN_TR1_4, approach=0, interp=True
        )
        ySN_FS2, ySN_TR2_2 = hp.improve(
            mSN, x, ySN_FS2, ySN_TR2_2, approach=0, interp=True
        )
        mSN = x

        # note: added factors of sqrt(2) here to compensate for typo

        ax[i].fill_between(mSN, ySN_TR1_2, ySN_FS1 / 2**0.5, color=col[4], alpha=0.4)

        ax[i].plot(mSN, ySN_TR2_2, color=col[4], dashes=[1, 1])
        ax[i].plot(mSN, ySN_FS2 / 2**0.5, color=col[4], dashes=[1, 1])

        ax[i].fill_between(
            mSN,
            hp.getMin(ySN_TR1_2, ySN_FS1_1 / 2**0.5),
            ySN_TR1_1,
            color="none",
            facecolor="none",
            hatch="xxx",
            edgecolor=col[4],
            linewidth=0.0,
        )
        ax[i].text(
            8.6e2,
            ymax[iL] / (np.log(ymax[iL]) - np.log(ymin[iL])) ** 0.6,
            name[iL],
            fontsize=fSize * 1.8,
            horizontalalignment="right",
        )

    plt.savefig(f"eft_{approx}.pdf", bbox_inches="tight")


# plot(approx="inv")
plot(approx="exact")
