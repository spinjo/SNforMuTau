import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import matplotlib as mpl

import os, sys

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../")
import helper_plot as hp
import OtherConstraints.Neff.ClaudioNeff as Neff
import OtherConstraints.g2.g2Bounds as g2

me = 0.5
mmu = 106

xmin = 3e0
xmax = 3e3
ymin = 1.5e-10
ymax = 4e-3
prec = int(1e3)

nPointsSim = 50
data = "data/"

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
# plt.rcParams['ytick.major.size'] = 5

col = mpl.cm.Set1(np.linspace(0, 1, 9))[0:9]


def getPlot(x):
    data = np.loadtxt("plot/existingBounds.csv", delimiter=",")
    mCCFR = data[:, 0]
    yCCFR = data[:, 1]
    fCCFR = itp.interp1d(mCCFR, yCCFR, fill_value="extrapolate")
    yCCFR = fCCFR(x)
    data = np.loadtxt("plot/m3phase1.csv", delimiter=",")
    mM3P1 = data[:, 0] * 1e3
    yM3P1 = data[:, 1]
    fM3P1 = itp.interp1d(mM3P1, yM3P1, fill_value="extrapolate")
    yM3P1 = fM3P1(x)
    data = np.loadtxt("plot/m3phase2.csv", delimiter=",")
    mM3P2 = data[:, 0] * 1e3
    yM3P2 = data[:, 1]
    fM3P2 = itp.interp1d(mM3P2, yM3P2, fill_value="extrapolate")
    yM3P2 = fM3P2(x)
    data = np.loadtxt("plot/na64mu.csv", delimiter=",")
    mNA64 = data[:, 0] * 1e3
    yNA64 = data[:, 1]
    fNA64 = itp.interp1d(mNA64, yNA64, fill_value="extrapolate")
    yNA64 = fNA64(x)
    data = np.loadtxt("plot/na62.csv", delimiter=",")
    mNA62 = data[:, 0]
    yNA62 = (4 * np.pi * data[:, 1]) ** 0.5
    fNA62 = itp.interp1d(np.log(mNA62), np.log(yNA62), fill_value="extrapolate")
    yNA62 = np.exp(fNA62(np.log(x)))
    data = np.loadtxt("plot/SHIPlu.csv", delimiter=",")
    mSHIPlu = data[:, 0] * 1e3
    ySHIPlu = data[:, 1]
    fSHIPlu = itp.interp1d(mSHIPlu, ySHIPlu, fill_value="extrapolate")
    ySHIPlu = np.zeros(len(x))
    for i in range(len(x)):
        ySHIPlu[i] = fSHIPlu(x[i]) if x[i] < np.max(mSHIPlu) else None
    data = np.loadtxt("plot/SHIPld.csv", delimiter=",")
    mSHIPld = data[:, 0] * 1e3
    ySHIPld = data[:, 1]
    fSHIPld = itp.interp1d(mSHIPld, ySHIPld, fill_value="extrapolate")
    ySHIPld = np.zeros(len(x))
    for i in range(len(x)):
        ySHIPld[i] = fSHIPld(x[i]) if x[i] < np.max(mSHIPld) else None
    data = np.loadtxt("plot/SHIPru.csv", delimiter=",")
    mSHIPru = data[:, 0] * 1e3
    ySHIPru = data[:, 1]
    fSHIPru = itp.interp1d(mSHIPru, ySHIPru, fill_value="extrapolate")
    ySHIPru = np.zeros(len(x))
    for i in range(len(x)):
        ySHIPru[i] = (
            fSHIPru(x[i])
            if (x[i] < np.max(mSHIPru) and x[i] > np.min(mSHIPru))
            else None
        )
    data = np.loadtxt("plot/SHIPrd.csv", delimiter=",")
    mSHIPrd = data[:, 0] * 1e3
    ySHIPrd = data[:, 1]
    fSHIPrd = itp.interp1d(mSHIPrd, ySHIPrd, fill_value="extrapolate")
    ySHIPrd = np.zeros(len(x))
    for i in range(len(x)):
        ySHIPrd[i] = (
            fSHIPrd(x[i])
            if (x[i] < np.max(mSHIPrd) and x[i] > np.min(mSHIPrd))
            else None
        )

    return yCCFR, yM3P1, yM3P2, yNA64, yNA62, ySHIPlu, ySHIPld, ySHIPru, ySHIPrd


def plot(iCompton=0, approx="exact"):
    x = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(prec)))
    yCCFR, yM3P1, yM3P2, yNA64, yNA62, ySHIPlu, ySHIPld, ySHIPru, ySHIPrd = getPlot(x)
    gNeffLower, gNeffUpper = Neff.getNeff(x)
    yg2Lower, yg2Upper = g2.getNsigma(x, 2)
    yCCFR = np.min(np.array([yCCFR, gNeffUpper]), axis=0)

    def loadSorted(file, iStart=0):
        m = np.loadtxt(file)[iStart:, 0]
        print(file, np.shape(m))
        y = np.loadtxt(file)[iStart:, 1]
        idx = np.argsort(m)
        return m[idx], y[idx]

    mSN, ySN_FS1 = hp.loadSorted(f"data/boundFS_{iCompton}_1.txt")
    _, ySN_FS2 = hp.loadSorted(f"data/boundFS_{iCompton}_2.txt")
    _, ySN_TR1 = hp.loadSorted(f"data/boundTR_1_2_{iCompton}_{approx}_{nPointsSim}.txt")
    _, ySN_TR2 = hp.loadSorted(f"data/boundTR_2_2_{iCompton}_{approx}_{nPointsSim}.txt")
    _, ySN_TR1opt = hp.loadSorted(
        f"data/boundTR_1_1_{iCompton}_{approx}_{nPointsSim}.txt"
    )
    _, ySN_TR1pes = hp.loadSorted(
        f"data/boundTR_1_4_{iCompton}_{approx}_{nPointsSim}.txt"
    )

    # fill up with nans (writing nans to file happened to fast when running the code)
    ySN_TR1 = np.append(ySN_TR1, np.full(mSN.size - ySN_TR1.size, np.nan))
    ySN_TR2 = np.append(ySN_TR2, np.full(mSN.size - ySN_TR2.size, np.nan))
    ySN_TR1pes = np.append(ySN_TR1pes, np.full(mSN.size - ySN_TR1pes.size, np.nan))
    ySN_FS1_0 = ySN_FS1

    ySN_FS1, ySN_TR1 = hp.improve(mSN, x, ySN_FS1_0, ySN_TR1, approach=1, interp=True)
    ySN_FS1opt, ySN_TR1opt = hp.improve(
        mSN, x, ySN_FS1_0, ySN_TR1opt, approach=1, interp=True
    )
    ySN_FS1pes, ySN_TR1pes = hp.improve(
        mSN, x, ySN_FS1_0, ySN_TR1pes, approach=1, interp=True
    )
    ySN_FS2, ySN_TR2 = hp.improve(mSN, x, ySN_FS2, ySN_TR2, approach=1, interp=True)
    mSN = x

    fSize = 15
    fSize_2 = 13
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$m_{Z'} = 3 m_\chi$ [MeV]", fontsize=fSize)
    ax.set_ylabel(r"$g_\chi = g_{\mu-\tau}$", fontsize=fSize)
    ax.tick_params(axis="both", which="major", labelsize=fSize)

    # axtop=ax.twiny()
    # axtop.set_xscale("log")
    # axtop.set_xlim(xmin, xmax)
    # axtop.set_xticklabels([])

    axright = ax.twinx()
    axright.set_yscale("log")
    axright.set_ylim(ymin, ymax)
    axright.set_yticklabels([])
    axright.set_yticks([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    ax.set_yticks([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])

    y_minor = mpl.ticker.LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axright.yaxis.set_minor_locator(y_minor)
    axright.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    x_minor = mpl.ticker.LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # ax.fill_between(np.concatenate([[xmin], x]), ymax, np.concatenate([[ymin], yCCFR]), color=(221/256,221/256,221/256))
    ax.fill_between(
        np.concatenate([[xmin], x]),
        ymax,
        np.concatenate([[ymin], yCCFR]),
        color=col[8],
        alpha=0.5,
    )
    # ax.text(3.5e0, 1.2e-3, "available\nexperiments", color=col[8], fontsize=fSize)
    ax.text(
        7.2e0, 7e-5, r"$N_\mathrm{eff}$", color=col[8], rotation=63, fontsize=fSize_2
    )
    # ax.text(1.15e1, 6.5e-4, "WD", color=col[8], rotation=20, fontsize=fSize_2)
    ax.text(4e1, 1.05e-3, "CCFR", color=col[8], rotation=1, fontsize=fSize_2)
    ax.text(3.1e2, 4e-4, r"BaBar $4\mu$", color=col[8], rotation=6, fontsize=fSize_2)

    # ax.plot(mSN, yM3P1, color=col[0], linestyle="--")
    ax.plot(mSN, yM3P2, color=col[1], linestyle="-.")
    ax.plot(mSN, yNA64, color=col[0], dashes=[3, 1])
    # ax.plot(mSN, yNA62, color=col[6], linestyle="--")
    ax.fill_between(
        mSN,
        yg2Upper,
        yg2Lower,
        color=col[2],
        alpha=0.5,
        label=r"$(g-2)_\mu \pm 2\sigma$",
    )
    ax.fill_between(
        mSN, gNeffLower, gNeffUpper, color=col[5], alpha=0.7, label=r"$H_0$ hint"
    )
    ax.plot(x, ySHIPlu, color=col[3], dashes=[5, 1])
    ax.plot(x, ySHIPld, color=col[3], dashes=[5, 1])
    ax.plot(x, ySHIPru, color=col[3], dashes=[5, 1])
    ax.plot(x, ySHIPrd, color=col[3], dashes=[5, 1])

    # ax.plot(x, yDM, color=col[7], linestyle="-", label="DM")

    ax.fill_between(mSN, ySN_TR1, ySN_FS1, color=col[4], alpha=0.4)
    plt.rcParams["hatch.linewidth"] = 0.2
    ax.fill_between(
        mSN,
        ySN_TR1opt,
        hp.getMin(ySN_TR1, ySN_FS1),
        color="none",
        facecolor="none",
        hatch="xxx",
        edgecolor=col[4],
        linewidth=0.0,
    )

    ax.plot(mSN, ySN_FS2, color=col[4], dashes=[1, 1])
    ax.plot(mSN, ySN_TR2, color=col[4], dashes=[1, 1])

    # ax.text(3.2e0, yNA62[0]*1.1, r"NA62", color=col[6], fontsize=fSize)
    # ax.text(3.2e0, yM3P1[0]*1.1, r"M$^3$ Phase 1", color=col[0], fontsize=fSize)
    ax.text(3.2e0, yNA64[0] * 1.3, r"NA64$\mu$", color=col[0], fontsize=fSize_2)
    ax.text(3.2e0, 2.4e-6, r"M$^3$ Phase 2", color=col[1], fontsize=fSize_2)
    ax.text(3.2e0, 6e-7, "SHIP", color=col[3], rotation=-5, fontsize=fSize_2)
    ax.text(3e1, 1e-8, "SN1987A", color=col[4], fontsize=fSize)

    ax.legend(loc=4, fontsize=fSize)

    plt.savefig(f"mutau_{approx}_{iCompton}.pdf", bbox_inches="tight")


# plot(iCompton=0, approx="inv")
# plot(iCompton=1, approx="inv")
# plot(iCompton=0, approx="exact")
plot(iCompton=1, approx="exact")
