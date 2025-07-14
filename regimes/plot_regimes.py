import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import matplotlib as mpl
import matplotlib.patches as patches

import os, sys

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../")
import helper_plot as hp

me = 0.5
mmu = 106

xmin = 3e0
xmax = 1e4
ymin = [1e-11] * 3
ymax = [1.0] * 3
prec = int(1e3)

xlight_FS = [8e1, 2e2, 1.5e3]
xheavy_FS = [1.5e3] * 3
xlight_TR = [xmin] * 3
xheavy_TR = [1.2e2, 2.4e1, 1.2e2]
y_FS = np.array(ymin) * 2
y_TR = np.array(ymax) / 2
text_FS_light = ["res. Photoprod.", "res. Photoprod.", "res. Ann."]
text_FS_res = ["res. Ann.", "res. Ann.", None]
text_FS_heavy = ["EFT Ann.", "EFT Ann.", "EFT Ann."]
text_TR_light = [None, None, None]
text_TR_res = ["res. Photoprod.", "res. Photoprod.", "res. Ann."]
text_TR_heavy = ["EFT Ann.", "EFT Ann.", "EFT Ann."]

nPointsSim = 50

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

col = mpl.cm.Set1(np.linspace(0, 1, 9))[0:9]


def plot(joint=True, iCompton=0, approx="exact"):
    x = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(prec)))

    fSize = 15
    plt.rcParams["hatch.linewidth"] = 0.2

    name = [r"$e$", r"$\mu$", r"$\nu$"]  # r"$\nu_e$", r"$\nu_\mu$"]

    def plot_ax(ax, iL, iCompton=0, approx="exact"):
        if iCompton == 2 and (iL == 0 or iL == 2):
            return
        # if iL==2 and iCompton==1:
        #    iCompton2=iCompton
        #    iCompton=0
        # else:
        #    iCompton2=0

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin[iL], ymax[iL])
        ax.set_xlabel(r"$m_{Z'}$ [MeV]", fontsize=fSize)

        y_minor = mpl.ticker.LogLocator(
            base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
        )
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        # axtop=ax.twiny()
        # axtop.set_xscale("log")
        # axtop.set_xlim(xmin, xmax)
        # axtop.set_xticklabels([])

        axright = ax.twinx()
        axright.set_yscale("log")
        axright.set_ylim(ymin[iL], ymax[iL])
        axright.set_yticks([1e-2, 1e-5, 1e-8, 1e-11])
        axright.set_yticklabels([])

        ax.set_yticks([1e0, 1e-3, 1e-6, 1e-9])
        ax.set_yticks([1e-1, 1e-2, 1e-4, 1e-5, 1e-7, 1e-8, 1e-10, 1e-11], minor=True)
        ax.set_yticklabels([r"$1$", r"$10^{-3}$", r"$10^{-6}$", r"$10^{-9}$"])
        axright.set_yticks([1e0, 1e-3, 1e-6, 1e-9])
        axright.set_yticks(
            [1e-1, 1e-2, 1e-4, 1e-5, 1e-7, 1e-8, 1e-10, 1e-11], minor=True
        )
        axright.set_yticklabels([], minor=True)

        if iL == 0:
            ax.set_ylabel(r"$g_e = g_\chi$", fontsize=fSize)
        elif iL == 1:
            ax.set_ylabel(r"$g_\mu = g_\chi$", fontsize=fSize)
        elif iL == 2:
            ax.set_ylabel(r"$g_\nu = g_\chi$", fontsize=fSize)

        ax.tick_params(axis="both", which="major", labelsize=fSize)

        mSN, ySN_FS1 = hp.loadSorted(f"data/boundFS_{iL}_{iCompton}_1.txt")
        _, ySN_FS2 = hp.loadSorted(f"data/boundFS_{iL}_{iCompton}_2.txt")

        _, ySN_TR1_2 = hp.loadSorted(
            f"data/boundTR_{iL}_1_2_{iCompton}_{nPointsSim}_{approx}.txt"
        )
        _, ySN_TR1_4 = hp.loadSorted(
            f"data/boundTR_{iL}_1_4_{iCompton}_{nPointsSim}_{approx}.txt"
        )
        _, ySN_TR1_1 = hp.loadSorted(
            f"data/boundTR_{iL}_1_1_{iCompton}_{nPointsSim}_{approx}.txt"
        )
        _, ySN_TR2_2 = hp.loadSorted(
            f"data/boundTR_{iL}_2_2_{iCompton}_{nPointsSim}_{approx}.txt"
        )

        ax.fill_between(mSN, ySN_TR1_2, ySN_FS1, color=col[4], alpha=0.4)
        ax.fill_between(
            mSN,
            ySN_TR1_2,
            ySN_TR1_1,
            color="none",
            facecolor="none",  # hp.getMin(ySN_TR1_4, ySN_FS1_1)
            hatch="xxx",
            edgecolor=col[4],
            linewidth=0.0,
        )

        ax.plot(mSN, ySN_TR2_2, color=col[4], dashes=[1, 1])
        ax.plot(mSN, ySN_FS2, color=col[4], dashes=[1, 1])

        def logmean(a, b):
            return np.exp((np.log(a) + np.log(b)) / 2)

        ars = 10
        al = 0.6
        arrow_FS_1 = patches.FancyArrowPatch(
            (xmin / 5, y_FS[iL]),
            (xlight_FS[iL], y_FS[iL]),
            arrowstyle="-|>",
            mutation_scale=ars,
            alpha=al,
        )
        arrow_FS_2 = patches.FancyArrowPatch(
            (xlight_FS[iL], y_FS[iL]),
            (xheavy_FS[iL], y_FS[iL]),
            arrowstyle="<|-|>",
            mutation_scale=ars,
            alpha=al,
        )
        arrow_FS_3 = patches.FancyArrowPatch(
            (xheavy_FS[iL], y_FS[iL]),
            (xmax * 5, y_FS[iL]),
            arrowstyle="<|-",
            mutation_scale=ars,
            alpha=al,
        )
        ax.add_patch(arrow_FS_1)
        ax.add_patch(arrow_FS_2)
        ax.add_patch(arrow_FS_3)
        arrow_TR_1 = patches.FancyArrowPatch(
            (xmin / 5, y_TR[iL]),
            (xheavy_TR[iL], y_TR[iL]),
            arrowstyle="-|>",
            mutation_scale=ars,
            alpha=al,
        )
        arrow_TR_2 = patches.FancyArrowPatch(
            (xheavy_TR[iL], y_TR[iL]),
            (xmax * 5, y_TR[iL]),
            arrowstyle="<|-",
            mutation_scale=ars,
            alpha=al,
        )
        ax.add_patch(arrow_TR_1)
        ax.add_patch(arrow_TR_2)

        facy_TR = 1 / 5
        facy_FS = 1.7
        fSize_1 = 12
        alReg = al
        ax.text(
            logmean(xmin, xlight_TR[iL]),
            y_TR[iL] * facy_TR,
            text_TR_light[iL],
            alpha=alReg,
            fontsize=fSize_1,
            horizontalalignment="center",
        )
        ax.text(
            logmean(xlight_TR[iL], xheavy_TR[iL]),
            y_TR[iL] * facy_TR,
            text_TR_res[iL],
            alpha=alReg,
            fontsize=fSize_1,
            horizontalalignment="center",
        )
        if iL != 1:
            ax.text(
                logmean(xheavy_TR[iL], xmax),
                y_TR[iL] * facy_TR,
                text_TR_heavy[iL],
                alpha=alReg,
                fontsize=fSize_1,
                horizontalalignment="center",
            )
        else:
            ax.text(
                logmean(xheavy_TR[iL], xmax),
                y_TR[iL] * facy_TR,
                text_TR_heavy[iL],
                alpha=alReg,
                fontsize=fSize_1,
                horizontalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.6,
                    "edgecolor": "white",
                    "boxstyle": "round",
                    "pad": 0.2,
                },
            )

        ax.text(
            logmean(xmin, xlight_FS[iL]),
            y_FS[iL] * facy_FS,
            text_FS_light[iL],
            alpha=alReg,
            fontsize=fSize_1,
            horizontalalignment="center",
        )
        ax.text(
            logmean(xlight_FS[iL], xheavy_FS[iL]),
            y_FS[iL] * facy_FS,
            text_FS_res[iL],
            alpha=alReg,
            fontsize=fSize_1,
            horizontalalignment="center",
        )
        ax.text(
            logmean(xheavy_FS[iL], xmax),
            y_FS[iL] * facy_FS,
            text_FS_heavy[iL],
            alpha=alReg,
            fontsize=fSize_1,
            horizontalalignment="center",
        )

        # has problem!?
        # 5e3
        ax.text(
            6e3,
            ymax[iL] / (np.log(ymax[iL]) - np.log(ymin[iL])) * 0.2,
            name[iL],
            fontsize=fSize * 1.8,
            horizontalalignment="left",
        )

        # if iL==2 and iCompton2==1:
        #    iCompton=1

    if joint:
        f, ax = plt.subplots(3, 1, figsize=(6, 12), gridspec_kw={"hspace": 0.25})
        iLarr = [1, 2, 0]
        for i in range(3):
            iL = iLarr[i]
            plot_ax(ax[i], iL, iCompton=iCompton, approx=approx)
        plt.savefig(f"regimes_{approx}_{iCompton}.pdf", bbox_inches="tight")
    else:
        for iL in range(3):
            f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
            plot_ax(ax, iL, iCompton=iCompton, approx=approx)
            plt.savefig(f"regimes_{approx}_{iCompton}_{iL}.pdf", bbox_inches="tight")


# plot(iCompton=0, approx="exact")
plot(iCompton=1, approx="exact")
# plot(iCompton=0, approx="inv")
# plot(iCompton=1, approx="inv")
# plot(iCompton=2, approx="exact") #have to manually use different iCompton for FS and TR
