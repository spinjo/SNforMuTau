import numpy as np
import scipy.integrate as itg
import scipy.interpolate as itp
import scipy.special as sp

import os, sys, time

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../../calc_v2/")
import helper_basics as hb
import helper_calc as hc

mL = hb.mmu


def getBRinv(mZp, mChi, mL, gL, gChi):
    GamTot = hc.GamZp(mZp, mL, mChi, gL, gChi, withNu=True)
    GamInv = hc.GamZpInv(mZp, mL, mChi, gL, gChi, withNu=True)
    return GamInv / GamTot


def getM3PhaseN(mZpNew, mChiOvermZp, n):  # agrees with Patricks bounds
    data = np.loadtxt(
        "/media/jonas/geheim/Studium/supernovaMuons/OtherConstraints/M3andNA64/m3phase"
        + str(int(n))
        + ".csv",
        delimiter=",",
    )
    mChiOld = data[:, 0]
    yOld = data[:, 1]
    mZp = 3 * mChiOld
    gLOld = 3**2 * yOld**0.5
    gChiOld = 1

    mChiNew = mZp * mChiOvermZp

    gNew = np.zeros(len(mZp))
    g0 = 1e-4
    for i in range(len(mZp)):

        def check(g0):
            facOld = gLOld[i] ** 2 * getBRinv(mZp[i], mChiOld[i], mL, gLOld[i], gChiOld)
            facNew = g0**2 * getBRinv(mZp[i], mChiNew[i], mL, g0, g0)
            diff = (facOld - facNew) / (facOld + facNew) * 2
            return diff

        gNew[i] = hc.find(g0, check, output=False)
        if i > 1:
            g0 = 2 * gNew[i] - gNew[i - 1]
    gItp = np.exp(
        itp.interp1d(
            np.log(mZp), np.log(gNew), kind="linear", fill_value="extrapolate"
        )(np.log(mZpNew))
    )
    return gItp


def getNA64mu(mZpNew, mChiOvermZp):  # agrees with Patricks bounds
    data = np.loadtxt(
        "/media/jonas/geheim/Studium/supernovaMuons/OtherConstraints/M3andNA64/na64mu.csv",
        delimiter=",",
    )
    mChiOld = data[:, 0]
    yOld = data[:, 1]
    mZp = 3 * mChiOld
    gLOld = 3**2 * yOld**0.5
    gChiOld = 1

    mChiNew = mZp * mChiOvermZp

    gNew = np.zeros(len(mZp))
    g0 = 1e-4
    for i in range(len(mZp)):

        def check(g0):
            facOld = gLOld[i] ** 2 * getBRinv(mZp[i], mChiOld[i], mL, gLOld[i], gChiOld)
            facNew = g0**2 * getBRinv(mZp[i], mChiNew[i], mL, g0, g0)
            diff = (facOld - facNew) / (facOld + facNew) * 2
            return diff

        gNew[i] = hc.find(g0, check, output=False)
        if i > 1:
            g0 = 2 * gNew[i] - gNew[i - 1]
    gItp = np.exp(
        itp.interp1d(
            np.log(mZp), np.log(gNew), kind="linear", fill_value="extrapolate"
        )(np.log(mZpNew))
    )
    return gItp


# TBD: Crosscheck Figure 10 left and right in M3 paper

# print(getM3PhaseN(5, 1)) #works
