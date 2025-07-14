import numpy as np
import scipy.optimize as opt

import os, sys, time

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("..")
import helper
import calcIndividual as calc
import cross_sections as cs

import warnings

warnings.filterwarnings("ignore")


def getCoupling_FS(iL, mChi, iInteraction, rangeSim=[40, 100], nSim=1, out=True):
    n1, n2 = rangeSim
    N = n2 - n1
    dQdR = np.zeros(N)
    R, T, mu_e, mu_mu, _, _, mu_nue, mu_numu = helper.unpack(nSim)
    mu = [mu_e, mu_mu, mu_nue, mu_numu][iL]
    mL = [helper.me, helper.mmu, 0.0, 0.0][iL]
    for i in range(N):
        dQdR[i] = calc.dQdR(
            iL,
            mL,
            mChi,
            mu[n1 + i],
            T[n1 + i],
            R[n1 + i],
            iInteraction,
            iCompton=0,
            Lambda=1.0,
            oneFermion=True,
            limit="eft",
        )
    Q = np.trapz(dQdR, x=R[n1:n2])

    Lambda = (Q / helper.getQbound(nSim)) ** 0.25 * 1e-6  # in TeV
    if out:
        print(
            "EFT scale with FSLumi = RaffeltLumi: Lambda = {0:.1e} TeV".format(Lambda)
        )
    return Lambda


def getCoupling_TR(
    iL, mChi, iInteraction, nPointsSim=50, nSim=1, scat=1, approx="exact", out=True
):
    R, T, mu_e, mu_mu, _, _, mu_nue, mu_numu = helper.unpack(nSim)
    iSphere = helper.getRadiusSphere(mChi, R, T, nSim=nSim, out=False)
    if iSphere is None:
        print(
            f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV"
        )
        return None
    lambdaInv = np.zeros(nPointsSim)
    mu = [mu_e, mu_mu, mu_nue, mu_numu][iL]
    mL = [helper.me, helper.mmu, 0.0, 0.0][iL]
    for i in range(nPointsSim):
        lambdaInv[i] = calc.lambdaInvMean(
            iL,
            mL,
            mChi,
            mu[iSphere + i],
            T[iSphere + i],
            iInteraction,
            scat=scat,
            Lambda=1.0,
            limit="eft",
            approx=approx,
            oneFermion=True,
            iCompton=0,
        )
    opacity = np.trapz(lambdaInv, R[iSphere : iSphere + nPointsSim])

    Lambda = (opacity / (2 / 3)) ** 0.25 * 1e-6  # in TeV
    if out:
        print("EFT scale with opacity = 2/3: Lambda = {0:.1e} TeV".format(Lambda))
    return Lambda


def main(cluster=True):
    if cluster:
        sys.stdout = open(f"debug/out_eftTable.txt", "w", buffering=1)
        sys.stderr = open(f"debug/err_eftTable.txt", "w", buffering=1)

    mChi = 0.0

    rangeSim = [40, 100]
    nPointsSim = 50
    nSim = 1
    scat = 2
    approx = "exact"  # use exact here!

    bounds = np.zeros(
        (2, 4, 12)
    )  # 2 for TR/FS, 4 for lepton type, 12 for interaction type
    for iL in [0, 1, 2, 3]:
        for iInteraction in range(12):
            t0 = time.time()
            print(
                f"Start calculation for lepton type {iL} and interaction type {iInteraction} ({cs.sigmaName[iInteraction]})"
            )
            bounds[0, iL, iInteraction] = getCoupling_FS(
                iL, mChi, iInteraction, rangeSim=rangeSim, nSim=nSim, out=True
            )
            bounds[1, iL, iInteraction] = getCoupling_TR(
                iL,
                mChi,
                iInteraction,
                nPointsSim=nPointsSim,
                scat=scat,
                approx=approx,
                nSim=nSim,
                out=True,
            )
            if iL == 0 and iInteraction == 0:
                t1 = time.time()
                dtEst = (t1 - t0) * 4 * 12
                print(f"Estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")
    np.save(f"data/eft_table_{approx}", bounds)

    if cluster:
        sys.stdout.close()
        sys.stderr.close()


main(cluster=True)


def printTable(approx="exact"):
    bounds = np.load(f"data/eft_table_{approx}.npy") / 2**0.25
    print(bounds)

    print(bounds[0, 0, 0], round(bounds[0, 0, 0]), round(bounds[0, 0, 0], -1))

    # iInteraction = [0, 2, 4, 6, 8, 9, 10]
    iInteraction = np.arange(12)
    outString = r"\hline " + "\n"
    # outString += r"$X_\chi Y_\ell$ & $\Lambda_e^{\rm eff}$ [TeV] & $\Lambda_\mu^{\rm eff}$ [TeV] &" \
    #            r"$\Lambda_{\nu_e}^{\rm eff}$ [TeV] & $\Lambda_{\nu_\mu}^{\rm eff}$ [TeV] \\"+"\n"
    outString += (
        r"$X_\chi Y_\ell$ & $\Lambda_e^{\rm eff}$ [TeV] & $\Lambda_\mu^{\rm eff}$ [TeV] &"
        r" $\Lambda_{\nu_\mu}^{\rm eff}$ [TeV] \\" + "\n"
    )
    outString += r"\hline " + "\n"
    for i in iInteraction:
        outString += "$" + cs.sigmaName[i] + "$"  # have to implement sigmaName again!
        for j in [0, 1, 2]:
            outString += " & ${0:.2g} - {1:.2g}$".format(
                bounds[1, j, i], bounds[0, j, i]
            )
        outString += r"\\" + "\n"
    outString += r"\hline "
    print(outString)


printTable()
"""
\hline
$X_\chi Y_\ell$ & $\Lambda_e^{\rm eff}$ [TeV] & $\Lambda_\mu^{\rm eff}$ [TeV] \\
\hline
$SS$ & $3.9-0.06$ & $3.1-0.0016$\\
%$SP$ &$3.9-0.06$ & $3.1-0.0016$\\
$PS$ & $3.9-0.06$& $3.6-0.007$ \\
%$PP$ & $3.9-0.06$ & $3.6-0.007$ \\
$VV$ & $4.2-0.1$& $4.1-0.0017$\\
%$VA$ & $4.2-0.1$& $4.1-0.0017$\\
$AV$ & $4.2-0.1$& $3.3-0.0021$\\
%$AA$ & $4.2-0.1$& $3.3-0.0021$\\
$LL$  & $2.9-0.07$& $2.7-0.0014$\\
$TT$ & $4.9-0.16$& $4.8-0.0031$ \\
%$T^\prime T$ &$4.9-0.16$ &  $4.8-0.0031$\\
\hline
"""
