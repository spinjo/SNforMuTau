import numpy as np
import scipy.optimize as opt

import os, sys, time

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("..")
import helper
import calcIndividual as calc

import warnings

warnings.filterwarnings("ignore")


def getCoupling_TR(iL, mChi, nPointsSim=30, nSim=1, scat=2, approx="inv", out=True):
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
    iInteraction = 4 if iL < 2 else 9
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
        print(
            "EFT scale with opacity = 2/3 for mChi={0:.1e} MeV: Lambda = {1:.1e} TeV".format(
                mChi, Lambda
            )
        )
    return Lambda


def main(cluster=True, iL=0, scat=2, nSim=1, approx="exact"):
    if cluster:
        CLargument = int(sys.argv[1])  # want 0-11
        sys.stdout = open(f"debug/outTR_{approx}_{CLargument}.txt", "w", buffering=1)
        sys.stderr = open(f"debug/errTR_{approx}_{CLargument}.txt", "w", buffering=1)
    prec = 100

    mChimin = 1e0
    mChimax = 1e3
    mChi = np.exp(np.linspace(np.log(mChimin), np.log(mChimax), prec))

    nPointsSim = 50

    if cluster:
        ibuffer = CLargument
        irun = ibuffer % 4
        iL = ibuffer // 4
        if irun < 3:
            nSim = 1
            scat = 2**irun  # want scat=2 here (because muon scattering)
        elif irun == 3:
            nSim = 2
            scat = 2

    print(f"### Calculation for lepton {iL}, simulation {nSim} and scat={scat}")

    bounds = np.zeros(prec)
    t0 = time.time()
    for j in range(prec):
        bounds[j] = getCoupling_TR(
            iL,
            mChi[j],
            scat=scat,
            approx=approx,
            nPointsSim=nPointsSim,
            nSim=nSim,
            out=True,
        )
        if j == 0:
            t1 = time.time()
            print(f"Estimate: {(t1-t0)*prec/60:.2f} min = {(t1-t0)*prec/60**2:.2f} h")
    dat = np.zeros((prec, 2))
    dat[:, 0] = mChi
    dat[:, 1] = bounds
    np.savetxt(f"data/boundTR_{iL}_{nSim}_{scat}_{approx}_{nPointsSim}.txt", dat)

    t1 = time.time()
    print(f"Total time: {t1-t0:.2f} s = {(t1-t0)/60:.2f} min = {(t1-t0)/60**2:.2f} h")

    if cluster:
        sys.stdout.close()
        sys.stderr.close()


main(cluster=False, iL=0, scat=1, nSim=1, approx="inv")

# main(cluster=True, approx="exact")
