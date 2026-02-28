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

# SN point with largest contribution: 62
def getCoupling_FS(iL, mChi, rangeSim=[50, 80], nSim=1, out=True):
    n1, n2 = rangeSim
    N = n2 - n1
    dQdR = np.zeros(N)
    R, T, mu_e, mu_mu, _, _, mu_nue, mu_numu = helper.unpack(nSim)
    mu = [mu_e, mu_mu, mu_nue, mu_numu][iL]
    mL = [helper.me, helper.mmu, 0.0, 0.0][iL]
    iInteraction = 4 if iL < 2 else 9
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
    Q = np.trapezoid(dQdR, x=R[n1:n2])

    Lambda = (Q / helper.getQbound(nSim)) ** 0.25 * 1e-6  # in TeV
    if out:
        print(
            "EFT scale with FSLumi = RaffeltLumi for mChi={0:.1e} MeV: Lambda = {1:.1e} TeV".format(
                mChi, Lambda
            )
        )
    return Lambda


def main(cluster=True, iL=0, nSim=1):
    if cluster:
        CLargument = int(sys.argv[1])  # want 0-7
        sys.stdout = open(f"debug/outFS_{CLargument}.txt", "w", buffering=1)
        sys.stderr = open(f"debug/errFS_{CLargument}.txt", "w", buffering=1)
    prec = 100

    mChimin = 1e0
    mChimax = 1e3
    mChi = np.exp(np.linspace(np.log(mChimin), np.log(mChimax), prec))

    rangeSim = [40, 100]

    if cluster:
        iL = CLargument % 4
        nSim = CLargument // 4 + 1

    print(f"### Calculation for simulation {nSim} and lepton {iL} ###")

    bound = np.zeros(prec)
    t00 = time.time()
    for j in range(prec):
        t0 = time.time()
        bound[j] = getCoupling_FS(iL, mChi[j], rangeSim=rangeSim, nSim=nSim, out=True)
        t1 = time.time()
        if j == 0:
            print(
                f"Estimate: {(t1-t0)*prec:.2f} s = {(t1-t0)/60*prec:.2f} min = {(t1-t0)/60**2*prec:.2f} h"
            )
    t01 = time.time()
    print(
        f"Total time: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min = {(t01-t00)/60**2:.2f} h"
    )

    dat = np.zeros((prec, 2))
    dat[:, 0] = mChi
    dat[:, 1] = bound
    np.savetxt(f"data/boundFS_{iL}_{nSim}.txt", dat)

    if cluster:
        sys.stdout.close()
        sys.stderr.close()


# main(cluster=False, iL=0, nSim=1)

main(cluster=True)
