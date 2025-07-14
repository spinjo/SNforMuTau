import numpy as np
import scipy.optimize as opt

import os, sys, time
import filelock

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("..")
import helper
import calcMuTau as calc

import warnings

warnings.filterwarnings("ignore")

# SN point with largest contribution: 62
def checkModel_FS(
    mZp, mChiOvermZp, gL, gChiOvergL, iCompton=0, rangeSim=[50, 80], nSim=1
):
    mChi = mChiOvermZp * mZp
    gChi = gL * gChiOvergL

    n1, n2 = rangeSim
    N = n2 - n1
    dQdR = np.zeros(N)
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    for i in range(N):
        dQdR[i] = calc.dQdR(
            helper.mmu,
            mChi,
            mu_mu[n1 + i],
            mu_numu[n1 + i],
            T[n1 + i],
            R[n1 + i],
            mZp=mZp,
            gChi=gChi,
            gL=gL,
            iCompton=iCompton,
            oneFermion=False,
            limit="full",
        )

    Q = np.trapz(dQdR, x=R[n1:n2])
    return Q


def getCoupling_FS(
    mZp,
    mChiOvermZp,
    gChiOvergL,
    rangeSim=[50, 80],
    nSim=1,
    iCompton=0,
    guessLower=1e-5,
    guessUpper=1e-3,
    defaultVal=1e10,
    out=True,
    outCheck=False,
):
    def checkFS(gL):
        Q = checkModel_FS(
            mZp,
            mChiOvermZp,
            gL,
            gChiOvergL,
            rangeSim=rangeSim,
            nSim=nSim,
            iCompton=iCompton,
        )
        Qbound = helper.getQbound(nSim)
        if outCheck:
            print(
                "gL = {0:.2e} \t checkFS (gL) = {1:.2e}".format(
                    gL, (Q - Qbound) / (Q + Qbound)
                )
            )
        return (Q - Qbound) / (Q + Qbound)

    try:
        sol = opt.root_scalar(
            checkFS, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748"
        )
    except ValueError:
        print(
            f"ERROR: No solution in the range [{guessLower}, {guessUpper}]. Please adapt the range."
        )
        return defaultVal
    gL = sol.root
    if out:
        print(
            "Coupling with FSLumi = RaffeltLumi for mZp={0:.1e}: gL = {1:.1e}".format(
                mZp, gL
            )
        )
    return gL


def main(iCompton=0, nSim=1, cluster=True, split=False):
    if cluster:
        CLargument = int(sys.argv[1])  # 0-1 (split=False), 0-199 (split=True)
        sys.stdout = open(f"debug/outFS_{nSim}_{iCompton}.txt", "w", buffering=1)
        sys.stderr = open(f"debug/errFS_{nSim}_{iCompton}.txt", "w", buffering=1)

    prec = 100
    mZpmin = 1e0
    mZpmax = 1e4
    mZp = np.exp(np.linspace(np.log(mZpmin), np.log(mZpmax), prec))

    mChiOvermZp = 1 / 3
    gChiOvergL = 1.0
    rangeSim = [40, 100]

    guessLower = 1e-12
    guessUpper = 1e-2
    defaultVal = 1e-2

    if cluster:
        if split:
            j = CLargument % prec
            nSim = CLargument // prec + 1
        else:
            nSim = CLargument + 1

    print(f"### Calculation for simulation {nSim} ###")

    bound = np.zeros(prec)
    t00 = time.time()
    if not split:
        for j in range(prec):
            t0 = time.time()
            bound[i] = getCoupling_FS(
                mZp[j],
                mChiOvermZp,
                gChiOvergL,
                rangeSim=rangeSim,
                guessLower=guessLower,
                guessUpper=guessUpper,
                defaultVal=defaultVal,
                nSim=nSim,
                iCompton=iCompton,
                out=True,
                outCheck=True,
            )
            t1 = time.time()
            if j == 0:
                print(
                    f"Estimate: {(t1-t0)*prec:.2f} s = {(t1-t0)/60*prec:.2f} min = {(t1-t0)/60**2*prec:.2f} h"
                )
        ret = np.zeros((prec, 2))
        ret[:, 0] = mZp
        ret[:, 1] = bound
        np.savetxt(f"data/boundFS_{iCompton}_{nSim}.txt", ret)
    else:
        bound = getCoupling_FS(
            mZp[j],
            mChiOvermZp,
            gChiOvergL,
            rangeSim=rangeSim,
            guessLower=guessLower,
            guessUpper=guessUpper,
            defaultVal=defaultVal,
            nSim=nSim,
            iCompton=iCompton,
            out=True,
            outCheck=True,
        )
        ret = np.zeros((1, 2))
        ret[:, 0] = mZp[j]
        ret[:, 1] = bound
        path = f"data/boundFS_{iCompton}_{nSim}.txt"
        lock = filelock.FileLock(path + ".lock")
        while True:
            try:
                with lock.acquire(timeout=1):
                    with open(path, "a") as file:
                        np.savetxt(file, ret)
                        print("Successfully wrote results to file")
                break
            except filelock.Timeout:
                print("File locked by another process. Retry after 1s")
                time.sleep(1.0)

    t01 = time.time()
    print(
        f"Total time: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min = {(t01-t00)/60**2:.2f} h"
    )

    if cluster:
        sys.stdout.close()
        sys.stderr.close()


main(cluster=False, iCompton=1)

# main(cluster=True, split=True, iCompton=1)
