import numpy as np
import scipy.optimize as opt

import os, sys, time
import filelock

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("..")
import helper
import calcIndividual as calc

import warnings

warnings.filterwarnings("ignore")
mChi = 0.0

# SN point with largest contribution: 62
def checkModel_FS(iL, mZp, gL, gChiOvergL, iCompton=0, rangeSim=[50, 80], nSim=1):
    gChi = gL * gChiOvergL

    n1, n2 = rangeSim
    N = n2 - n1
    dQdR = np.zeros(N)
    R, T, mu_e, mu_mu, _, _, mu_nue, mu_numu = helper.unpack(nSim)
    mu = [mu_e, mu_mu, mu_nue, mu_numu][iL]
    mL = [helper.me, helper.mmu, 0.0, 0.0][iL]
    iInteraction = 4 if (iL <= 1) else 9
    for i in range(N):
        dQdR[i] = calc.dQdR(
            iL,
            mL,
            mChi,
            mu[n1 + i],
            T[n1 + i],
            R[n1 + i],
            iInteraction,
            iCompton=iCompton,
            mZp=mZp,
            gChi=gChi,
            gL=gL,
            oneFermion=True,
            limit="full",
        )
    Q = np.trapz(dQdR, x=R[n1:n2])
    return Q


def getCoupling_FS(
    iL,
    mZp,
    gChiOvergL,
    iCompton=0,
    rangeSim=[50, 80],
    nSim=1,
    guessLower=1e-5,
    guessUpper=1e-3,
    defaultVal=1e10,
    out=True,
    outCheck=False,
):
    def checkFS(gL):
        Q = checkModel_FS(
            iL, mZp, gL, gChiOvergL, iCompton=iCompton, rangeSim=rangeSim, nSim=nSim
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


def main(cluster=True, split=False, iL=0, nSim=1, iCompton=0):
    if cluster:
        CLargument = int(sys.argv[1])  # want 0-7
        sys.stdout = open(f"debug/outFS_{iCompton}_{CLargument}.txt", "w", buffering=1)
        sys.stderr = open(f"debug/errFS_{iCompton}_{CLargument}.txt", "w", buffering=1)

    prec = 100
    mZpmin = 3e0  # skip electron resonance
    mZpmax = 1e4
    mZp = np.exp(np.linspace(np.log(mZpmin), np.log(mZpmax), prec))

    gChiOvergL = 1.0
    rangeSim = [40, 100]

    guessLower = 1e-10
    guessUpper = 1e0
    defaultVal = 1e0

    if cluster:
        if split:
            j = CLargument % prec
            ibuffer = CLargument // prec
        else:
            ibuffer = CLargument
        iL = ibuffer % 4
        nSim = ibuffer // 4 + 1

    print(f"### Calculation for simulation {nSim} and lepton {iL} ###")

    bound = np.zeros(prec)
    t00 = time.time()
    if not split:
        for j in range(prec):
            t0 = time.time()
            bound[j] = getCoupling_FS(
                iL,
                mZp[j],
                gChiOvergL,
                iCompton=iCompton,
                rangeSim=rangeSim,
                guessLower=guessLower,
                guessUpper=guessUpper,
                defaultVal=defaultVal,
                nSim=nSim,
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
        np.savetxt(f"data/boundFS_{iL}_{iCompton}_{nSim}.txt", ret)
    else:
        bound = getCoupling_FS(
            iL,
            mZp[j],
            gChiOvergL,
            iCompton=iCompton,
            rangeSim=rangeSim,
            guessLower=guessLower,
            guessUpper=guessUpper,
            defaultVal=defaultVal,
            nSim=nSim,
            out=True,
            outCheck=True,
        )

        dat = np.zeros((1, 2))
        dat[:, 0] = mZp[j]
        dat[:, 1] = bound
        path = f"data/boundFS_{iL}_{iCompton}_{nSim}.txt"
        lock = filelock.FileLock(path + ".lock")
        while True:
            try:
                with lock.acquire(timeout=1):
                    with open(path, "a") as file:
                        np.savetxt(file, dat)
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


# main(cluster=False, iL=0)
# main(cluster=False, iCompton=1, iL=0)
# main(cluster=False, iL=2)

main(cluster=True, split=False, iCompton=0)
# main(cluster=True, iCompton=1)
