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


def checkModel_TR(
    mZp,
    mChiOvermZp,
    gL,
    gChiOvergL,
    nPointsSim=30,
    nSim=1,
    scat=2,
    approx="inv",
    iCompton=0,
    iSphere=None,
):
    mChi = mChiOvermZp * mZp
    gChi = gL * gChiOvergL

    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    if iSphere is None:  # calculate radius of sphere first, if not calculated before
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim=nSim, out=False)
        if iSphere is None:
            print(
                f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV"
            )
            return None
    lambdaInv = np.zeros(nPointsSim)
    for i in range(nPointsSim):
        lambdaInv[i] = calc.lambdaInvMean(
            helper.mmu,
            mChi,
            mu_mu[iSphere + i],
            mu_numu[iSphere + i],
            T[iSphere + i],
            scat=scat,
            iCompton=iCompton,
            mZp=mZp,
            gChi=gChi,
            gL=gL,
            approx=approx,
            oneFermion=False,
            limit="full",
        )
    opacity = np.trapezoid(lambdaInv, R[iSphere : iSphere + nPointsSim])
    return opacity


def getCoupling_TR(
    mZp,
    mChiOvermZp,
    gChiOvergL,
    nPointsSim=30,
    nSim=1,
    guessLower=1e-4,
    guessUpper=1e-3,
    iCompton=0,
    scat=2,
    approx="inv",
    defaultVal=1e0,
    outCheck=False,
    out=True,
):
    # calculate iSphere
    R, T, _, _, _, _, _, _ = helper.unpack(nSim)
    mChi = mChiOvermZp * mZp
    for i in range(nPointsSim):
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim=nSim, out=False)
    if iSphere is None:
        print(
            f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV"
        )
        return None

    # calculate coupling (if iSphere is not None)
    def checkTR(gL):
        opacity = checkModel_TR(
            mZp,
            mChiOvermZp,
            gL,
            gChiOvergL,
            nPointsSim,
            nSim,
            scat=scat,
            iCompton=iCompton,
            approx=approx,
            iSphere=iSphere,
        )
        if outCheck:
            print(
                "gL = {0:.2e} \t checkTR (gL) = {1:.2e}".format(
                    gL, (opacity - 2 / 3) / (opacity + 2 / 3)
                )
            )
        return (opacity - helper.twoThirds) / (opacity + helper.twoThirds)

    try:
        t0 = time.time()
        sol = opt.root_scalar(
            checkTR, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748"
        )
        t1 = time.time()
        print(f"Time consumed: {t1-t0:.2f} s = {(t1-t0)/60:.2f} min")
    except ValueError as e:
        if len(e.args) > 0 and e.args[0].__contains__(
            "a, b must bracket a root"
        ):  # catch only "no solution error"
            print(
                f"ERROR: No solution in the range [{guessLower}, {guessUpper}]. Please adapt the range."
            )
            return defaultVal
        else:
            raise e
    gL = sol.root
    if out:
        print(
            "Coupling with opacity = 2/3 for mZp={0:.1e} MeV: gL = {1:.1e} ({2})".format(
                mZp, gL, approx
            )
        )
    return gL


def main(cluster=True, split=False, iCompton=0, approx="exact", scat=2, nSim=1):
    if cluster:
        CLargument = int(sys.argv[1])  # 0-3 (split=False) or 0-399 (split=True)
        sys.stdout = open(
            f"debug/outTR_{CLargument}_{approx}_{iCompton}.txt", "w", buffering=1
        )
        sys.stderr = open(
            f"debug/errTR_{CLargument}_{approx}_{iCompton}.txt", "w", buffering=1
        )

    prec = 100
    mZpmin = 1e0
    mZpmax = 1e4
    mZp = np.exp(np.linspace(np.log(mZpmin), np.log(mZpmax), prec))

    mChiOvermZp = 1 / 3
    gChiOvergL = 1.0
    nPointsSim = 10  # 50

    guessLower = 1e-8
    guessUpper = 1e-2
    defaultVal = 1e-2

    if cluster:
        if split:
            j = CLargument % prec
            irun = CLargument // prec
        else:
            irun = CLargument
        if irun < 3:
            nSim = 1
            scat = 2**irun
        elif irun == 3:
            nSim = 2
            scat = 2
        else:
            print("ERROR: INVALID CLARGUMENT")
            return None

    print(f"### Calculation for simulation {nSim} and scat={scat}")

    t0 = time.time()
    if not split:
        bounds = np.zeros(prec)
        for j in range(prec):
            bounds[j] = getCoupling_TR(
                mZp[j],
                mChiOvermZp,
                gChiOvergL,
                scat=scat,
                approx=approx,
                nPointsSim=nPointsSim,
                nSim=nSim,
                guessLower=guessLower,
                guessUpper=guessUpper,
                outCheck=True,
                out=True,
                defaultVal=defaultVal,
            )
        dat = np.zeros((prec, 2))
        dat[:, 0] = mZp
        dat[:, 1] = bounds
        np.savetxt(
            f"data/boundTR_{nSim}_{scat}_{iCompton}_{approx}_{nPointsSim}.txt", dat
        )
    else:
        bounds = getCoupling_TR(
            mZp[j],
            mChiOvermZp,
            gChiOvergL,
            scat=scat,
            approx=approx,
            nPointsSim=nPointsSim,
            nSim=nSim,
            guessLower=guessLower,
            guessUpper=guessUpper,
            outCheck=True,
            out=True,
            defaultVal=defaultVal,
        )
        dat = np.zeros((1, 2))
        dat[0, 0] = mZp[j]
        dat[0, 1] = bounds
        path = f"data/boundTR_{nSim}_{scat}_{iCompton}_{approx}_{nPointsSim}.txt"
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

    t1 = time.time()
    print(f"Total time: {t1-t0:.2f} s = {(t1-t0)/60:.2f} min = {(t1-t0)/60**2:.2f} h")

    if cluster:
        sys.stdout.close()
        sys.stderr.close()


# main(cluster=False, iCompton=1, scat=1, approx="exact")

main(cluster=True, split=True, iCompton=1, approx="exact")
