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


def checkModel_TR(
    iL,
    mZp,
    gL,
    gChiOvergL,
    nPointsSim=30,
    nSim=1,
    scat=2,
    approx="inv",
    iCompton=0,
    iSphere=None,
):
    gChi = gL * gChiOvergL

    R, T, mu_e, mu_mu, _, _, mu_nue, mu_numu = helper.unpack(nSim)
    if iSphere is None:  # calculate radius of sphere first, if not calculated before
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim=nSim, out=False)
        if iSphere is None:
            print(
                f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV"
            )
            return None
    lambdaInv = np.zeros(nPointsSim)
    mu = [mu_e, mu_mu, mu_nue, mu_numu][iL]
    mL = [helper.me, helper.mmu, 0.0, 0.0][iL]
    iInteraction = 4 if (iL <= 1) else 9
    for i in range(nPointsSim):
        lambdaInv[i] = calc.lambdaInvMean(
            iL,
            mL,
            mChi,
            mu[iSphere + i],
            T[iSphere + i],
            iInteraction,
            scat=scat,
            mZp=mZp,
            gChi=gChi,
            gL=gL,
            approx=approx,
            oneFermion=True,
            iCompton=iCompton,
            limit="full",
        )
    opacity = np.trapezoid(lambdaInv, R[iSphere : iSphere + nPointsSim])
    print(f"Opacity: {opacity:.2e}")
    return opacity


def getCoupling_TR(
    iL,
    mZp,
    gChiOvergL,
    nPointsSim=30,
    nSim=1,
    guessLower=1e-4,
    guessUpper=1e-3,
    scat=2,
    approx="inv",
    defaultVal=1e0,
    iCompton=0,
    outCheck=False,
    out=True,
):
    # calculate iSphere
    R, T, _, _, _, _, _, _ = helper.unpack(nSim)
    for i in range(nPointsSim):
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim=nSim, out=False)
    if iSphere is None:
        print(
            f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV"
        )
        return None

    # calculate coupling (if iSphere is not None)
    def checkTR(gL):
        t0 = time.time()
        opacity = checkModel_TR(
            iL,
            mZp,
            gL,
            gChiOvergL,
            nPointsSim,
            nSim,
            scat=scat,
            approx=approx,
            iSphere=iSphere,
            iCompton=iCompton,
        )
        t1 = time.time()
        # print(opacity, helper.twoThirds)
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
                f"ERROR: No solution at mZp={mZp:.2e} in the range [{guessLower}, {guessUpper}]. Please adapt the range."
            )
            return defaultVal
        else:
            raise e
    # except ZeroDivisionError as e:
    #    print(f"ERROR: Dividing by zero at mZp={mZp:.2e}. This is probably a numerical issue that arises for too large/small couplings.")
    #    return defaultVal
    gL = sol.root
    if out:
        print(
            "Coupling with opacity = 2/3 for mZp={0:.1e} MeV: gL = {1:.1e} ({2})".format(
                mZp, gL, approx
            )
        )
    return gL


def main(cluster=True, split=False, iL=0, scat=2, nSim=1, iCompton=0, approx="exact"):
    if cluster:
        CLargument = int(sys.argv[1])  # want 0-11 (split=False) or 0-1199 (split=True)
        sys.stdout = open(
            f"debug/outTR_{iCompton}_{approx}_{CLargument}.txt", "w", buffering=1
        )
        sys.stderr = open(
            f"debug/errTR_{iCompton}_{approx}_{CLargument}.txt", "w", buffering=1
        )

    prec = 100

    mZpmin = 3e0
    mZpmax = 1e4
    mZp = np.exp(np.linspace(np.log(mZpmin), np.log(mZpmax), prec))

    gChiOvergL = 1.0
    nPointsSim = 50

    guessLower = 1e-8
    guessUpper = 1e0
    defaultVal = 1e0

    if cluster:
        if split:
            j = CLargument % prec
            ibuffer = CLargument // prec
        else:
            ibuffer = CLargument
        irun = ibuffer % 4
        iL = ibuffer // 4  # no need for iL=3
        if irun < 3:
            nSim = 1
            scat = 2**irun
        elif irun == 3:
            nSim = 2
            scat = 2
        else:
            print("ERROR: INVALID CLARGUMENT")
            return None

    print(
        f"### Calculation for lepton {iL}, simulation {nSim} and scat={scat} (approx={approx})"
    )

    if iL == 1 and scat == 1:
        print(
            "Warning: Trying to calculate trapping bound for muons with annihilation only."
            "This is numerically challenging because strongly phase-space suppressed."
        )

    t0 = time.time()
    if not split:
        bounds = np.zeros(prec)
        for j in range(prec):
            print(f"mZp = {mZp[j]:.2e}")
            bounds[j] = getCoupling_TR(
                iL,
                mZp[j],
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
                iCompton=iCompton,
            )
            if j == 0:
                t1 = time.time()
                print(
                    f"Estimate: {(t1-t0)*prec/60:.2f} min = {(t1-t0)*prec/60**2:.2f} h"
                )

        dat = np.zeros((prec, 2))
        dat[:, 0] = mZp
        dat[:, 1] = bounds
        np.savetxt(
            f"data/boundTR_{iL}_{nSim}_{scat}_{iCompton}_{nPointsSim}_{approx}.txt", dat
        )
    else:
        bounds = getCoupling_TR(
            iL,
            mZp[j],
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
            iCompton=iCompton,
        )
        dat = np.zeros((1, 2))
        dat[0, 0] = mZp[j]
        dat[0, 1] = bounds

        # write to file and lock file while writing
        path = f"data/boundTR_{iL}_{nSim}_{scat}_{iCompton}_{nPointsSim}_{approx}.txt"
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


# main(cluster=False, iL=0, scat=1, nSim=1, iCompton=0, approx="exact")

# main(cluster=True, iCompton=0, approx="exact")
# main(cluster=True, iCompton=1, approx="exact")

main(cluster=True, split=True, iCompton=1, approx="exact")
