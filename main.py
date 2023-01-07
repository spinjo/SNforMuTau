import numpy as np
import scipy.optimize as opt

import os, sys, time
scriptPath=os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
import helper
import calc

import warnings
warnings.filterwarnings("ignore")

#SN point with largest contribution: 62
def checkModel_FS(mZp, mChiOvermZp, gL, gChiOvergL, rangeSim=[50,80], nSim=1, out=True):
    mChi = mChiOvermZp*mZp
    gChi = gL*gChiOvergL
    
    n1, n2 = rangeSim
    N = n2-n1
    dQdR = np.zeros(N)
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    t00 = time.time()
    for i in range(N):
        t0 = time.time()
        dQdR_mu = calc.dQdR(helper.mmu, mChi, mu_mu[n1+i], T[n1+i], R[n1+i], 0, mZp=mZp, gChi=gChi, gL=gL, withNu=True)
        dQdR_nu = calc.dQdR(0., mChi, mu_numu[n1+i], T[n1+i], R[n1+i], 1, mZp=mZp, gChi=gL*gChiOvergL, gL=gL, withNu=True)
        dQdR[i] = dQdR_mu + 2*dQdR_nu #factor 2 because need both nu_mu and nu_tau
        t1 = time.time()
        if i==0 and out:
            print(f"Estimate: {(t1-t0)*N:.2f} s = {(t1-t0)/60*N:.2f} min")
    t01 = time.time()
    if out:
        print(f"Total time used: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min")
    Q = np.trapz(dQdR, x=R[n1:n2])

    if(out):
        if(Q > helper.Qbound):
            print(f"### Model excluded by free-streaming ###")
            print("Free-streaming luminosity is {0:.2e} MeV^2, this is LARGER than the Raffelt bound {1:.2e} MeV^2.".format(Q, helper.Qbound))
        else:
            print(f"### Model not excluded by free-streaming ###")
            print("Free-streaming luminosity is {0:.2e} MeV^2, this is SMALLER than the Raffelt bound {1:.2e} MeV^2.".format(Q, helper.Qbound))
        print("NOTE: The contributions to the luminosity where calculated using only a limited number of points in the SN simulation, more explicitly points in the range rangeSim=[{0}, {1}].".format(n1, n2))
        print("Using more points INCREASES the free-streaming luminosity, moving the couplings where the Raffelt bound is met to SMALLER values.")
    return Q

def checkModel_TR(mZp, mChiOvermZp, gL, gChiOvergL, nPointsSim=30, nSim=1, scat=2, approx="inv", out=True):
    mChi = mChiOvermZp*mZp
    gChi = gL*gChiOvergL
    
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    for i in range(nPointsSim):
        iSphere = helper.getRadiusSphere(mChi, R, T, out=False)
    if(iSphere is None):
        return None
    lambdaInv = np.zeros(nPointsSim)
    t00 = time.time()
    for i in range(nPointsSim):
        t0 = time.time()
        lambdaInv_mu = calc.lambdaInvMean(helper.mmu, mChi, mu_mu[iSphere+i], T[iSphere+i], 0, scat=scat, mZp=mZp, gChi=gChi, gL=gL, withNu=True, approx=approx)
        lambdaInv_nu = calc.lambdaInvMean(0., mChi, mu_numu[iSphere+i], T[iSphere+i], 1, scat=scat, mZp=mZp, gChi=gChi, gL=gL, withNu=True, approx=approx)
        lambdaInv[i] = lambdaInv_mu + 2*lambdaInv_nu
        t1 = time.time()
        if i==0 and out:
            print(f"Estimate: {(t1-t0)*nPointsSim:.2f} s = {(t1-t0)/60*nPointsSim:.2f} min")  
    t01 = time.time()
    if out:
        print(f"Total time used: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min")          
    opacity = np.trapz(lambdaInv, R[iSphere:iSphere+nPointsSim])

    if(out):
        if(opacity < 2/3):
            print(f"### Model excluded by Boltzmann trapping ({approx} approximation) ###\nOpacity at the critical radius where the corresponding Boltzmann luminosity satisfies the Raffelt bound is {0:.2e}.".format(opacity))
            print("This is SMALLER than the value 2/3 that defines the critical radius, resulting in a SMALLER radius of the DM sphere and therefore a Boltzmann luminosity LARGER than the Raffelt bound")
        else:
            print(f"### Model not excluded by Boltzmann trapping ({approx} approximation) ###\nOpacity at the critical radius where the corresponding Boltzmann luminosity satisfies the Raffelt bound is {0:.2e}.".format(opacity))
            print("This is LARGER than the value 2/3 that defines the critical radius, resulting in a LARGER radius of the DM sphere and therefore a Boltzmann luminosity SMALLER than the Raffelt bound")
        print("NOTE: The contributions to the opacity where calculated using only a limited number of points in the SN simulation, more explicitly the first nSimPoints={0} points after the critical point {1} where the chi sphere has to end to satisfy the Raffelt bound.".format(nPointsSim, iSphere))
        print("Using more points INCREASES the opacity, moving the couplings where the Raffelt bound is met to SMALLER values.")
    return opacity

def getCoupling_FS(mZp, mChiOvermZp, gChiOvergL, rangeSim=[50,80], nSim=1, guessLower=1e-5, guessUpper=1e-3, outCheck=False, out=True):    
    def checkFS(gL):
        Q = checkModel_FS(mZp, mChiOvermZp, gL, gChiOvergL, rangeSim, nSim, out=False)
        if(outCheck):
            print("gL = {0:.2e} \t checkFS (gL) = {1:.2e}".format(gL, (Q - helper.Qbound)/(Q + helper.Qbound)))
        return (Q - helper.Qbound)/(Q + helper.Qbound)
    try:
        sol = opt.root_scalar(checkFS, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748")
    except ValueError:
        print(f"ERROR: No solution in the range [{guessLower}, {guessUpper}]. Please adapt the range.")
        return None
    gL = sol.root
    if out:
        print("Coupling with FSLumi = RaffeltLumi: gL = {0:.1e}".format(gL))
    return gL

def getCoupling_TR(mZp, mChiOvermZp, gChiOvergL, nPointsSim=30, nSim=1, guessLower=1e-10, guessUpper=1e0, scat=2, approx="inv", outCheck=False, out=True):   
    def checkTR(gL):
        opacity = checkModel_TR(mZp, mChiOvermZp, gL, gChiOvergL, nPointsSim, nSim, scat=scat, approx=approx, out=False)
        if(outCheck):
            print("gL = {0:.2e} \t checkTR (gL) = {1:.2e}".format(gL, (opacity - 2/3)/(opacity + 2/3)))
        return (opacity - helper.twoThirds)/(opacity + helper.twoThirds)
    try:
        sol = opt.root_scalar(checkTR, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748")
    except ValueError:
        print(f"ERROR: No solution in the range [{guessLower}, {guessUpper}]. Please adapt the range.")
        return None
    gL = sol.root
    if out:
        print("Coupling with opacity = 2/3: gL = {0:.1e} ({1})".format(gL, approx))
    return gL

# Examples
'''
checkModel_FS(3., 1/3, 1e-5, 1.)
checkModel_TR(3., 1/3, 1e-3, 1., scat=4, nPointsSim=2, approx="inv")
checkModel_TR(3., 1/3, 1e-3, 1., scat=4, nPointsSim=2, approx="CM")
checkModel_TR(3., 1/3, 1e-3, 1., scat=4, nPointsSim=2, approx="exact")

getCoupling_FS(3., 1/3, 1, rangeSim=[55,70], outCheck=True)
getCoupling_TR(3., 1/3, 1, scat=1, approx="inv", nPointsSim=2)
getCoupling_TR(3., 1/3, 1, scat=1, approx="CM", nPointsSim=2)
getCoupling_TR(3., 1/3, 1, scat=1, approx="exact", nPointsSim=2)
'''
