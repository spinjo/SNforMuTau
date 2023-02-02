import numpy as np
import scipy.optimize as opt
import time
import helper
import calc
import warnings
warnings.filterwarnings("ignore")

'''
Main file for evaluation of SN limits on L_mu-tau models, including 4 methods:
- checkModel_FS: Compute free-streaming luminosity for a given model specified
  by (mZp, mChi/mZp, gL, gChi/gL) and using SN simulation points in the range
  rangeSim, where 62 is the point with the largest contribution to the integral.
  The parameter nSim controls, which of the SN simulations is used, where nSim=1
  is the coldest and nSim=2 is the hottest.
- checkModel_TR: Compute opacity for a given model specified by the same
  parameters as in checkModel_FS and using SN simulation points in the range
  [iSphere, iSphere+nPointsSim], where the index of the chi sphere iSphere that
  is needed to satisfy the Raffelt bound can be calculated independent of the
  specific processes. Note that it is more efficient to rule out models already
  at the level of the opacity (and not at the level of the trapping luminosity),
  because the step from the opacity to the trapping luminosity depends only on
  the kinematics of the supernova (R, T) and not on the processes in question.
  Further, the parameter approx specifies the approximation used in the calculation
  with possible values approx=exact (takes lots of time), approx=inv (fast and
  decent approximation), approx=CM (fast and acceptable approximation). Finally,
  the parameter scat specifies which processes were included, with possible values
  scat=1 (only chi chi -> mu mu), scat=2 (add chi mu -> chi mu) and
  scat=4 (add chi chi -> chi chi).
- getCoupling_FS: Uses checkModel_FS to find the value of muon coupling gL where
  the free-streaming luminosity satisfies the Raffelt bound. Have to specify
  a range [guessLower, guessUpper] of values for gL to check.
- getCoupling_TR: Similar to getCoupling_FS, but calling checkModel_TR.
'''

#SN point with largest contribution: 62
def checkModel_FS(mZp, mChiOvermZp, gL, gChiOvergL, rangeSim=[50,80], nSim=1, approx="exact", out=True):
    mChi = mChiOvermZp*mZp
    gChi = gL*gChiOvergL
    
    n1, n2 = rangeSim
    N = n2-n1
    dQdR = np.zeros(N)
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    t00 = time.time()
    for i in range(N):
        t0 = time.time()
        dQdR_mu = calc.dQdR(0., mChi, mu_mu[n1+i], T[n1+i], R[n1+i], 0, mZp=mZp, gChi=gChi, gL=gL, approx=approx)
        dQdR_nu = calc.dQdR(0., mChi, mu_numu[n1+i], T[n1+i], R[n1+i], 1, mZp=mZp, gChi=gL*gChiOvergL, gL=gL, approx=approx)
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
            print("Free-streaming luminosity is {0:.2e} MeV^2, this is LARGER than the Raffelt bound {1:.2e} MeV^2.".format(Q, helper.getQbound(nSim)))
        else:
            print(f"### Model not excluded by free-streaming ###")
            print("Free-streaming luminosity is {0:.2e} MeV^2, this is SMALLER than the Raffelt bound {1:.2e} MeV^2.".format(Q, helper.getQbound(nSim)))
        print("NOTE: The contributions to the luminosity where calculated using only a limited number of points in the SN simulation, more explicitly points in the range rangeSim=[{0}, {1}].".format(n1, n2))
        print("Using more points INCREASES the free-streaming luminosity, moving the couplings where the Raffelt bound is met to SMALLER values.")
    return Q

def checkModel_TR(mZp, mChiOvermZp, gL, gChiOvergL, nPointsSim=30, nSim=1, scat=2, approx="inv", iSphere=None, out=True):
    mChi = mChiOvermZp*mZp
    gChi = gL*gChiOvergL
    
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    if iSphere is None:
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim, out=False)
        if(iSphere is None):
            print(f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV")
            return None
    lambdaInv = np.zeros(nPointsSim)
    t00 = time.time()
    for i in range(nPointsSim):
        t0 = time.time()
        lambdaInv[i] = calc.lambdaInvMean(helper.mmu, mChi, mu_mu[iSphere+i], mu_numu[iSphere+i], T[iSphere+i], scat=scat, mZp=mZp, gChi=gChi, gL=gL, approx=approx)
        t1 = time.time()
        if i==0 and out:
            print(f"Estimate: {(t1-t0)*nPointsSim:.2f} s = {(t1-t0)/60*nPointsSim:.2f} min")  
    t01 = time.time()
    if out:
        print(f"Total time used: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min")          
    opacity = np.trapz(lambdaInv, R[iSphere:iSphere+nPointsSim])

    if(out):
        if(opacity < 2/3):
            print(f"### Model excluded by Boltzmann trapping ({approx} approximation) ###\nOpacity at the critical radius where the corresponding Boltzmann luminosity satisfies the Raffelt bound is {opacity:.2e}.")
            print("This is SMALLER than the value 2/3 that defines the critical radius, resulting in a SMALLER radius of the DM sphere and therefore a Boltzmann luminosity LARGER than the Raffelt bound")
        else:
            print(f"### Model not excluded by Boltzmann trapping ({approx} approximation) ###\nOpacity at the critical radius where the corresponding Boltzmann luminosity satisfies the Raffelt bound is {opacity:.2e}.")
            print("This is LARGER than the value 2/3 that defines the critical radius, resulting in a LARGER radius of the DM sphere and therefore a Boltzmann luminosity SMALLER than the Raffelt bound")
        print("NOTE: The contributions to the opacity where calculated using only a limited number of points in the SN simulation, more explicitly the first nSimPoints={0} points after the critical point {1} where the chi sphere has to end to satisfy the Raffelt bound.".format(nPointsSim, iSphere))
        print("Using more points INCREASES the opacity, moving the couplings where the Raffelt bound is met to SMALLER values.")
    return opacity

def getCoupling_FS(mZp, mChiOvermZp, gChiOvergL, rangeSim=[50,80], nSim=1, guessLower=1e-5, guessUpper=1e-3, approx="exact", outCheck=False, out=True):    
    def checkFS(gL):
        Q = checkModel_FS(mZp, mChiOvermZp, gL, gChiOvergL, rangeSim, nSim, approx=approx, out=False)
        Qbound = helper.getQbound(nSim)
        if(outCheck):
            print("gL = {0:.2e} \t checkFS (gL) = {1:.2e}".format(gL, (Q - Qbound)/(Q + Qbound)))
        return (Q - Qbound)/(Q + Qbound)
    try:
        sol = opt.root_scalar(checkFS, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748")
    except ValueError as e:
        if len(e.args)>0 and e.args[0].__contains__("a, b must bracket a root"): #catch only "no solution error"
            print(f"ERROR: No solution in the range [{guessLower}, {guessUpper}]. Please adapt the range.")
            return defaultVal
        else:
            raise e
    gL = sol.root
    if out:
        print("Coupling with FSLumi = RaffeltLumi: gL = {0:.1e}".format(gL))
    return gL

def getCoupling_TR(mZp, mChiOvermZp, gChiOvergL, nPointsSim=30, nSim=1, guessLower=1e-5, guessUpper=1e0, scat=2, approx="inv", outCheck=False, out=True):   
    # calculate iSphere
    R, T, _, _, _, _, _, _ = helper.unpack(nSim)
    mChi = mChiOvermZp*mZp
    for i in range(nPointsSim):
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim, out=False) 
    if(iSphere is None):
        print(f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV")
        return None
    def checkTR(gL):
        opacity = checkModel_TR(mZp, mChiOvermZp, gL, gChiOvergL, nPointsSim, nSim, scat=scat, approx=approx, iSphere=iSphere, out=False)
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
#checkModel_FS(3000., 0., 1e-5, 1., rangeSim=[60,65], approx="exact")
#checkModel_FS(3000., 0., 1e-5, 1., rangeSim=[60,65], approx="CM")

getCoupling_FS(3., 1/3, 1, rangeSim=[55,70], approx="exact", outCheck=True)
getCoupling_FS(3., 1/3, 1, rangeSim=[55,70], approx="CM", outCheck=True)
'''
# Examples

checkModel_TR(3., 1/3, 1e-3, 1., scat=2, nPointsSim=2, approx="CM")
checkModel_TR(3., 1/3, 1e-3, 1., scat=2, nPointsSim=2, approx="inv")
checkModel_TR(3., 1/3, 1e-3, 1., scat=2, nPointsSim=2, approx="exact")

getCoupling_FS(3., 1/3, 1, rangeSim=[55,70], outCheck=True)
getCoupling_TR(3., 1/3, 1, scat=2, approx="CM", nPointsSim=2)
getCoupling_TR(3., 1/3, 1, scat=2, approx="inv", nPointsSim=2)
getCoupling_TR(3., 1/3, 1, scat=2 approx="exact", nPointsSim=2)
'''
