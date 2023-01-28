import numpy as np
import scipy.special as sp
import scipy.integrate as itg
import scipy.interpolate as itp
import vegas
import time
import cross_sections as cs

'''
Main calculations for the project. The methods dQdR and lambdaInvMean are called by main.py.
- Calculations are performed for each Supernova radius individually
- Integrals: Use scipy.quad for single integrations and vegas for multiple integrations. We
  made the experience that evaluating the multi-dimensional integrals is quite tricky, with
  different methods implemented in scipy.quad and the Mathematica Integrate function giving
  very different results. We find very stable results with the vegas implementation.
- Calculation for the free-streaming case is straight-forward (just the triple-integral)
- Calculation for the trapping case is challenging: The exact calculation is time consuming,
  so we propose two approximations (inv and CM), which are typically off by small O(1)
  factors with inv giving better results than CM. Further, it is not clear how to include
  scattering and chi self-interactions in the trapping regime, so we implemented 3 versions
  scat=1 (only chi chi -> mu mu), scat=2 (add chi mu -> chi mu) and
  scat=4 (add chi chi -> chi chi) to obtain a range in which the "true" result has to lie.
'''

### free-streaming
def dQdR(mL, mChi, mu, T, R, iSigma, nIt=10, nEval=500, **kwargs):
    def intf1(x1, x2, y1, mL, mChi, mu, T, iSigma, statistics=0, **kwargs):
        sigmaVal=cs.sigmaPropagator_0DM(mL, mChi, T, y1, iSigma, **kwargs)
        stat=1/(np.exp(x1+mu/T)+1) *1/(np.exp(x2-mu/T)+1)
        facAverage=(x1+x2) * (x1**2-(mL/T)**2)**.5 * (x2**2-(mL/T)**2)**.5 * stat
        facv=(1-4*(mL/T)**2/y1)**.5 *y1
        ret=facAverage*facv*sigmaVal
        return ret
    def intf2(x):
        z1, z2, cosTh=x
        x1=z1/(1-z1)
        x2=z2/(1-z2)
        preFac=1/(1-z1)**2 * 1/(1-z2)**2
        y1=cs.y_0DM((mL/T)**2, x1, x2, cosTh)
        res=intf1(x1, x2, y1, mL, mChi, mu, T, iSigma, **kwargs)
        return preFac*res
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2( mL/T ), 1.], [f2( mL/T ), 1.], [-1.,1.]])
    res=integ(intf2, nitn=nIt, neval=nEval, alpha=.5)
    fac = res.mean
    ret = 4*np.pi * R**2 * 1/(4*np.pi**4)*T**7 * fac
    return ret

### trapping ###

# main function for trapping #
def lambdaInvMean(*args, approx="inv", scat=2, **kwargs):
    if(scat!=1 and scat!=2 and scat!=4):
        raise ValueError(f"ERROR: Option scat={scat} not implemented. Use one of scat=1,2,4.")
    if(approx=="exact"):
        return lambdaInvMean_exact(*args, **kwargs, scat=scat)  
    elif(approx=="inv"):
        return lambdaInvMean_inv(*args, **kwargs, scat=scat)
    elif(approx=="CM"):
        return lambdaInvMean_CM(*args, **kwargs, scat=scat)
    else:
        raise ValueError(f"ERROR: Approximation approx={approx} not implemented. Use one of approx=inv,CM")

# helper functions
def rosselandWeight(x, xChi):
    weight1 = (1-xChi**2/x**2) * x**4
    weight2 = np.exp(x)/(np.exp(x)+1)**2 if x<1e2 else np.exp(-x)
    return weight1 * weight2

def rosselandNorm(xChi):
    '''Calculate normalization factor in the Rosseland average'''
    normR, _=itg.quad(lambda x: rosselandWeight(x, xChi), xChi, np.inf)
    return normR

# CM approximation #
def lambdaInvMean_CM(mL, mChi, mu, T, iSigma, scat=2, **kwargs):
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegLm=Fdeg(mL, T, mu)
    FdegLp=Fdeg(mL, T, -mu)
    FdegChi=Fdeg(mChi, T, T*0.)

    xChi = mChi/T
    xL = mL/T

    def n(m, T, mu):
        fac, _ = itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return 2/(2*np.pi**2) * T**3 * fac
    nLm = n(mL, T, mu)
    nLp = n(mL, T, -mu)
    nChi = n(mChi, T, T*0.)
    
    def sigma_2DM(x):
        y1 = 2*xChi**2 + 2*x**2
        sigmaVal=cs.sigmaPropagator_2DM(mL, mChi, T, y1, iSigma, **kwargs)
        return sigmaVal
    def sigma_1DM(x):
        y1 = xChi**2 + xL**2 + 2*x**2
        sigmaVal=cs.sigmaPropagator_1DM(mL, mChi, T, y1, iSigma, **kwargs)
        return sigmaVal
    def sigma_DMself_s(x):
        y1 = 2*xChi**2 + 2*x**2
        sigmaVal=cs.sigmaPropagator_DMself_s(mChi, T, y1, iSigma, **kwargs)
        return sigmaVal
    def sigma_DMself_t(x):
        y1 = 2*xChi**2 + 2*x**2
        sigmaVal=cs.sigmaPropagator_DMself_t(mChi, T, y1, iSigma, **kwargs)
        return sigmaVal
    def lambdaFunc(x):
        Gamma = nChi * sigma_2DM(x)*FdegLm*FdegLp * (1-xChi**4/x**4)**.5
        Gamma += nLm * sigma_1DM(x)*FdegLm*FdegChi * (1-xL**2*xChi**2/x**4)**.5 if (scat==2 or scat==4) else 0.
        Gamma += nChi * sigma_DMself_s(x)*FdegChi**2 * (1-xChi**4/x**4)**.5 if scat==4 else 0.
        Gamma += 2 * nChi * sigma_DMself_t(x)*FdegChi**2 * (1-xChi**4/x**4)**.5 if scat==4 else 0.
        lambd = (1-xChi**2/x**2)**.5 / Gamma if Gamma>1e-50 else 1e-50 # sigma returns exactly 0, because area is kinematically forbidden -> no contribution to integral
        return lambd
    def f(x):
        lambd = lambdaFunc(x)
        weight = rosselandWeight(x, xChi)
        ret = lambd*weight
        return ret 
    res, _=itg.quad(f, np.max([xChi, (2*xL**2-xChi**2)**.5]) if (scat==1 and 2*xL*2-xChi**2>0) else xChi,
                    np.inf, epsrel=1e-2)    
    norm= rosselandNorm(xChi)
    lambdaMeanInv = norm/res if res>1e-50 else 1e-50
    return lambdaMeanInv

# inv approximation #
def integ_MuAnn_inv(mL, mChi, mu, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    xL=mL/T
    def lambdaInv(x1, x2, y1, mL, mChi, mu, T, iSigma, **kwargs): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaPropagator_2DM(mL, mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    def f(x):
        z1, z2, cosTh=x
        x1=z1/(1-z1)
        x2=z2/(1-z2)
        preFac=1/(1-z1)**2* 1/(1-z2)**2
        y1=cs.y_2DM((mChi/T)**2, x1, x2, cosTh)
        lambdaInvV=lambdaInv(x1, x2, y1, mL, mChi, mu, T, iSigma, **kwargs)
        weight = rosselandWeight(x2,xChi)
        ret=preFac * lambdaInvV * weight
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2( xChi ), 1.], [f2( xChi ), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    norm = rosselandNorm(xChi)
    return res/norm

def integ_MuScat_inv(mL, mChi, mu, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    xL=mL/T
    def lambdaInv(x1, x2, y1, mL, mChi, mu, T, iSigma, **kwargs): #x1: lepton, x2: chi
        sigmaVal=cs.sigmaPropagator_1DM(mL, mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xL**2)**.5/(np.exp(x1-mu/T)+1) #set muChi=0
        facGamma2=( (y1-xChi**2-xL**2)**2-4*xChi**2*xL**2)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    def f(x):
        z1, z2, cosTh=x
        x1=z1/(1-z1)
        x2=z2/(1-z2)
        y1=cs.y_1DM(xL**2, xChi**2, x1, x2, cosTh)
        preFac=1/(1-z1)**2 * 1/(1-z2)**2
        lambdaInvV = lambdaInv(x1, x2, y1, mL, mChi, mu, T, iSigma, **kwargs)
        weight = rosselandWeight(x2,xChi)
        ret=preFac * lambdaInvV * weight
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2(xL), 1.], [f2(xChi), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    norm = rosselandNorm(xChi)
    return res/norm

def integ_DMAnn_inv(mChi, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    def lambdaInv(x1, x2, y1, mChi, T, iSigma, **kwargs): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaPropagator_DMself_s(mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    def f(x):
        z1, z2, cosTh=x
        x1=z1/(1-z1)
        x2=z2/(1-z2)
        preFac=1/(1-z1)**2* 1/(1-z2)**2
        y1=cs.y_2DM((mChi/T)**2, x1, x2, cosTh)
        lambdaInvV=lambdaInv(x1, x2, y1, mChi, T, iSigma, **kwargs)
        weight= rosselandWeight(x2,xChi)
        ret=preFac * lambdaInvV * weight
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2( xChi ), 1.], [f2( xChi ), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    norm = rosselandNorm(xChi)
    return res/norm

def integ_DMScat_inv(mChi, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    def lambdaInv(x1, x2, y1, mChi, T, iSigma, **kwargs): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaPropagator_DMself_t(mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=( (y1-2*xChi**2)**2-4*xChi**4)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    def f(x):
        z1, z2, cosTh=x
        x1=z1/(1-z1)
        x2=z2/(1-z2)
        y1=cs.y_1DM(xChi**2, xChi**2, x1, x2, cosTh)
        preFac=1/(1-z1)**2 * 1/(1-z2)**2
        lambdaInvV = lambdaInv(x1, x2, y1, mChi, T, iSigma, **kwargs)
        weight= rosselandWeight(x2,xChi)
        ret=preFac * lambdaInvV * weight
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2(xChi), 1.], [f2(xChi), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    norm = rosselandNorm(xChi)
    return res/norm

def lambdaInvMean_inv(mL, mChi, mu, T, iSigma, scat=2, **kwargs): 
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegLm=Fdeg(mL, T, mu)
    FdegLp=Fdeg(mL, T, -mu)
    FdegChi=Fdeg(mChi, T, T*0.)

    lambdaInvMean = integ_MuAnn_inv(mL, mChi, mu, T, iSigma, **kwargs) *FdegLm *FdegLp
    lambdaInvMean += integ_MuScat_inv(mL, mChi, mu, T, iSigma, **kwargs) *FdegLm *FdegChi if (scat==2 or scat==4) else 0.
    lambdaInvMean += integ_DMAnn_inv(mChi, T, iSigma, **kwargs) *FdegChi**2 if scat==4 else 0.
    lambdaInvMean += 2*integ_DMScat_inv(mChi, T, iSigma, **kwargs) *FdegChi**2 if scat==4 else 0.
    return lambdaInvMean

# exact treatment #
def integ_MuAnn_exact(x2, mL, mChi, mu, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    xL=mL/T
    def Gamma(x1, y1, mL, mChi, mu, T, iSigma, **kwargs): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaPropagator_2DM(mL, mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    def f(x):
        z1, cosTh=x
        x1=z1/(1-z1)
        preFac=1/(1-z1)**2
        y1=cs.y_2DM((mChi/T)**2, x1, x2, cosTh)
        GammaV=Gamma(x1, y1, mL, mChi, mu, T, iSigma, **kwargs)
        ret=preFac * GammaV
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2( xChi ), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    return res
def integ_MuScat_exact(x2, mL, mChi, mu, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    xL=mL/T
    def Gamma(x1, y1, mL, mChi, mu, T, iSigma, **kwargs): #x1: lepton, x2: chi
        sigmaVal=cs.sigmaPropagator_1DM(mL, mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xL**2)**.5/(np.exp(x1-mu/T)+1) #set muChi=0
        facGamma2=( (y1-xChi**2-xL**2)**2-4*xChi**2*xL**2)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    def f(x):
        z1, cosTh=x
        x1=z1/(1-z1)
        y1=cs.y_1DM(xL**2, xChi**2, x1, x2, cosTh)
        preFac=1/(1-z1)**2
        GammaV = Gamma(x1, y1, mL, mChi, mu, T, iSigma, **kwargs)
        ret=preFac * GammaV
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2(xL), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    return res
def integ_DMAnn_exact(x2, mChi, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    def Gamma(x1, y1, mChi, T, iSigma, **kwargs): #x1: boring chi, x2: interesting chi  (integrated last)
        sigmaVal=cs.sigmaPropagator_DMself_s(mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    def f(x):
        z1, cosTh=x
        x1=z1/(1-z1)
        preFac=1/(1-z1)**2
        y1=cs.y_2DM((mChi/T)**2, x1, x2, cosTh)
        GammaV=Gamma(x1, y1, mChi, T, iSigma, **kwargs)
        ret=preFac * GammaV
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2( xChi ), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    return res
def integ_DMScat_exact(x2, mChi, T, iSigma, nIt=10, nEval=500, al=.5, **kwargs):
    xChi=mChi/T
    def Gamma(x1, y1, mChi, T, iSigma, **kwargs): #x1: boring chi, x2: interesting chi  (integrated last)
        sigmaVal=cs.sigmaPropagator_DMself_t(mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=( (y1-2*xChi**2)**2-4*xChi**4)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    def f(x):
        z1, cosTh=x
        x1=z1/(1-z1)
        y1=cs.y_1DM(xChi**2, xChi**2, x1, x2, cosTh)
        preFac=1/(1-z1)**2
        GammaV = Gamma(x1, y1, mChi, T, iSigma, **kwargs)
        ret=preFac * GammaV
        return ret
    def f2(x):
        return x/(1+x)
    integ=vegas.Integrator([[f2(xChi), 1.], [-1.,1.]])
    res=integ(f, nitn=nIt, neval=nEval, alpha=al).mean
    return res

def lambdaInvMean_exact(mL, mChi, mu, T, iSigma, scat=2, xMax=1e2, xSteps=10, **kwargs):
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegLm=Fdeg(mL, T, mu)
    FdegLp=Fdeg(mL, T, -mu)
    FdegChi=Fdeg(mChi, T, T*0.)

    xChi=mChi/T
    if(xChi<=1e-10):
        xChi = 1e-10

    #calculate Gamma on array (2 integrations)
    xMin = xChi * (1+1e-3)
    xFirst = np.exp(np.linspace(np.log(xMin), np.log(xMax), xSteps))
    Gamma = np.zeros(xSteps)
    for i in range(xSteps):
        Gamma[i] = integ_MuAnn_exact(xFirst[i], mL, mChi, mu, T, iSigma, **kwargs)
        Gamma[i] += integ_MuScat_exact(xFirst[i], mL, mChi, mu, T, iSigma, **kwargs) if (scat==2 or scat==4) else 0.
        Gamma[i] += integ_DMAnn_exact(xFirst[i], mChi, T, iSigma, **kwargs) if scat==4 else 0.
        Gamma[i] += integ_DMScat_exact(xFirst[i], mChi, T, iSigma, **kwargs) if scat==4 else 0.
        
    #interpolate over array
    GammaFunc = lambda x: np.exp(itp.interp1d(np.log(xFirst), np.log(Gamma), kind="linear")(np.log(x)))

    #final integration
    def f(x):
        Gamma = GammaFunc(x)
        lambd = (1-xChi**2/x**2)**.5 / Gamma if Gamma > 1e-50 else 1e-50
        weighting = rosselandWeight(x, xChi)
        ret = weighting * lambd
        return ret
    
    res, _ = itg.quad(f, xMin, xMax)
    norm = rosselandNorm(xChi)
    lambdaMean = res/norm
    return 1/lambdaMean
