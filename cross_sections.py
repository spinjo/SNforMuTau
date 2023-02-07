import numpy as np
import helper

'''
Hard-coded expressions for the Mandelstam variables and cross sections, to be used in calc.py
- Notation for dimensionless quantities: x=E/T, eps=m^2/T^2,
  y=s/T^2 or y=t/T^2 with Mandelstam variables s and t
- Notation for different processes: 0DM for mu mu -> chi chi;
  1DM for mu chi -> mu chi, 2DM for chi chi -> mu mu,
  DMself_s and DMself_t for different channels of chi chi -> chi chi
- sigmaX is dimensionless a pre-version of the cross section,
  sigmaPropagatorX is the final expression with correct dimensions and
  propagator included for s-channel processes (bad naming!)
'''

### free-streaming
def y_0DM(epsL, x1, x2, cosTh):
    ret=2*(epsL + x1*x2 - (x1**2-epsL)**.5 * (x2**2-epsL)**.5 * cosTh)
    return ret

def sigma0DM_V(epsL, epsDM, y1, LV, LA, DMV, DMA): 
    sig= 1/(12*np.pi *y1) * (y1 - 4*epsDM)**.5/(y1-4*epsL)**.5 * (LA**2 *(4*epsL*(epsDM*(7*DMA**2 - 2*DMV**2) - y1 *(DMA**2 + DMV**2)) + y1*(epsDM*(2*DMV**2-4*DMA**2) + y1 * (DMA**2+DMV**2)))+LV**2*(2*epsL+y1)*(epsDM*(2*DMV**2-4*DMA**2)+y1*(DMA**2+DMV**2)))
    return sig
def sigma0DM_VV(epsL, epsDM, y1):
    return sigmaFS_A_V(epsL, epsDM, y1, 1, 0, 1, 0)
def sigma0DM_LV(epsL, epsDM, y1):
    return sigmaFS_A_V(epsL, epsDM, y1, 1/2, -1/2, 1, 0)
sigmaArr0DM=np.array([sigma0DM_VV, sigma0DM_LV])
nSig=len(sigmaArr0DM)

def sigmaPropagator_0DM(mL, mChi, T, y1, iSigma, mZp=1, gL=1, gChi=1, **kwargs):
    if((y1 - 4*(mL/T)**2) < 0 or (y1-4*(mChi/T)**2)<0):
        return 0.
    gamZp=helper.GamZp(mZp, mChi, gL, gChi, **kwargs)
    propFac=gL**2*gChi**2/( (y1*T**2 -mZp**2)**2 +mZp**2 * gamZp**2)
    sigmaVal = sigmaArrFS[iSigma]( (mL/T)**2, (mChi/T)**2, y1) *T**2 #factor T^2 to get correct dimensions
    ret=propFac * sigmaVal
    return ret

### trapping
def y_1DM(epsL, epsDM, x1, x2, cosTh):
    ret=epsL + epsDM + 2*(x1*x2 - (x1**2-epsL)**.5 * (x2**2-epsDM)**.5 * cosTh)
    return ret
def y_2DM(epsDM, x1, x2, cosTh):
        return y_0DM(epsDM, x1, x2, cosTh) #a bit unnecessary...

def sigma1DM_V(epsL, epsDM, epsZ, y, LV, LA, DMV, DMA):
    sig=1/(8*np.pi) * \
        ( 1/(y*epsZ) /(y**2+y*(-2*epsL-2*epsDM+epsZ)+(epsL-epsDM)**2) *\
            ( LA**2*(2*y**3*(DMA**2+DMV**2)-y**2*(DMA**2+DMV**2)*(4*epsL+4*epsDM-3*epsZ)+2*y*(DMA**2*(epsL**2+10*epsL*epsDM-3*epsL*epsZ+epsDM**2+epsZ**2-3*epsDM*epsZ)+DMV**2*(epsL**2-epsDM*(2*epsL+epsZ)-3*epsL*epsZ+epsDM**2+epsZ**2)) + epsZ*(DMA**2+DMV**2)*(epsL-epsDM)**2) \
              + LV**2*(2*y**3*(DMA**2+DMV**2)-y**2*(DMA**2+DMV**2)*(4*epsL+4*epsDM-3*epsZ)+2*y*(DMA**2*(epsL**2-epsL*(2*epsDM+epsZ)+epsDM**2+epsZ**2-3*epsDM*epsZ)+DMV**2*(epsL**2+2*epsL*epsDM-epsL*epsZ+epsDM**2+epsZ**2-epsDM*epsZ))+epsZ*(DMA**2+DMV**2)*(epsL-epsDM)**2)) \
          -(2*np.log((y**2 -y*(2*epsL+2*epsDM-epsZ)+(epsL-epsDM)**2)/(y*epsZ)) /(y**2-2*y*(epsL+epsDM)+(epsL-epsDM)**2) \
          * (LA**2*(DMV**2*(y-2*epsL+epsZ)+DMA**2*(y-2*epsL-2*epsDM+epsZ))+LV**2*(DMV**2*(y+epsZ)+DMA**2*(y-2*epsDM+epsZ)))))
    return sig
def sigma1DM_VV(epsL, epsDM, epsZ, y1):
    return sigmaTR_A_V(epsL, epsDM, epsZ, y1, 1, 0, 1, 0)
def sigma1DM_LV(epsL, epsDM, epsZ, y1):
    return sigmaTR_A_V(epsL, epsDM, epsZ, y1, 1/2, -1/2, 1, 0)
sigmaArr1DM=np.array([sigma1DM_VV, sigma1DM_LV])

def sigmaPropagator_1DM(mL, mChi, T, y1, iSigma, mZp=1, gL=1, gChi=1): 
    if(y1*T**2 < (mL + mChi)**2):
        return 0.
    sigmaVal=sigmaArrTR[iSigma]( (mL/T)**2, (mChi/T)**2, (mZp/T)**2, y1) *gL**2*gChi**2 /T**2
    return sigmaVal

def sigmaPropagator_2DM(mL, mChi, T, y1, iSigma, **kwargs):
    sigmaVal = sigmaPropagator_0DM(mChi, mL, T, y1, iSigma, **kwargs)
    korrFac = (y1 -4*(mL/T)**2)/(y1-4*(mChi/T)**2) #just different fluxes, everything else is the same as 0DM
    return sigmaVal * korrFac

def sigmaPropagator_DMself_s(mChi, T, y1, iSigma, gChi=1, gL=1, **kwargs):
    sigmaVal = sigmaPropagator_0DM(mChi, mChi, T, y1, iSigma, gChi=gChi, gL=gChi, **kwargs)
    return sigmaVal

def sigmaPropagator_DMself_t(mChi, T, y1, iSigma, gChi=1, gL=1, mZp=1, **kwargs):
    if(y1*T**2 < (2*mChi)**2):
        return 0.
    sigmaVal=sigmaArrTR[iSigma]( (mChi/T)**2, (mChi/T)**2, (mZp/T)**2, y1) *gChi**4 /T**2
    return sigmaVal
