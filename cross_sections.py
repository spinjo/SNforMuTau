import numpy as np
import time
import helper

'''
Formulae for the cross-sections.
- Internally use dimensionless quantities x = E/T, eps = (m/T)^2, y = s/T^2
- The top-level functions for the cross sections are called sigmaPropagator_X
  and return objects with the correct dimensionality (fill up with powers of T)
- The 5 relevant processes are
  mu mu -> chi chi (s channel, relevant for free-streaming) = 0DM
  chi chi -> mu mu (s channel, relevant for trapping) = 2DM
  mu chi -> mu chi (t channel, relevant for trapping) = 1DM
  chi chi -> chi chi (s channel, relevant for trapping) = DMself_s
  chi chi -> chi chi (t channel, relevant for trapping) = DMself_t
'''

### free-streaming
def y_0DM(epsL, x1, x2, cosTh):
    ret=2*(epsL + x1*x2 - (x1**2-epsL)**.5 * (x2**2-epsL)**.5 * cosTh)
    return ret
def sigmaFS_A_V(epsL, epsDM, y1, LV, LA, DMV, DMA):
    sig= 1/(12*np.pi *y1) * (y1 - 4*epsDM)**.5/(y1-4*epsL)**.5 * (LA**2 *(4*epsL*(epsDM*(7*DMA**2 - 2*DMV**2) - y1 *(DMA**2 + DMV**2)) + y1*(epsDM*(2*DMV**2-4*DMA**2) + y1 * (DMA**2+DMV**2)))+LV**2*(2*epsL+y1)*(epsDM*(2*DMV**2-4*DMA**2)+y1*(DMA**2+DMV**2)))
    return sig
def sigmaFS_A_VV(epsL, epsDM, y1):
    return sigmaFS_A_V(epsL, epsDM, y1, 1, 0, 1, 0)
def sigmaFS_A_LV(epsL, epsDM, y1):
    return sigmaFS_A_V(epsL, epsDM, y1, 1/2, -1/2, 1, 0)
sigmaArrFS=np.array([sigmaFS_A_VV, sigmaFS_A_LV])
nSig=len(sigmaArrFS)

def sigmaPropagator_0DM(mL, mChi, T, y1, iSigma, mZp=1, gL=1, gChi=1, **kwargs):
    if((y1 - 4*(mL/T)**2) < 0 or (y1-4*(mChi/T)**2)<0):
        return 0.
    gamZp=helper.GamZp(mZp, mL, mChi, gL, gChi, **kwargs)
    propFac=gL**2*gChi**2/( (y1*T**2 -mZp**2)**2 +mZp**2 * gamZp**2)
    sigmaVal = sigmaArrFS[iSigma]( (mL/T)**2, (mChi/T)**2, y1) *T**2 #factor T^2 to get correct dimensions
    ret=propFac * sigmaVal
    return ret

### trapping
def y_1DM(epsL, epsDM, x1, x2, cosTh):
    ret=epsL + epsDM + 2*(x1*x2 - (x1**2-epsL)**.5 * (x2**2-epsDM)**.5 * cosTh)
    return ret
def y_2DM(epsDM, x1, x2, cosTh):
        ret=2*(epsDM+x1*x2-(x1**2-epsDM)**.5 *(x2**2-epsDM)**.5 *cosTh)
        return ret

def sigmaTR_A_V(epsL, epsDM, epsZ, y, LV, LA, DMV, DMA):
    sig=1/(8*np.pi) * \
        ( 1/(y*epsZ) /(y**2+y*(-2*epsL-2*epsDM+epsZ)+(epsL-epsDM)**2) *\
            ( LA**2*(2*y**3*(DMA**2+DMV**2)-y**2*(DMA**2+DMV**2)*(4*epsL+4*epsDM-3*epsZ)+2*y*(DMA**2*(epsL**2+10*epsL*epsDM-3*epsL*epsZ+epsDM**2+epsZ**2-3*epsDM*epsZ)+DMV**2*(epsL**2-epsDM*(2*epsL+epsZ)-3*epsL*epsZ+epsDM**2+epsZ**2)) + epsZ*(DMA**2+DMV**2)*(epsL-epsDM)**2) \
              + LV**2*(2*y**3*(DMA**2+DMV**2)-y**2*(DMA**2+DMV**2)*(4*epsL+4*epsDM-3*epsZ)+2*y*(DMA**2*(epsL**2-epsL*(2*epsDM+epsZ)+epsDM**2+epsZ**2-3*epsDM*epsZ)+DMV**2*(epsL**2+2*epsL*epsDM-epsL*epsZ+epsDM**2+epsZ**2-epsDM*epsZ))+epsZ*(DMA**2+DMV**2)*(epsL-epsDM)**2)) \
          -(2*np.log((y**2 -y*(2*epsL+2*epsDM-epsZ)+(epsL-epsDM)**2)/(y*epsZ)) /(y**2-2*y*(epsL+epsDM)+(epsL-epsDM)**2) \
          * (LA**2*(DMV**2*(y-2*epsL+epsZ)+DMA**2*(y-2*epsL-2*epsDM+epsZ))+LV**2*(DMV**2*(y+epsZ)+DMA**2*(y-2*epsDM+epsZ)))))
    return sig
def sigmaTR_A_VV(epsL, epsDM, epsZ, y1):
    return sigmaTR_A_V(epsL, epsDM, epsZ, y1, 1, 0, 1, 0)
def sigmaTR_A_LV(epsL, epsDM, epsZ, y1):
    return sigmaTR_A_V(epsL, epsDM, epsZ, y1, 1/2, -1/2, 1, 0)
sigmaArrTR=np.array([sigmaTR_A_VV, sigmaTR_A_LV])

def sigmaPropagator_1DM(mL, mChi, T, y1, iSigma, mZp=1, gL=1, gChi=1, withNu=False): #withNu irrelevant here, only matters for s channel
    if(y1*T**2 < (mL + mChi)**2):
        return 0.
    sigmaVal=sigmaArrTR[iSigma]( (mL/T)**2, (mChi/T)**2, (mZp/T)**2, y1) *gL**2*gChi**2 /T**2
    return sigmaVal

def sigmaPropagator_2DM(mL, mChi, T, y1, iSigma, **kwargs):
    sigmaVal = sigmaPropagator_0DM(mChi, mL, T, y1, iSigma, **kwargs)
    korrFac = (y1 -4*(mL/T)**2)/(y1-4*(mChi/T)**2) #just the different fluxes, everything else is the same
    return sigmaVal * korrFac

def sigmaPropagator_DMself_s(mChi, T, y1, iSigma, gChi=1, gL=1, **kwargs):
    sigmaVal = sigmaPropagator_0DM(mChi, mChi, T, y1, iSigma, gChi=gChi, gL=gChi, **kwargs)
    return sigmaVal

def sigmaPropagator_DMself_t(mChi, T, y1, iSigma, gChi=1, gL=1, mZp=1, **kwargs):
    if(y1*T**2 < (2*mChi)**2):
        return 0.
    sigmaVal=sigmaArrTR[iSigma]( (mChi/T)**2, (mChi/T)**2, (mZp/T)**2, y1) *gChi**4 /T**2
    return sigmaVal
