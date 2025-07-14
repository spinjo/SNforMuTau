import numpy as np
import scipy.interpolate as itp


def preWork(valFS, valTR, approach=0, cut=0):
    # approach 0: TR low, FS high; approach 1: TR high, FS low
    # cut 0: right; cut 1: low
    if cut == 0:
        notDone = True
        for i in range(len(valFS)):
            if valTR[i] == None and notDone:
                valTR[i] = valFS[i]
                continue
            if approach == 0 and valFS[i] < valTR[i] and notDone:
                valFS[i] = (valFS[i - 1] + valTR[i - 1]) / 2
                valTR[i] = (valFS[i - 1] + valTR[i - 1]) / 2
                notDone = False
                continue
            elif approach == 1 and valFS[i] > valTR[i] and notDone:
                valFS[i] = (valFS[i - 1] + valTR[i - 1]) / 2
                valTR[i] = (valFS[i - 1] + valTR[i - 1]) / 2
                notDone = False
                continue
            if not notDone:
                valFS[i] = None
                valTR[i] = None
    elif cut == 1:
        for i in range(len(valFS)):
            if valTR[i] <= valFS[i] and approach == 1:
                valTR[i] = None
                valFS[i] = None
    return valFS, valTR


def improve(mDM, x, valFS, valTR, approach=0, interp=True):
    if interp:
        nans = np.invert(np.isnan(valTR))
        # valFS=itp.interp1d(mDM, valFS, kind="linear", fill_value="extrapolate")(x) #different ways to interpolate good
        # valTR=itp.interp1d(mDM[nans], valTR[nans], kind="linear", fill_value="extrapolate")(x)
        valFS = np.exp(
            itp.interp1d(
                np.log(mDM), np.log(valFS), kind="linear", fill_value="extrapolate"
            )(np.log(x))
        )
        valTR = np.exp(
            itp.interp1d(
                np.log(mDM[nans]),
                np.log(valTR[nans]),
                kind="linear",
                fill_value="extrapolate",
            )(np.log(x))
        )
        # valFS=np.exp(itp.interp1d(mDM, np.log(valFS), kind="linear", fill_value="extrapolate")(x))
        # valTR=np.exp(itp.interp1d(mDM[nans], np.log(valTR[nans]), kind="linear", fill_value="extrapolate")(x))
    valFS, valTR = preWork(valFS, valTR, approach)
    return valFS, valTR


def getMin(tr, fs):  # works for both min and max (has bad name!)
    for i in range(len(tr)):
        if np.isnan(tr[i]) and not np.isnan(fs[i]):
            tr[i] = fs[i]
    return tr


def loadSorted(file, iStart=0):
    m = np.loadtxt(file)[iStart:, 0]
    print(file, np.shape(m))
    y = np.loadtxt(file)[iStart:, 1]
    idx = np.argsort(m)
    return m[idx], y[idx]
