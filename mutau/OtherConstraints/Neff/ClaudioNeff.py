import numpy as np
import scipy.interpolate as itp


def getNeff(mZp):
    dataLower = np.loadtxt("OtherConstraints/Neff/Neff32.csv", delimiter=",")
    dataUpper = np.loadtxt("OtherConstraints/Neff/Neff35.csv", delimiter=",")
    gLower = np.exp(
        itp.interp1d(
            np.log(dataLower[:, 0]),
            np.log(dataLower[:, 1]),
            kind="linear",
            fill_value="extrapolate",
        )(np.log(mZp))
    )
    gUpper = np.exp(
        itp.interp1d(
            np.log(dataUpper[:, 0]),
            np.log(dataUpper[:, 1]),
            kind="linear",
            fill_value="extrapolate",
        )(np.log(mZp))
    )
    return gLower, gUpper
