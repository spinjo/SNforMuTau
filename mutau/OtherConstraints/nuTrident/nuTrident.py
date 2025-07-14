import numpy as np
import scipy.interpolate as itp

# bounds do not depend on mChi (because no chi included in the process)
def getNTPbound(mZp):  # mZp is array
    data = np.loadtxt(
        "/media/jonas/geheim/Studium/supernovaMuons/OtherConstraints/nuTrident/NTPdata.csv",
        delimiter=",",
    )
    m = data[:, 0] * 1e3  # in MeV
    g = data[:, 1]
    gOut = np.exp(
        itp.interp1d(np.log(m), np.log(g), kind="linear", fill_value="extrapolate")(
            np.log(mZp)
        )
    )
    return gOut
