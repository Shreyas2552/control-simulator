"""Analog and digital filter design with Bode and step-response helpers."""
import numpy as np
from scipy import signal
from typing import Tuple

FILTER_TYPES   = ["Low Pass", "High Pass", "Band Pass", "Band Stop"]
FILTER_FAMILIES = ["Butterworth", "Chebyshev Type I", "Bessel"]

_BTYPE = {
    "Low Pass":  "low",
    "High Pass": "high",
    "Band Pass": "band",
    "Band Stop": "bandstop",
}


def design_analog(
    ftype: str, family: str, order: int,
    wc: float, wc2: float = None, ripple_db: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Design an analog (s-domain) filter. wc in rad/s."""
    bt = _BTYPE[ftype]
    Wn = [wc, wc2] if bt in ("band", "bandstop") else wc
    try:
        if family == "Butterworth":
            return signal.butter(order, Wn, btype=bt, analog=True)
        if family == "Chebyshev Type I":
            return signal.cheby1(order, ripple_db, Wn, btype=bt, analog=True)
        if family == "Bessel":
            return signal.bessel(order, Wn, btype=bt, analog=True, norm="phase")
    except Exception:
        pass
    return np.array([1.0]), np.array([1.0])


def design_digital(
    ftype: str, family: str, order: int,
    wn: float, wn2: float = None, ripple_db: float = 1.0, fs: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a digital filter.
    wn is normalised [0, 1] where 1 = Nyquist, OR in Hz when fs is given.
    """
    bt = _BTYPE[ftype]
    Wn = [wn, wn2] if bt in ("band", "bandstop") else wn
    kw = {"fs": fs} if fs else {}
    try:
        if family == "Butterworth":
            return signal.butter(order, Wn, btype=bt, **kw)
        if family == "Chebyshev Type I":
            return signal.cheby1(order, ripple_db, Wn, btype=bt, **kw)
        if family == "Bessel":
            return signal.bessel(order, Wn, btype=bt, norm="phase", **kw)
    except Exception:
        pass
    return np.array([1.0]), np.array([1.0])


def filter_bode(b, a, analog: bool = True, n: int = 500):
    """Returns (omega_or_freq, mag_dB, phase_deg)."""
    if analog:
        w = np.logspace(-2, 4, n)
        w_out, H = signal.freqs(b, a, worN=w)
    else:
        w_out, H = signal.freqz(b, a, worN=n)
    mag_db    = 20 * np.log10(np.abs(H) + 1e-15)
    phase_deg = np.angle(H, deg=True)
    return w_out, mag_db, phase_deg


def filter_step(b, a, analog: bool = True, t_end: float = 10.0, n: int = 1000):
    """Step response of the filter."""
    if analog:
        t = np.linspace(0, t_end, n)
        try:
            tout, y = signal.step((b, a), T=t)
            return tout, y
        except Exception:
            return t, np.zeros(n)
    else:
        t = np.linspace(0, t_end, n)
        u = np.ones(n)
        try:
            y = signal.lfilter(b, a, u)
            return t, y
        except Exception:
            return t, np.zeros(n)
