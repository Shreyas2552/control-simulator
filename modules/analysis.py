"""Core control-system analysis: TF algebra, responses, Bode, root locus."""
import numpy as np
from scipy import signal
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# Transfer-function algebra
# ---------------------------------------------------------------------------

def build_ol_cl(plant_num, plant_den, ctrl_num, ctrl_den):
    """
    Open-loop  L  = C·G   (no feedback)
    Closed-loop CL = L / (1 + L)   (unity negative feedback)

    Returns: ol_num, ol_den, cl_num, cl_den  (numpy arrays)
    """
    pn = np.asarray(plant_num, float)
    pd = np.asarray(plant_den, float)
    cn = np.asarray(ctrl_num, float)
    cd = np.asarray(ctrl_den, float)

    ol_num = np.polymul(cn, pn)
    ol_den = np.polymul(cd, pd)

    cl_num = ol_num.copy()
    cl_den = np.polyadd(ol_den, ol_num)   # polyadd handles different lengths

    return ol_num, ol_den, cl_num, cl_den


# ---------------------------------------------------------------------------
# Time-domain responses
# ---------------------------------------------------------------------------

def step_response(num, den, t_end: float, n: int = 1000,
                  discrete: bool = False, Ts: Optional[float] = None):
    """Step response of a transfer function."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)

    if discrete and Ts:
        t = np.arange(0, t_end, Ts)
        sys_d = signal.dlti(num, den, dt=Ts)
        tout, y = signal.dstep(sys_d, n=len(t))
        return tout, y[0].flatten()

    t = np.linspace(0, t_end, n)
    tout, y = signal.step((num, den), T=t)
    return tout, y


def ramp_response(num, den, t_end: float, n: int = 1000):
    """Ramp response = step response of G(s)/s."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    ramp_den = np.polymul(den, [1.0, 0.0])
    t = np.linspace(0, t_end, n)
    try:
        tout, y = signal.step((num, ramp_den), T=t)
        return tout, y
    except Exception:
        return t, np.zeros(n)


def control_signal_step(plant_num, plant_den, ctrl_num, ctrl_den, t_end, n=1000):
    """
    TF from reference R to control signal U:
        U/R = C(s) / (1 + C·G)
    num_ru = C_num · G_den
    den_ru = C_den · G_den + C_num · G_num  (= closed-loop denominator)
    """
    cn = np.asarray(ctrl_num, float)
    cd = np.asarray(ctrl_den, float)
    pn = np.asarray(plant_num, float)
    pd = np.asarray(plant_den, float)

    num_ru = np.polymul(cn, pd)
    den_ru = np.polyadd(np.polymul(cd, pd), np.polymul(cn, pn))

    t = np.linspace(0, t_end, n)
    try:
        tout, u = signal.step((num_ru, den_ru), T=t)
        return tout, u
    except Exception:
        return t, np.zeros(n)


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------

def _auto_omega(num, den, n):
    """Choose a sensible log-frequency range from pole/zero locations."""
    try:
        freqs = [abs(p) for p in np.roots(den) if abs(p) > 1e-6]
        if len(num) > 1:
            freqs += [abs(z) for z in np.roots(num) if abs(z) > 1e-6]
        if freqs:
            lo = max(np.log10(min(freqs)) - 2, -3)
            hi = min(np.log10(max(freqs)) + 2,  4)
        else:
            lo, hi = -2, 2
    except Exception:
        lo, hi = -2, 2
    return np.logspace(lo, hi, n)


def bode_data(num, den, n: int = 500, discrete: bool = False, Ts: Optional[float] = None):
    """Bode data: (omega, mag_dB, phase_deg)."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)

    if discrete and Ts:
        w, H = signal.freqz(num, den, worN=n)
        omega = w / Ts            # rad/sample → rad/s
        mag_db = 20 * np.log10(np.abs(H) + 1e-15)
        phase_deg = np.angle(H, deg=True)
        return omega, mag_db, phase_deg

    omega = _auto_omega(num, den, n)
    w_out, mag_db, phase_deg = signal.bode((num, den), w=omega)
    return w_out, mag_db, phase_deg


def stability_margins(ol_num, ol_den, discrete: bool = False, Ts: Optional[float] = None) -> Dict:
    """
    Returns dict with keys: gm_db, pm_deg, wgc (gain-crossover), wpc (phase-crossover).
    Infinite margin returned as np.inf when the crossover doesn't exist.
    """
    try:
        w, mag, phase = bode_data(ol_num, ol_den, n=8000, discrete=discrete, Ts=Ts)
        phase_u = np.unwrap(np.deg2rad(phase)) * (180 / np.pi)   # unwrapped degrees

        result = dict(gm_db=np.inf, pm_deg=np.inf, wgc=None, wpc=None)

        # Gain crossover: mag crosses 0 dB
        for i in np.where(np.diff(np.sign(mag)))[0]:
            frac = -mag[i] / (mag[i+1] - mag[i] + 1e-30)
            wgc  = w[i] + frac * (w[i+1] - w[i])
            ph   = phase_u[i] + frac * (phase_u[i+1] - phase_u[i])
            if result["wgc"] is None:
                result["wgc"]    = float(wgc)
                result["pm_deg"] = float(180 + ph)

        # Phase crossover: phase crosses −180°
        shifted = phase_u + 180
        for i in np.where(np.diff(np.sign(shifted)))[0]:
            if shifted[i] <= 0 <= shifted[i+1]:          # crossing upward
                frac = -shifted[i] / (shifted[i+1] - shifted[i] + 1e-30)
                wpc  = w[i] + frac * (w[i+1] - w[i])
                mg   = mag[i] + frac * (mag[i+1] - mag[i])
                if result["wpc"] is None:
                    result["wpc"]   = float(wpc)
                    result["gm_db"] = float(-mg)

        return result
    except Exception:
        return dict(gm_db=None, pm_deg=None, wgc=None, wpc=None)


# ---------------------------------------------------------------------------
# Root locus
# ---------------------------------------------------------------------------

def root_locus_data(ol_num, ol_den, n_gains: int = 400):
    """
    Compute root locus by sweeping gain K.
    Returns: (locus list of (K, roots_array), ol_poles, ol_zeros)
    """
    num = np.asarray(ol_num, float)
    den = np.asarray(ol_den, float)

    # Pad num to same length as den for char-poly arithmetic
    num_p = np.zeros(len(den))
    num_p[-len(num):] = num

    k_arr = np.concatenate([
        np.logspace(-3, 0, n_gains // 2),
        np.logspace(0,  4, n_gains // 2),
    ])

    locus: List[Tuple] = []
    prev  = None

    for k in k_arr:
        cp = den + k * num_p
        try:
            rts = np.roots(cp)
            if prev is not None and len(rts) == len(prev):
                # Nearest-neighbour assignment for smooth branch continuity
                avail = list(range(len(rts)))
                ordered = []
                for pr in prev:
                    dists = [(abs(pr - rts[j]), j) for j in avail]
                    _, best = min(dists)
                    ordered.append(rts[best])
                    avail.remove(best)
                rts = np.array(ordered)
            prev = rts
            locus.append((float(k), rts))
        except Exception:
            continue

    ol_poles = np.roots(den) if len(den) > 1 else np.array([], dtype=complex)
    ol_zeros = np.roots(num) if len(num) > 1 else np.array([], dtype=complex)

    return locus, ol_poles, ol_zeros


# ---------------------------------------------------------------------------
# Closed-loop pole analysis
# ---------------------------------------------------------------------------

def cl_pole_analysis(cl_den) -> List[Dict]:
    """
    Analyse closed-loop poles.
    Returns list of dicts with: pole, kind, real, imag, wn, zeta, wd,
                                 stable, settling_time, overshoot_pct.
    """
    poles = np.roots(np.asarray(cl_den, float))
    used  = set()
    rows  = []

    for i, p in enumerate(poles):
        if i in used:
            continue

        if abs(p.imag) > 1e-4:
            # Find conjugate partner
            for j, q in enumerate(poles):
                if j != i and j not in used and abs(p - q.conj()) < 1e-4:
                    used.add(j)
                    break

            wn   = abs(p)
            zeta = -p.real / (wn + 1e-30)
            wd   = abs(p.imag)
            os_  = (100 * np.exp(-np.pi * zeta / np.sqrt(max(1 - zeta**2, 1e-12)))
                    if 0 < zeta < 1 else 0.0)
            ts   = 4 / (abs(p.real) + 1e-30) if p.real < 0 else np.inf

            rows.append(dict(
                pole=f"{p.real:+.4f} ± {abs(p.imag):.4f}j",
                kind="Complex Pair",
                real=float(p.real), imag=float(abs(p.imag)),
                wn=float(wn), zeta=float(zeta), wd=float(wd),
                stable=bool(p.real < 0),
                settling_time=float(ts),
                overshoot_pct=float(os_),
            ))
        else:
            ts = 4 / abs(p.real) if p.real != 0 and p.real < 0 else np.inf
            rows.append(dict(
                pole=f"{p.real:+.4f}",
                kind="Real",
                real=float(p.real), imag=0.0,
                wn=float(abs(p.real)), zeta=1.0, wd=0.0,
                stable=bool(p.real < 0),
                settling_time=float(ts),
                overshoot_pct=0.0,
            ))
        used.add(i)

    return rows


# ---------------------------------------------------------------------------
# Time-domain performance metrics
# ---------------------------------------------------------------------------

def performance_metrics(t, y, ref: float = 1.0) -> Dict:
    if len(y) == 0 or abs(ref) < 1e-12:
        return {}

    m: Dict = {}
    m["Steady-State Value"] = float(y[-1])
    m["Steady-State Error"] = float(ref - y[-1])
    m["Peak Value"]         = float(np.max(y))
    m["Overshoot (%)"]      = float(max(0, (np.max(y) - ref) / ref * 100))
    m["Peak Time (s)"]      = float(t[np.argmax(y)])

    try:
        i10 = np.where(y >= 0.1 * ref)[0][0]
        i90 = np.where(y >= 0.9 * ref)[0][0]
        m["Rise Time (s)"] = float(t[i90] - t[i10])
    except IndexError:
        m["Rise Time (s)"] = None

    try:
        outside = np.where(np.abs(y - ref) > 0.02 * abs(ref))[0]
        m["Settling Time (2%) (s)"] = float(t[outside[-1]]) if len(outside) else 0.0
    except Exception:
        m["Settling Time (2%) (s)"] = None

    return m
