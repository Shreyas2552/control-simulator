"""
Advanced PID control strategies: anti-windup, gain scheduling, feedforward.

All simulations use Euler integration (dt=0.005 s) so saturation, scheduling,
and additive disturbances can be applied step-by-step — scipy.signal.step
cannot handle these nonlinear / time-varying effects.
"""
import numpy as np
from scipy.signal import tf2ss
from typing import Tuple, Dict

_DT = 0.005   # 200 Hz integration step


def _plant_ss(num, den):
    """Convert TF to state-space; return (A, B, C, D) as 2-D numpy arrays."""
    A, B, C, D = tf2ss(np.asarray(num, float), np.asarray(den, float))
    return A, B.reshape(-1, 1), C.reshape(1, -1), float(D.flat[0])


# ── Anti-integral windup ────────────────────────────────────────────────────

def simulate_antiwindup(
    plant_num, plant_den,
    Kp: float, Ki: float, Kd: float, N: float,
    u_min: float, u_max: float, Tt: float,
    ref: float, t_end: float,
) -> Dict:
    """
    Simulate PID with and without back-calculation anti-windup.

    Back-calculation: dxi/dt = Ki·e + (u_sat − u_unsat) / Tt
    Tt is the tracking time constant (typical: sqrt(Kd/Ki) or Kp/Ki).

    Returns dict with keys 'no_aw' and 'aw', each being (y, u, xi) arrays,
    plus shared time vector 't'.
    """
    A, B, C, D = _plant_ss(plant_num, plant_den)
    n_p = A.shape[0]
    t = np.arange(0.0, t_end, _DT)
    N_steps = len(t)

    out = {}
    for mode in ('no_aw', 'aw'):
        x_p = np.zeros(n_p)
        xi  = 0.0   # integral accumulator (u_i = xi directly, Ki folded in below)
        xd  = 0.0   # derivative filter state
        y   = np.zeros(N_steps)
        u_h = np.zeros(N_steps)
        xi_h = np.zeros(N_steps)

        for k in range(N_steps - 1):
            y_k = float((C @ x_p).flat[0]) + D * u_h[k - 1] if k > 0 else 0.0
            y[k] = y_k
            e = ref - y_k

            # Derivative filter: xd tracks N·e
            der = Kd * N * (e - xd)
            u_unsat = Kp * e + xi + der
            u_sat   = float(np.clip(u_unsat, u_min, u_max))

            u_h[k]  = u_sat
            xi_h[k] = xi

            # Euler update — derivative filter
            xd += _DT * N * (e - xd)
            # Euler update — integrator
            if mode == 'aw':
                xi += _DT * (Ki * e + (u_sat - u_unsat) / max(Tt, 1e-6))
            else:
                xi += _DT * Ki * e
            # Euler update — plant
            x_p += _DT * ((A @ x_p).flatten() + (B * u_sat).flatten())

        y[-1]  = float((C @ x_p).flat[0])
        xi_h[-1] = xi
        out[mode] = (y, u_h, xi_h)

    return {'t': t, 'no_aw': out['no_aw'], 'aw': out['aw']}


# ── Gain scheduling ─────────────────────────────────────────────────────────

def simulate_gain_scheduling(
    tau: float,
    K_regions: Tuple[float, float],   # plant gain at low and high output
    y_breakpoints: Tuple[float, float],
    Kp_fixed: float, Ki_fixed: float, Kd_fixed: float, N: float,
    ref: float, t_end: float,
) -> Dict:
    """
    Demonstrate gain scheduling on a first-order plant whose gain varies with output.

    Plant: dy/dt = (-y + K(y)·u) / tau    (output-dependent gain)
    K(y) is linearly interpolated between K_regions at y_breakpoints.

    Two controllers are simulated:
      'fixed'     — Kp/Ki/Kd fixed (tuned for K_regions[0])
      'scheduled' — Kp/Ki/Kd scaled inversely with K(y) to maintain loop gain

    Returns dict with 't', 'fixed', 'scheduled', 'K_history' (scheduled run).
    """
    t = np.arange(0.0, t_end, _DT)
    N_steps = len(t)
    K_lo, K_hi = K_regions
    y_lo, y_hi = y_breakpoints

    def plant_gain(y_val):
        alpha = np.clip((y_val - y_lo) / (y_hi - y_lo + 1e-9), 0.0, 1.0)
        return K_lo + alpha * (K_hi - K_lo)

    out = {}
    K_hist = np.zeros(N_steps)

    for mode in ('fixed', 'scheduled'):
        y_p = 0.0   # plant output (state for 1st-order)
        xi  = 0.0
        xd  = 0.0
        y   = np.zeros(N_steps)
        u_h = np.zeros(N_steps)

        for k in range(N_steps - 1):
            y[k] = y_p
            Kp_gain = plant_gain(y_p)

            if mode == 'scheduled':
                # Scale controller gains inversely with plant gain
                scale = K_lo / max(Kp_gain, 1e-6)
                Kp_k = Kp_fixed * scale
                Ki_k = Ki_fixed * scale
                Kd_k = Kd_fixed * scale
            else:
                Kp_k, Ki_k, Kd_k = Kp_fixed, Ki_fixed, Kd_fixed

            if mode == 'scheduled':
                K_hist[k] = Kp_gain

            e = ref - y_p
            der = Kd_k * N * (e - xd)
            u_k = float(np.clip(Kp_k * e + xi + der, -1e6, 1e6))
            u_h[k] = u_k

            xd += _DT * N * (e - xd)
            xi += _DT * Ki_k * e
            # Nonlinear plant Euler
            y_p += _DT * (-y_p + Kp_gain * u_k) / tau

        y[-1] = y_p
        out[mode] = (y, u_h)

    K_hist[-1] = plant_gain(out['scheduled'][0][-1])
    return {'t': t, 'fixed': out['fixed'], 'scheduled': out['scheduled'],
            'K_history': K_hist}


# ── Feedforward ─────────────────────────────────────────────────────────────

def simulate_feedforward(
    plant_num, plant_den,
    Kp: float, Ki: float, Kd: float, N: float,
    disturbance_amp: float, disturbance_time: float,
    ff_gain: float,
    ref: float, t_end: float,
) -> Dict:
    """
    Simulate PID with and without feedforward disturbance rejection.

    Disturbance d(t) = disturbance_amp for t >= disturbance_time (step disturbance
    entering at the plant input — same channel as actuator).

    Feedforward: u_ff = -ff_gain * d(t)
    Perfect rejection requires ff_gain = 1.0 (unity gain cancellation).

    Returns dict with 't', 'fb_only', 'fb_ff', 'disturbance'.
    """
    A, B, C, D = _plant_ss(plant_num, plant_den)
    n_p = A.shape[0]
    t = np.arange(0.0, t_end, _DT)
    N_steps = len(t)
    d_arr = np.where(t >= disturbance_time, disturbance_amp, 0.0)

    out = {}
    for mode in ('fb_only', 'fb_ff'):
        x_p = np.zeros(n_p)
        xi  = 0.0
        xd  = 0.0
        y   = np.zeros(N_steps)
        u_h = np.zeros(N_steps)

        for k in range(N_steps - 1):
            y_k = float((C @ x_p).flat[0])
            y[k] = y_k
            e = ref - y_k

            der = Kd * N * (e - xd)
            u_fb = Kp * e + xi + der
            u_ff = -ff_gain * d_arr[k] if mode == 'fb_ff' else 0.0
            u_k  = float(u_fb + u_ff)
            u_h[k] = u_k

            xd += _DT * N * (e - xd)
            xi += _DT * Ki * e
            # Plant + disturbance at input
            total_input = u_k + d_arr[k]
            x_p += _DT * ((A @ x_p).flatten() + (B * total_input).flatten())

        y[-1] = float((C @ x_p).flat[0])
        out[mode] = (y, u_h)

    return {'t': t, 'fb_only': out['fb_only'], 'fb_ff': out['fb_ff'],
            'disturbance': d_arr}


# ── Info text for UI tip panels ─────────────────────────────────────────────

ANTIWINDUP_TIP = """
**What is integrator windup?**
When the actuator saturates (e.g. valve fully open), the PID integrator keeps
accumulating error even though the extra integral action has no effect on the
output — the actuator is already at its limit. When the setpoint is then reduced,
the integrator must first unwind through zero before the controller can act
normally again, causing a large overshoot and slow recovery.

**Back-calculation anti-windup fix**
A correction term is fed back into the integrator when saturation is detected:

`dxi/dt = Ki·e + (u_sat − u) / Tt`

- When unsaturated: `u_sat = u`, correction = 0 → normal PID.
- When saturated: correction actively drives the integrator toward the
  value that would produce exactly `u_sat` — it stops wind-up immediately.
- **Tt** (tracking time constant): smaller → faster unwind but noisier.
  Rule of thumb: `Tt = sqrt(Kd/Ki)` or `Tt = Kp/Ki`.

**When to use:** Any time the actuator can saturate — motor current limits,
valve positions (0–100%), pneumatic pressure limits, robot joint torque limits.
"""

GAIN_SCHEDULING_TIP = """
**Why does a single PID fail on nonlinear plants?**
A linear PID is tuned at one operating point. If the plant gain changes
significantly with operating conditions (flow rate, temperature, load), the
same Kp/Ki/Kd produces a different loop gain — causing oscillation at high
gain or sluggish response at low gain.

**Gain scheduling: the idea**
Measure an easily-available *scheduling variable* (usually the current setpoint,
output, or a known process variable), look up or interpolate pre-tuned gains
from a table, and update the PID coefficients in real time.

`Kp(y) = Kp_base × K_nominal / K_plant(y)`

This keeps the open-loop gain constant across operating points.

**Real examples**
- **Aircraft autopilot** — pitch dynamics change with airspeed → schedule gains
  on measured Mach number.
- **Exothermic CSTR** — reaction rate (and plant gain) doubles every 10 °C
  → schedule on reactor temperature.
- **pH neutralisation** — gain varies by 1000× across pH range → schedule on pH.

**When to use:** Plant gain, time constant, or dynamics vary significantly
(>2×) across the expected operating range.
"""

FEEDFORWARD_TIP = """
**Why is feedback alone sometimes not enough?**
Feedback is reactive — the controller only acts *after* an error has appeared.
If a disturbance is *measurable* before it affects the output, feedback wastes
time waiting for the error to develop.

**Feedforward: act before the disturbance arrives**
If disturbance d(t) enters the plant at the input:
```
u_total = u_fb + u_ff = PID(e) − ff_gain × d
```
With ff_gain = 1.0, the feedforward exactly cancels the disturbance at the
input, and the feedback loop only needs to handle residual model error.

**Two feedforward strategies**
| Type | When to use |
|------|-------------|
| **Disturbance FF** | Measurable load / disturbance (e.g. feed flow changes, ambient temperature, grid load) |
| **Reference FF (2DOF)** | Add `u_ff = ref / K_plant` to track setpoint changes faster without increasing integral action |

**Real examples**
- **Boiler control** — feedforward from steam demand before pressure drops.
- **CNC machining** — feed force measured and cancelled before position error builds.
- **HVAC** — outdoor temperature measured → feedforward to heating setpoint.

**Limitation:** Requires an accurate plant model for the feedforward path.
Model mismatch → partial (but still useful) disturbance rejection.
"""
