"""
LQR / Kalman Filter / LQG design and simulation routines.

All state-space systems follow the convention:
    ẋ = A x + B u + w        (process noise  w ~ N(0, Qn))
    y = C x + v              (measurement noise v ~ N(0, Rn))

LQR minimises  J = ∫(x'Qx + u'Ru) dt  (infinite-horizon, continuous-time).
Kalman filter minimises estimation error covariance.
LQG = LQR gain applied to Kalman-estimated state  (Separation Principle).
"""
import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
from typing import Dict, Tuple, List, Optional


# ── Plant catalogue ────────────────────────────────────────────────────────

SS_PLANTS: Dict = {
    "Mass-Spring-Damper": {
        "desc":          "ẋ = [v ; (F − kx − bv)/m]",
        "physical":      "Vehicle suspension, vibration absorber, robot joint compliance",
        "n_states": 2,  "n_inputs": 1,  "n_outputs": 1,
        "state_names":  ["Position  x (m)", "Velocity  v (m/s)"],
        "input_names":  ["Force  F (N)"],
        "output_names": ["Position  x (m)"],
        "ref_state":    0,          # which state the reference drives
        "params": {
            "m": {"label": "Mass  m (kg)",         "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "k": {"label": "Spring  k (N/m)",      "default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5},
            "b": {"label": "Damping  b (N·s/m)",   "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1},
        },
        "default_Q": [10.0, 1.0],
        "default_R":  0.1,
    },
    "DC Motor (Full Electrical + Mechanical)": {
        "desc":          "States: [armature current  i (A), angular velocity  ω (rad/s)]",
        "physical":      "Servo motor control, robotics, electric-vehicle drives",
        "n_states": 2,  "n_inputs": 1,  "n_outputs": 1,
        "state_names":  ["Current  i (A)", "Angular velocity  ω (rad/s)"],
        "input_names":  ["Voltage  V (V)"],
        "output_names": ["Angular velocity  ω (rad/s)"],
        "ref_state":    1,
        "params": {
            "Ra": {"label": "Armature R (Ω)",          "default": 1.0,   "min": 0.1, "max": 10.0, "step": 0.1},
            "L":  {"label": "Inductance L (H)",         "default": 0.5,   "min": 0.01,"max": 5.0,  "step": 0.01},
            "Km": {"label": "Motor const Km (N·m/A)",   "default": 0.5,   "min": 0.01,"max": 2.0,  "step": 0.01},
            "J":  {"label": "Rotor inertia J (kg·m²)",  "default": 0.01,  "min": 0.001,"max":1.0,  "step": 0.001},
            "bm": {"label": "Viscous friction b",       "default": 0.1,   "min": 0.0, "max": 2.0,  "step": 0.01},
        },
        "default_Q": [1.0, 10.0],
        "default_R":  0.1,
    },
    "Inverted Pendulum on Cart": {
        "desc":          "States: [cart pos p, cart vel ṗ, angle θ, angular rate θ̇]",
        "physical":      "Balancing robots (Segway, bipeds), rocket attitude, overhead crane",
        "n_states": 4,  "n_inputs": 1,  "n_outputs": 1,
        "state_names":  ["Cart pos  p (m)", "Cart vel  ṗ (m/s)",
                         "Angle  θ (rad)",  "Angular rate  θ̇ (rad/s)"],
        "input_names":  ["Horizontal force  F (N)"],
        "output_names": ["Cart pos  p (m)"],
        "ref_state":    0,
        "params": {
            "M": {"label": "Cart mass  M (kg)",    "default": 0.5,  "min": 0.1, "max": 5.0,  "step": 0.1},
            "m": {"label": "Pend. mass  m (kg)",   "default": 0.2,  "min": 0.05,"max": 2.0,  "step": 0.05},
            "l": {"label": "Pend. length  l (m)",  "default": 0.3,  "min": 0.1, "max": 2.0,  "step": 0.05},
            "g": {"label": "Gravity  g (m/s²)",    "default": 9.81, "min": 9.0, "max": 10.0, "step": 0.01},
        },
        "default_Q": [100.0, 1.0, 100.0, 1.0],
        "default_R":  0.1,
    },
    "Double Integrator": {
        "desc":          "ẍ = k·u  →  States: [position  x, velocity  v]",
        "physical":      "Spacecraft attitude, frictionless conveyor, stepper motor",
        "n_states": 2,  "n_inputs": 1,  "n_outputs": 1,
        "state_names":  ["Position  x (m)", "Velocity  v (m/s)"],
        "input_names":  ["Acceleration  u (m/s²)"],
        "output_names": ["Position  x (m)"],
        "ref_state":    0,
        "params": {
            "scale": {"label": "Input gain  k", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
        },
        "default_Q": [10.0, 1.0],
        "default_R":  0.1,
    },
}


# ── State-space matrix builders ────────────────────────────────────────────

def get_ss(plant_name: str, params: Dict) -> Tuple[np.ndarray, ...]:
    """Return (A, B, C, D) for the chosen plant and parameters."""

    if plant_name == "Mass-Spring-Damper":
        m, k, b = params["m"], params["k"], params["b"]
        A = np.array([[0.,    1.],
                      [-k/m, -b/m]])
        B = np.array([[0.], [1./m]])
        C = np.array([[1., 0.]])
        D = np.zeros((1, 1))
        return A, B, C, D

    if plant_name == "DC Motor (Full Electrical + Mechanical)":
        Ra, L, Km, J, bm = params["Ra"], params["L"], params["Km"], params["J"], params["bm"]
        A = np.array([[-Ra/L, -Km/L],
                      [ Km/J, -bm/J]])
        B = np.array([[1./L], [0.]])
        C = np.array([[0., 1.]])
        D = np.zeros((1, 1))
        return A, B, C, D

    if plant_name == "Inverted Pendulum on Cart":
        M, m, l, g = params["M"], params["m"], params["l"], params["g"]
        # Linearised about upright equilibrium θ=0 using Lagrangian mechanics.
        # p̈  = F/M  −  (m·g/M)·φ
        # φ̈  = −F/(M·l)  +  (M+m)·g/(M·l)·φ
        A = np.array([[0.,  1.,               0.,  0.],
                      [0.,  0.,        -m*g/M,     0.],
                      [0.,  0.,               0.,  1.],
                      [0.,  0.,  (M+m)*g/(M*l),    0.]])
        B = np.array([[0.], [1./M], [0.], [-1./(M*l)]])
        C = np.array([[1., 0., 0., 0.]])       # observe cart position
        D = np.zeros((1, 1))
        return A, B, C, D

    if plant_name == "Double Integrator":
        s = params["scale"]
        A = np.zeros((2, 2));  A[0, 1] = 1.
        B = np.array([[0.], [float(s)]])
        C = np.array([[1., 0.]])
        D = np.zeros((1, 1))
        return A, B, C, D

    return np.eye(2), np.ones((2, 1)), np.eye(1, 2), np.zeros((1, 1))


# ── Controllability / observability ────────────────────────────────────────

def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    n = A.shape[0]
    Co = B.copy()
    for i in range(1, n):
        Co = np.hstack([Co, np.linalg.matrix_power(A, i) @ B])
    return int(np.linalg.matrix_rank(Co))


def observability_rank(A: np.ndarray, C: np.ndarray) -> int:
    n = A.shape[0]
    Ob = C.copy()
    for i in range(1, n):
        Ob = np.vstack([Ob, C @ np.linalg.matrix_power(A, i)])
    return int(np.linalg.matrix_rank(Ob))


# ── LQR design ─────────────────────────────────────────────────────────────

def lqr_design(A: np.ndarray, B: np.ndarray,
               Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the continuous-time LQR problem.
        Riccati: A'P + PA − P·B·R⁻¹·B'·P + Q = 0
        Gain:    K = R⁻¹ B' P
    Returns (K, P, closed-loop eigenvalues).
    Raises ValueError if Riccati equation fails (uncontrollable/unstabilisable).
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    eigs_cl = eigvals(A - B @ K)
    return K, P, eigs_cl


# ── Kalman filter design ────────────────────────────────────────────────────

def kalman_design(A: np.ndarray, C: np.ndarray,
                  Qn: np.ndarray, Rn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dual of LQR: solve the observer Riccati equation.
        A·Pe + Pe·A' − Pe·C'·Rn⁻¹·C·Pe + Qn = 0
        Observer gain: L = Pe·C'·Rn⁻¹
    Returns (L, Pe).
    """
    Pe = solve_continuous_are(A.T, C.T, Qn, Rn)
    L  = Pe @ C.T @ np.linalg.inv(Rn)
    return L, Pe


# ── Reference pre-compensator ──────────────────────────────────────────────

def compute_nbar(A: np.ndarray, B: np.ndarray,
                 C_row: np.ndarray, K: np.ndarray) -> float:
    """
    Compute Nbar so that the steady-state output y_ss = reference r.
        u = −K·x + Nbar·r
        y_ss = r  ⟺  Nbar = −1 / [C·(A−BK)⁻¹·B]
    Returns 1.0 on failure (systems with integrators don't need it).
    """
    try:
        Acl = A - B @ K
        dc  = C_row @ np.linalg.solve(-Acl, B)
        val = float(dc.flat[0])
        return 1.0 / val if abs(val) > 1e-10 else 1.0
    except Exception:
        return 1.0


# ── Simulations ────────────────────────────────────────────────────────────

_DT = 0.005          # fixed integration step (200 Hz)


def simulate_lqr(A, B, C_out, K, Nbar, ref: float, t_end: float,
                 ref_state: int = 0):
    """
    Euler-integration of the LQR closed-loop system (no noise).
    Reference tracking via  u = −K·(x − x_ref) + feedforward.
    Returns (t, x_history, y_history, u_history).
    """
    n = A.shape[0]
    t = np.arange(0., t_end, _DT)
    N = len(t)

    x   = np.zeros((N, n))
    y   = np.zeros(N)
    u_h = np.zeros(N)

    # Build desired state x_ref: only the tracked state is non-zero
    x_ref = np.zeros(n)
    x_ref[ref_state] = ref

    for k in range(N - 1):
        u_k      = (-K @ (x[k] - x_ref)).item()
        u_h[k]   = u_k
        xdot     = A @ x[k] + (B @ np.array([[u_k]])).flatten()
        x[k + 1] = x[k] + _DT * xdot

    for k in range(N):
        y[k] = (C_out @ x[k]).item()

    u_h[-1] = u_h[-2]
    return t, x, y, u_h


def simulate_lqg(A, B, C_out, C_obs, K, L, Nbar,
                 ref: float, t_end: float,
                 q_std: float, r_std: float,
                 ref_state: int = 0, seed: int = 42):
    """
    Euler-integration of the full LQG loop:
        True plant  + process noise  w ~ N(0, q_std²·I)
        Measurement + noise         v ~ N(0, r_std²)
        Kalman estimator running on noisy y
        LQR gain applied to estimated state

    Returns (t, x_true, x_est, y_meas, u_history).
    """
    n  = A.shape[0]
    p  = C_obs.shape[0]
    t  = np.arange(0., t_end, _DT)
    N  = len(t)

    x_true = np.zeros((N, n))
    x_est  = np.zeros((N, n))
    y_meas = np.zeros((N, p))
    u_h    = np.zeros(N)

    x_ref = np.zeros(n)
    x_ref[ref_state] = ref

    rng = np.random.default_rng(seed)

    for k in range(N - 1):
        u_k    = (-K @ (x_est[k] - x_ref)).item()
        u_h[k] = u_k

        # True plant
        w            = rng.normal(0., q_std, n)
        xdot_t       = A @ x_true[k] + (B @ [[u_k]]).flatten() + w
        x_true[k+1]  = x_true[k] + _DT * xdot_t

        # Noisy measurement
        v          = rng.normal(0., r_std, p)
        y_meas[k]  = C_obs @ x_true[k] + v

        # Kalman observer
        innov        = y_meas[k] - C_obs @ x_est[k]
        xdot_e       = A @ x_est[k] + (B @ [[u_k]]).flatten() + (L @ innov).flatten()
        x_est[k+1]   = x_est[k] + _DT * xdot_e

    y_meas[-1] = C_obs @ x_true[-1]
    u_h[-1]    = u_h[-2]
    return t, x_true, x_est, y_meas, u_h
