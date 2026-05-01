import numpy as np
from scipy import signal
from typing import Tuple, List

_EPS = 1e-12


def get_pid_tf(Kp: float, Ki: float, Kd: float, N: float = 100.0) -> Tuple[List, List]:
    """
    PID with first-order derivative filter:
        C(s) = Kp + Ki/s + Kd·N·s/(s+N)

    Handles P, PI, PD, and full PID as special cases.
    Returns (num, den) coefficient lists, highest power first.
    """
    has_I = Ki > _EPS
    has_D = Kd > _EPS

    if not has_I and not has_D:
        # Pure P
        return [float(Kp)], [1.0]

    if not has_I:
        # PD: [(Kp + Kd·N)s + Kp·N] / (s + N)
        return [float(Kp + Kd*N), float(Kp*N)], [1.0, float(N)]

    if not has_D:
        # PI: (Kp·s + Ki) / s
        return [float(Kp), float(Ki)], [1.0, 0.0]

    # Full PID: [(Kp+Kd·N)s² + (Kp·N+Ki)s + Ki·N] / [s² + N·s]
    return (
        [float(Kp + Kd*N), float(Kp*N + Ki), float(Ki*N)],
        [1.0, float(N), 0.0],
    )


def discretize_tf(num, den, Ts: float, method: str = "tustin") -> Tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous TF via scipy.cont2discrete."""
    num_d, den_d, _ = signal.cont2discrete(
        (np.asarray(num, float), np.asarray(den, float)), Ts, method=method
    )
    return num_d.flatten(), den_d.flatten()
