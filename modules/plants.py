import numpy as np
from typing import Dict, Tuple, List

PLANT_MODELS: Dict = {
    "First Order": {
        "tf_display": "G(s) = K / (τs + 1)",
        "physical_context": "Thermal systems, RC circuits, first-order chemical reactors, simple tank level",
        "params": {
            "K":   {"label": "DC Gain (K)",           "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau": {"label": "Time Constant τ (s)",    "default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1},
        },
    },
    "Second Order": {
        "tf_display": "G(s) = K·ωₙ² / (s² + 2ζωₙs + ωₙ²)",
        "physical_context": "Mass-spring-damper, RLC circuit, servo motor, electromechanical systems",
        "params": {
            "K":    {"label": "DC Gain (K)",              "default": 1.0,  "min": 0.1,  "max": 10.0, "step": 0.1},
            "wn":   {"label": "Natural Freq ωₙ (rad/s)",  "default": 2.0,  "min": 0.1,  "max": 20.0, "step": 0.1},
            "zeta": {"label": "Damping Ratio ζ",           "default": 0.5,  "min": 0.01, "max": 2.0,  "step": 0.01},
        },
    },
    "DC Motor (Position)": {
        "tf_display": "G(s) = Km / (s(τₘs + 1))",
        "physical_context": "DC servo motor with position output — type-1 system with natural integrator",
        "params": {
            "Km":    {"label": "Motor Gain (Km)",          "default": 1.0, "min": 0.1,  "max": 5.0, "step": 0.1},
            "tau_m": {"label": "Motor Time Const τₘ (s)",  "default": 0.5, "min": 0.01, "max": 5.0, "step": 0.01},
        },
    },
    "Integrating Plant": {
        "tf_display": "G(s) = K / s",
        "physical_context": "Pure integrator: liquid level control, velocity→position, robot joint angle",
        "params": {
            "K": {"label": "Integrator Gain (K)", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        },
    },
    "Unstable Plant": {
        "tf_display": "G(s) = K / (s − a)",
        "physical_context": "Linearised inverted pendulum, exothermic CSTR, unstable aircraft pitch mode",
        "params": {
            "K": {"label": "Gain (K)",              "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
            "a": {"label": "Unstable Pole a (> 0)", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
        },
    },
    "Third Order": {
        "tf_display": "G(s) = K / ((τ₁s+1)(τ₂s+1)(τ₃s+1))",
        "physical_context": "Cascaded industrial processes, multi-stage heat exchangers, complex fluid systems",
        "params": {
            "K":    {"label": "DC Gain (K)",           "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau1": {"label": "Time Constant τ₁ (s)",  "default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1},
            "tau2": {"label": "Time Constant τ₂ (s)",  "default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau3": {"label": "Time Constant τ₃ (s)",  "default": 0.5, "min": 0.1, "max":  5.0, "step": 0.1},
        },
    },
    "Non-minimum Phase": {
        "tf_display": "G(s) = K(1 − τ_z·s) / ((τ₁s+1)(τ₂s+1))",
        "physical_context": "Distillation column, boiler drum level, flexible robot arm — initial response moves opposite to setpoint",
        "params": {
            "K":     {"label": "DC Gain (K)",          "default": 1.0, "min": 0.1, "max": 5.0,  "step": 0.1},
            "tau_z": {"label": "RHP Zero τ_z (s)",     "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau1":  {"label": "Lag τ₁ (s)",           "default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1},
            "tau2":  {"label": "Lag τ₂ (s)",           "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        },
    },
    "Time Delay Process": {
        "tf_display": "G(s) = K·e^{−θs} / (τs+1)  [2nd-order Padé]",
        "physical_context": "Conveyor belt, pipeline transport, sensor dead-time, thermal process — delay limits achievable bandwidth",
        "params": {
            "K":     {"label": "DC Gain (K)",          "default": 1.0, "min": 0.1, "max": 5.0,  "step": 0.1},
            "tau":   {"label": "Time Constant τ (s)",  "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
            "theta": {"label": "Dead Time θ (s)",      "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
        },
    },
    "Integrating + Delay": {
        "tf_display": "G(s) = K·e^{−θs} / s  [2nd-order Padé]",
        "physical_context": "Tank level, batch reactor, integrating process with transport lag — common in process industries",
        "params": {
            "K":     {"label": "Integrator Gain (K)",  "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
            "theta": {"label": "Dead Time θ (s)",      "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
        },
    },
}


def get_plant_tf(plant_name: str, params: Dict) -> Tuple[List[float], List[float]]:
    """Return (numerator, denominator) coefficient lists, highest power first."""
    if plant_name == "First Order":
        return [float(params["K"])], [float(params["tau"]), 1.0]

    if plant_name == "Second Order":
        K, wn, z = params["K"], params["wn"], params["zeta"]
        return [float(K * wn**2)], [1.0, float(2*z*wn), float(wn**2)]

    if plant_name == "DC Motor (Position)":
        return [float(params["Km"])], [float(params["tau_m"]), 1.0, 0.0]

    if plant_name == "Integrating Plant":
        return [float(params["K"])], [1.0, 0.0]

    if plant_name == "Unstable Plant":
        return [float(params["K"])], [1.0, float(-params["a"])]

    if plant_name == "Third Order":
        d = np.polymul(
            np.polymul([float(params["tau1"]), 1.0], [float(params["tau2"]), 1.0]),
            [float(params["tau3"]), 1.0]
        )
        return [float(params["K"])], d.tolist()

    if plant_name == "Non-minimum Phase":
        K, tz = float(params["K"]), float(params["tau_z"])
        t1, t2 = float(params["tau1"]), float(params["tau2"])
        num = [K * (-tz), K * 1.0]
        den = np.polymul([t1, 1.0], [t2, 1.0]).tolist()
        return num, den

    if plant_name == "Time Delay Process":
        K, tau, th = float(params["K"]), float(params["tau"]), float(params["theta"])
        if th < 1e-6:
            return [K], [tau, 1.0]
        # 2nd-order Padé: e^{-θs} ≈ (θ²s²/12 - θs/2 + 1) / (θ²s²/12 + θs/2 + 1)
        p_num = [th**2 / 12, -th / 2, 1.0]
        p_den = [th**2 / 12,  th / 2, 1.0]
        num = [K * c for c in p_num]
        den = np.polymul([tau, 1.0], p_den).tolist()
        return num, den

    if plant_name == "Integrating + Delay":
        K, th = float(params["K"]), float(params["theta"])
        if th < 1e-6:
            return [K], [1.0, 0.0]
        p_num = [th**2 / 12, -th / 2, 1.0]
        p_den = [th**2 / 12,  th / 2, 1.0]
        num = [K * c for c in p_num]
        den = np.polymul([1.0, 0.0], p_den).tolist()
        return num, den

    return [1.0], [1.0]
