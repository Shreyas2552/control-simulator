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

    return [1.0], [1.0]
