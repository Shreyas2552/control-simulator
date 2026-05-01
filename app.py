"""
Interactive Control Systems Simulator
======================================
PID controller with realistic plant models, full stability analysis,
root locus, Bode plots, pole-zero maps, and filter design.
Educational tool for students and practising control engineers.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import signal as sci_signal

from modules.plants        import PLANT_MODELS, get_plant_tf
from modules.pid_controller import get_pid_tf, discretize_tf
from modules.analysis      import (
    build_ol_cl, step_response, ramp_response, control_signal_step,
    bode_data, stability_margins, root_locus_data,
    cl_pole_analysis, performance_metrics,
)
from modules.filters import (
    FILTER_TYPES, FILTER_FAMILIES,
    design_analog, design_digital,
    filter_bode, filter_step,
)

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Control Systems Simulator",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .stable-badge   { color: #4CAF50; font-weight: bold; font-size: 1.1rem; }
    .unstable-badge { color: #F44336; font-weight: bold; font-size: 1.1rem; }
    .marginal-badge { color: #FF9800; font-weight: bold; font-size: 1.1rem; }
    .theory-box {
        background: #1a1f2e;
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Colour palette (consistent across all plots)
# ============================================================
C = dict(
    ref       = "#2196F3",   # reference
    output    = "#4CAF50",   # system output
    error     = "#FF5722",   # error signal
    control   = "#9C27B0",   # control signal u(t)
    locus     = "#607D8B",   # root-locus paths
    ol_pole   = "#F44336",   # open-loop pole (×)
    ol_zero   = "#2196F3",   # open-loop zero (○)
    cl_pole   = "#FF9800",   # closed-loop pole (★)
    stable    = "#4CAF50",
    unstable  = "#F44336",
    neutral   = "#90A4AE",
    gm_line   = "#FF9800",
    pm_line   = "#E91E63",
    grid      = "rgba(255,255,255,0.07)",
)

PLOT_LAYOUT = dict(
    template    = "plotly_dark",
    plot_bgcolor = "#0e1117",
    paper_bgcolor= "#0e1117",
    font        = dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend      = dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin      = dict(l=60, r=30, t=40, b=50),
)

# ============================================================
# Sidebar — configuration
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ Control Simulator")
    st.markdown("---")

    # --- System domain ---
    st.markdown("### System Domain")
    domain = st.radio("", ["Continuous", "Discrete"], horizontal=True, label_visibility="collapsed")
    Ts = None
    if domain == "Discrete":
        Ts = st.number_input("Sample Time Tₛ (s)", min_value=0.001, max_value=1.0,
                             value=0.05, step=0.005, format="%.3f")
    st.markdown("---")

    # --- Plant ---
    st.markdown("### Plant Model")
    plant_name = st.selectbox("Plant", list(PLANT_MODELS.keys()), label_visibility="collapsed")
    info = PLANT_MODELS[plant_name]
    st.caption(f"**{info['tf_display']}**")
    st.caption(f"📌 {info['physical_context']}")

    plant_params = {}
    for pname, pmeta in info["params"].items():
        plant_params[pname] = st.slider(
            pmeta["label"],
            min_value=float(pmeta["min"]),
            max_value=float(pmeta["max"]),
            value=float(pmeta["default"]),
            step=float(pmeta["step"]),
        )
    st.markdown("---")

    # --- PID ---
    st.markdown("### PID Controller")
    col1, col2 = st.columns(2)
    with col1:
        Kp = st.number_input("Kp", min_value=0.0, max_value=1000.0, value=2.0, step=0.1, format="%.2f")
        Ki = st.number_input("Ki", min_value=0.0, max_value=1000.0, value=1.0, step=0.1, format="%.2f")
    with col2:
        Kd = st.number_input("Kd", min_value=0.0, max_value=500.0,  value=0.5, step=0.05, format="%.3f")
        N  = st.number_input("Filter N", min_value=1.0, max_value=1000.0, value=100.0, step=10.0,
                             help="Derivative filter coefficient. Higher → less lag, more noise sensitivity.")
    st.markdown("---")

    # --- Simulation ---
    st.markdown("### Reference & Simulation")
    ref_type = st.selectbox("Reference Shape", ["Step", "Ramp"])
    ref_amp  = st.number_input("Amplitude", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    t_end    = st.slider("Simulation Duration (s)", 2.0, 200.0, 30.0, 1.0)

# ============================================================
# Core computations
# ============================================================
plant_num, plant_den = get_plant_tf(plant_name, plant_params)
plant_num = np.array(plant_num, float)
plant_den = np.array(plant_den, float)

ctrl_num, ctrl_den = get_pid_tf(Kp, Ki, Kd, N)
ctrl_num = np.array(ctrl_num, float)
ctrl_den = np.array(ctrl_den, float)

discrete = (domain == "Discrete")

if discrete and Ts:
    # Discretise plant and PID
    try:
        pn_d, pd_d = discretize_tf(plant_num, plant_den, Ts)
        cn_d, cd_d = discretize_tf(ctrl_num,  ctrl_den,  Ts)
        ol_num, ol_den, cl_num, cl_den = build_ol_cl(pn_d, pd_d, cn_d, cd_d)
    except Exception:
        ol_num, ol_den, cl_num, cl_den = build_ol_cl(plant_num, plant_den, ctrl_num, ctrl_den)
        pn_d, pd_d, cn_d, cd_d = plant_num, plant_den, ctrl_num, ctrl_den
else:
    pn_d = pd_d = cn_d = cd_d = None
    ol_num, ol_den, cl_num, cl_den = build_ol_cl(plant_num, plant_den, ctrl_num, ctrl_den)

# Step / Ramp response
try:
    if ref_type == "Step":
        t_resp, y_resp = step_response(
            cl_num, cl_den, t_end, n=2000, discrete=discrete, Ts=Ts
        )
        ref_signal = np.full_like(t_resp, ref_amp)
        y_resp     = y_resp * ref_amp
    else:
        t_resp, y_resp = ramp_response(cl_num, cl_den, t_end, n=2000)
        ref_signal = ref_amp * t_resp
        y_resp     = y_resp * ref_amp

    err_signal = ref_signal - y_resp

    t_u, u_sig = control_signal_step(plant_num, plant_den, ctrl_num, ctrl_den, t_end, n=2000)
    u_sig = u_sig * ref_amp
except Exception as e:
    t_resp = np.linspace(0, t_end, 200)
    y_resp = ref_signal = err_signal = u_sig = np.zeros_like(t_resp)
    t_u = t_resp
    st.error(f"Simulation error: {e}")

# Stability metrics
margins = stability_margins(ol_num, ol_den, discrete=discrete, Ts=Ts)
pole_rows = cl_pole_analysis(cl_den)
metrics   = performance_metrics(t_resp, y_resp, ref=ref_amp if ref_type == "Step" else None)

all_stable = all(r["stable"] for r in pole_rows) if pole_rows else False
any_unstable = any(not r["stable"] for r in pole_rows) if pole_rows else False


def _stability_label():
    if all_stable:
        return '<span class="stable-badge">● STABLE</span>'
    elif any_unstable:
        return '<span class="unstable-badge">● UNSTABLE</span>'
    return '<span class="marginal-badge">● MARGINAL</span>'


# ============================================================
# Header
# ============================================================
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("## 🎛️ Interactive Control Systems Simulator")
    st.markdown(
        f"**Plant:** {info['tf_display']}  |  "
        f"**Controller:** PID (Kp={Kp}, Ki={Ki}, Kd={Kd})  |  "
        f"**Domain:** {domain}"
    )
with hcol2:
    st.markdown(f"### {_stability_label()}", unsafe_allow_html=True)
    pm = margins.get("pm_deg")
    gm = margins.get("gm_db")
    pm_str = f"{pm:.1f}°" if pm is not None and not np.isinf(pm) else ("∞" if pm == np.inf else "N/A")
    gm_str = f"{gm:.1f} dB" if gm is not None and not np.isinf(gm) else ("∞" if gm == np.inf else "N/A")
    st.caption(f"PM = {pm_str}  |  GM = {gm_str}")

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab_labels = [
    "📈 Time Response",
    "🌀 Root Locus",
    "📊 Bode Plot",
    "🔍 Stability Analysis",
    "🔈 Filters",
    "📚 Theory",
]
tabs = st.tabs(tab_labels)

# ─────────────────────────────────────────────────────────────
# TAB 1 — Time Response
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    # Metrics strip
    if metrics:
        mc = st.columns(5)
        items = [
            ("Overshoot", metrics.get("Overshoot (%)"),       "%.1f %%"),
            ("Rise Time",  metrics.get("Rise Time (s)"),      "%.3f s"),
            ("Settle(2%)", metrics.get("Settling Time (2%) (s)"), "%.3f s"),
            ("SS Error",   metrics.get("Steady-State Error"), "%.4f"),
            ("Peak Time",  metrics.get("Peak Time (s)"),      "%.3f s"),
        ]
        for col, (label, val, fmt) in zip(mc, items):
            col.metric(label, fmt % val if val is not None else "N/A")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Output y(t) vs Reference r(t)", "Error e(t)", "Control Signal u(t)"),
    )

    # Row 1: Reference + Output
    fig.add_trace(go.Scatter(x=t_resp, y=ref_signal, name="Reference r(t)",
                             line=dict(color=C["ref"], dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_resp, y=y_resp, name="Output y(t)",
                             line=dict(color=C["output"], width=2)), row=1, col=1)

    # Row 2: Error
    fig.add_trace(go.Scatter(x=t_resp, y=err_signal, name="Error e(t)",
                             line=dict(color=C["error"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(255,87,34,0.12)"), row=2, col=1)

    # Row 3: Control signal
    fig.add_trace(go.Scatter(x=t_u, y=u_sig, name="Control u(t)",
                             line=dict(color=C["control"], width=1.5)), row=3, col=1)

    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor=C["grid"])
    fig.update_yaxes(title_text="Error",     row=2, col=1, gridcolor=C["grid"])
    fig.update_yaxes(title_text="u(t)",      row=3, col=1, gridcolor=C["grid"])
    fig.update_xaxes(title_text="Time (s)",  row=3, col=1, gridcolor=C["grid"])
    fig.update_layout(height=650, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2 — Root Locus
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "Root locus shows where the **closed-loop poles move** as the loop gain K scales "
        "from 0 → ∞. The ★ marks the current operating point. "
        "Branches left of the imaginary axis (or inside the unit circle for discrete) "
        "indicate stable operation."
    )

    locus, ol_poles, ol_zeros = root_locus_data(ol_num, ol_den)

    fig_rl = go.Figure()

    # Draw stability boundary
    if discrete and Ts:
        theta = np.linspace(0, 2*np.pi, 400)
        fig_rl.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode="lines", name="Unit Circle",
            line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.5),
        ))
        axis_label = "Re(z)"
        imag_label = "Im(z)"
    else:
        y_range = [-20, 20]
        fig_rl.add_shape(type="line", x0=0, x1=0, y0=-50, y1=50,
                         line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.5))
        axis_label = "Re(s)"
        imag_label = "Im(s)"

    # Plot locus branches
    if locus:
        n_branches = len(locus[0][1])
        colours_rl = [f"hsl({int(i*360/n_branches)},70%,60%)" for i in range(n_branches)]
        branch_data = [{"x": [], "y": []} for _ in range(n_branches)]
        for _, roots in locus:
            for b_idx, r in enumerate(roots[:n_branches]):
                branch_data[b_idx]["x"].append(r.real)
                branch_data[b_idx]["y"].append(r.imag)

        for b_idx, bd in enumerate(branch_data):
            fig_rl.add_trace(go.Scatter(
                x=bd["x"], y=bd["y"],
                mode="lines", name=f"Branch {b_idx+1}",
                line=dict(color=colours_rl[b_idx], width=1.5),
                showlegend=(b_idx < 4),
            ))

    # OL poles (×) and zeros (○)
    if len(ol_poles):
        fig_rl.add_trace(go.Scatter(
            x=ol_poles.real, y=ol_poles.imag,
            mode="markers", name="OL Poles",
            marker=dict(symbol="x", size=14, color=C["ol_pole"], line=dict(width=2.5)),
        ))
    if len(ol_zeros):
        fig_rl.add_trace(go.Scatter(
            x=ol_zeros.real, y=ol_zeros.imag,
            mode="markers", name="OL Zeros",
            marker=dict(symbol="circle-open", size=14, color=C["ol_zero"], line=dict(width=2.5)),
        ))

    # Current CL poles (★)
    cl_poles_arr = np.roots(cl_den)
    fig_rl.add_trace(go.Scatter(
        x=cl_poles_arr.real, y=cl_poles_arr.imag,
        mode="markers", name="CL Poles (current)",
        marker=dict(symbol="star", size=16, color=C["cl_pole"],
                    line=dict(color="white", width=1)),
    ))

    fig_rl.update_layout(
        xaxis_title=axis_label,
        yaxis_title=imag_label,
        yaxis_scaleanchor="x",
        height=550,
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_rl, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Open-Loop Poles (×)**")
        if len(ol_poles):
            st.dataframe(pd.DataFrame({
                "Real": ol_poles.real.round(4),
                "Imag": ol_poles.imag.round(4),
            }), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Open-Loop Zeros (○)**")
        if len(ol_zeros):
            st.dataframe(pd.DataFrame({
                "Real": ol_zeros.real.round(4),
                "Imag": ol_zeros.imag.round(4),
            }), use_container_width=True, hide_index=True)
        else:
            st.info("No finite zeros.")

# ─────────────────────────────────────────────────────────────
# TAB 3 — Bode Plot
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    try:
        w_b, mag_b, ph_b = bode_data(ol_num, ol_den, n=800, discrete=discrete, Ts=Ts)

        fig_bode = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Magnitude (dB)", "Phase (°)"),
        )

        fig_bode.add_trace(go.Scatter(
            x=w_b, y=mag_b, name="|L(jω)| (dB)",
            line=dict(color="#2196F3", width=2)), row=1, col=1)
        fig_bode.add_trace(go.Scatter(
            x=w_b, y=ph_b, name="∠L(jω) (°)",
            line=dict(color="#4CAF50", width=2)), row=2, col=1)

        # 0 dB line
        fig_bode.add_hline(y=0,    row=1, col=1, line=dict(color="gray", dash="dot", width=1))
        # −180° line
        fig_bode.add_hline(y=-180, row=2, col=1, line=dict(color="gray", dash="dot", width=1))

        wgc = margins.get("wgc")
        wpc = margins.get("wpc")
        pm_ = margins.get("pm_deg")
        gm_ = margins.get("gm_db")

        # Draw vertical crossover lines on each subplot row explicitly
        for row_idx in [1, 2]:
            if wgc:
                fig_bode.add_vline(x=wgc, row=row_idx, col=1,
                                   line=dict(color=C["pm_line"], dash="dash", width=1.5))
            if wpc:
                fig_bode.add_vline(x=wpc, row=row_idx, col=1,
                                   line=dict(color=C["gm_line"], dash="dash", width=1.5))
        # Annotations on magnitude row
        if wgc:
            fig_bode.add_annotation(x=np.log10(wgc), y=0, xref="x", yref="y",
                                    text=f"ωgc={wgc:.2f}", showarrow=False,
                                    font=dict(color=C["pm_line"], size=11),
                                    yshift=10, row=1, col=1)
        if wpc:
            fig_bode.add_annotation(x=np.log10(wpc), y=-20, xref="x", yref="y",
                                    text=f"ωpc={wpc:.2f}", showarrow=False,
                                    font=dict(color=C["gm_line"], size=11),
                                    yshift=10, row=1, col=1)

        xaxis_type = "log"
        fig_bode.update_xaxes(type=xaxis_type, title_text="Frequency (rad/s)", row=2, col=1, gridcolor=C["grid"])
        fig_bode.update_xaxes(type=xaxis_type, gridcolor=C["grid"], row=1, col=1)
        fig_bode.update_yaxes(title_text="dB",  row=1, col=1, gridcolor=C["grid"])
        fig_bode.update_yaxes(title_text="deg", row=2, col=1, gridcolor=C["grid"])
        fig_bode.update_layout(height=560, **PLOT_LAYOUT)
        st.plotly_chart(fig_bode, use_container_width=True)

    except Exception as e:
        st.error(f"Bode computation failed: {e}")

    # Margin summary
    st.markdown("#### Stability Margins")
    m_cols = st.columns(4)
    def _fmt_margin(v, unit=""):
        if v is None:       return "N/A"
        if np.isinf(v):     return "∞"
        return f"{v:.2f}{unit}"

    m_cols[0].metric("Phase Margin",           _fmt_margin(pm_, "°"))
    m_cols[1].metric("Gain Margin",            _fmt_margin(gm_, " dB"))
    m_cols[2].metric("Gain Crossover ωgc",     _fmt_margin(wgc, " rad/s"))
    m_cols[3].metric("Phase Crossover ωpc",    _fmt_margin(wpc, " rad/s"))

    if pm_ is not None and not np.isinf(pm_):
        if pm_ > 45:
            st.success(f"✅ Good robustness: PM = {pm_:.1f}° (target > 45°)")
        elif pm_ > 0:
            st.warning(f"⚠️ Low phase margin: PM = {pm_:.1f}° — system may be lightly damped")
        else:
            st.error(f"❌ Negative phase margin: PM = {pm_:.1f}° — closed loop is UNSTABLE")

# ─────────────────────────────────────────────────────────────
# TAB 4 — Stability Analysis
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Pole-Zero Map")

    fig_pz = go.Figure()

    # Stability boundary
    if discrete and Ts:
        theta = np.linspace(0, 2*np.pi, 400)
        fig_pz.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode="lines",
            name="Unit Circle", line=dict(color="rgba(255,255,255,0.3)", dash="dot")))
    else:
        fig_pz.add_shape(type="line", x0=0, x1=0, y0=-50, y1=50,
                         line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.5))

    # CL poles coloured by stability
    for row in pole_rows:
        colour = C["stable"] if row["stable"] else C["unstable"]
        fig_pz.add_trace(go.Scatter(
            x=[row["real"]], y=[row["imag"] if row["imag"] else 0],
            mode="markers", name=row["pole"],
            marker=dict(symbol="x", size=16, color=colour, line=dict(width=3)),
            showlegend=True,
        ))
        if row["imag"] > 0:     # conjugate
            fig_pz.add_trace(go.Scatter(
                x=[row["real"]], y=[-row["imag"]],
                mode="markers", showlegend=False,
                marker=dict(symbol="x", size=16, color=colour, line=dict(width=3)),
            ))

    # OL zeros
    cl_zeros = np.roots(cl_num) if len(cl_num) > 1 else np.array([])
    if len(cl_zeros):
        fig_pz.add_trace(go.Scatter(
            x=cl_zeros.real, y=cl_zeros.imag,
            mode="markers", name="CL Zeros",
            marker=dict(symbol="circle-open", size=14, color=C["ol_zero"],
                        line=dict(width=2.5)),
        ))

    # Damping-ratio lines (continuous only)
    if not discrete:
        for zeta_line in [0.1, 0.2, 0.4, 0.7, 1.0]:
            angle = np.arccos(zeta_line)
            r_max = max(
                [abs(p["real"]) + abs(p["imag"]) for p in pole_rows] or [5]
            ) * 1.5
            xv = [-r_max * zeta_line, 0, -r_max * zeta_line]
            yv = [ r_max * np.sin(angle), 0, -r_max * np.sin(angle)]
            fig_pz.add_trace(go.Scatter(
                x=xv, y=yv, mode="lines", showlegend=False,
                line=dict(color="rgba(255,255,255,0.12)", dash="dot", width=1),
            ))
            fig_pz.add_annotation(
                x=-r_max * zeta_line * 0.9, y=r_max * np.sin(angle) * 0.9,
                text=f"ζ={zeta_line}", showarrow=False,
                font=dict(size=9, color="rgba(200,200,200,0.6)"),
            )

    fig_pz.update_layout(
        xaxis_title="Real Axis", yaxis_title="Imaginary Axis",
        yaxis_scaleanchor="x", height=450, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_pz, use_container_width=True)

    # Pole table
    st.markdown("### Closed-Loop Pole Analysis")
    if pole_rows:
        df_poles = pd.DataFrame([{
            "Pole Location":     r["pole"],
            "Type":              r["kind"],
            "ωₙ (rad/s)":        f"{r['wn']:.4f}",
            "ζ (Damping)":       f"{r['zeta']:.4f}",
            "ωd (rad/s)":        f"{r['wd']:.4f}" if r["wd"] else "—",
            "Overshoot (%)":     f"{r['overshoot_pct']:.1f}" if r["overshoot_pct"] else "0",
            "Settling Tₛ (s)":   f"{r['settling_time']:.3f}" if np.isfinite(r['settling_time']) else "∞",
            "Stable":            "✅" if r["stable"] else "❌",
        } for r in pole_rows])
        st.dataframe(df_poles, use_container_width=True, hide_index=True)

    # System type
    integrators_plant = int(round(len([p for p in np.roots(plant_den) if abs(p) < 1e-4])))
    st.markdown(f"**Plant System Type:** {integrators_plant} "
                f"({'type-' + str(integrators_plant) if integrators_plant > 0 else 'type-0 (no integrator)'})")

# ─────────────────────────────────────────────────────────────
# TAB 5 — Filters
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown(
        "Design and visualise analog or digital filters. "
        "These complement the control loop for signal conditioning and noise rejection."
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_domain  = st.radio("Domain", ["Analog", "Digital"], horizontal=True)
        ftype     = st.selectbox("Filter Type",   FILTER_TYPES)
        ffamily   = st.selectbox("Filter Family", FILTER_FAMILIES)
        forder    = st.slider("Filter Order", 1, 10, 4)

    with fc2:
        if f_domain == "Analog":
            wc1 = st.number_input("Cutoff ω₁ (rad/s)", 0.01, 1000.0, 10.0, 1.0)
            wc2 = None
            if ftype in ("Band Pass", "Band Stop"):
                wc2 = st.number_input("Cutoff ω₂ (rad/s)", 0.01, 1000.0, 50.0, 1.0)
        else:
            fs_filter = st.number_input("Sample Rate (Hz)", 1.0, 10000.0, 1000.0, 10.0)
            fc1_hz = st.number_input("Cutoff f₁ (Hz)", 0.01, fs_filter/2-0.1, min(10.0, fs_filter/4), 0.1)
            fc2_hz = None
            if ftype in ("Band Pass", "Band Stop"):
                fc2_hz = st.number_input("Cutoff f₂ (Hz)", fc1_hz+0.1, fs_filter/2-0.01,
                                         min(100.0, fs_filter/3), 0.1)

    with fc3:
        if ffamily == "Chebyshev Type I":
            ripple = st.slider("Passband Ripple (dB)", 0.1, 10.0, 1.0, 0.1)
        else:
            ripple = 1.0
        show_step = st.checkbox("Show Step Response", True)

    # Design
    try:
        if f_domain == "Analog":
            b_f, a_f = design_analog(ftype, ffamily, forder, wc1,
                                     wc2 if wc2 and wc2 > wc1 else None, ripple)
            w_f, mag_f, ph_f = filter_bode(b_f, a_f, analog=True)
            x_label = "Frequency (rad/s)"
            x_is_log = True
        else:
            wn_norm = fc1_hz / (fs_filter / 2)
            wn_norm2 = fc2_hz / (fs_filter / 2) if fc2_hz else None
            b_f, a_f = design_digital(ftype, ffamily, forder, wn_norm, wn_norm2, ripple)
            w_f, mag_f, ph_f = filter_bode(b_f, a_f, analog=False)
            w_f = w_f / np.pi * (fs_filter / 2)   # normalised → Hz
            x_label = "Frequency (Hz)"
            x_is_log = False

        rows_f = 3 if show_step else 2
        fig_f = make_subplots(rows=rows_f, cols=1, shared_xaxes=(not show_step),
                              vertical_spacing=0.08,
                              subplot_titles=(
                                  ["Magnitude (dB)", "Phase (°)", "Step Response"]
                                  if show_step else ["Magnitude (dB)", "Phase (°)"]
                              ))

        fig_f.add_trace(go.Scatter(x=w_f, y=mag_f, name="Magnitude",
                                   line=dict(color="#2196F3", width=2)), row=1, col=1)
        fig_f.add_trace(go.Scatter(x=w_f, y=ph_f, name="Phase",
                                   line=dict(color="#4CAF50", width=2)), row=2, col=1)

        if show_step:
            t_fs, y_fs = filter_step(b_f, a_f, analog=(f_domain == "Analog"),
                                     t_end=5.0 if f_domain == "Analog" else 0.1)
            fig_f.add_trace(go.Scatter(x=t_fs, y=y_fs, name="Step",
                                       line=dict(color=C["output"], width=2)), row=3, col=1)
            fig_f.update_xaxes(title_text="Time (s)", row=3, col=1, gridcolor=C["grid"])
            fig_f.update_yaxes(title_text="Amplitude", row=3, col=1, gridcolor=C["grid"])

        xtype = "log" if x_is_log else "linear"
        for r in [1, 2]:
            fig_f.update_xaxes(type=xtype, gridcolor=C["grid"], row=r, col=1)
        fig_f.update_xaxes(title_text=x_label, row=2, col=1)
        fig_f.update_yaxes(title_text="dB",  row=1, col=1, gridcolor=C["grid"])
        fig_f.update_yaxes(title_text="°",   row=2, col=1, gridcolor=C["grid"])
        fig_f.update_layout(height=600 if show_step else 450, **PLOT_LAYOUT)
        st.plotly_chart(fig_f, use_container_width=True)

    except Exception as e:
        st.error(f"Filter design error: {e}")

# ─────────────────────────────────────────────────────────────
# TAB 6 — Theory Guide
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## Control Theory Quick Reference")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("### PID Controller")
        st.markdown("""<div class="theory-box">

**Continuous (ideal with derivative filter):**

$$C(s) = K_p + \\frac{K_i}{s} + \\frac{K_d \\cdot N \\cdot s}{s + N}$$

- **Kp** — proportional gain: faster response, more overshoot
- **Ki** — integral gain: eliminates steady-state error, risks oscillation
- **Kd** — derivative gain: damps overshoot, noise-sensitive
- **N** — derivative filter pole (100–1000 typical)

**Discrete (Tustin / bilinear):** maps *s → 2(z−1)/[Tₛ(z+1)]*
</div>""", unsafe_allow_html=True)

        st.markdown("### Second-Order System Characterisation")
        st.markdown("""<div class="theory-box">

$$G(s) = \\frac{K\\omega_n^2}{s^2 + 2\\zeta\\omega_n s + \\omega_n^2}$$

| Damping | Regime |
|---------|--------|
| ζ < 0   | Unstable |
| ζ = 0   | Undamped (oscillates) |
| 0 < ζ < 1 | Underdamped (overshoot) |
| ζ = 1   | Critically damped |
| ζ > 1   | Overdamped |

**Approximate formulas (underdamped):**
- Overshoot: $M_p = e^{-\\pi\\zeta/\\sqrt{1-\\zeta^2}} \\times 100\\%$
- Settling (2%): $T_s ≈ 4/(\\zeta\\omega_n)$
- Rise time: $T_r ≈ 1.8/\\omega_n$
- Damped freq: $\\omega_d = \\omega_n\\sqrt{1-\\zeta^2}$
</div>""", unsafe_allow_html=True)

        st.markdown("### Filter Summary")
        st.markdown("""<div class="theory-box">

| Type | Butterworth | Chebyshev | Bessel |
|------|-------------|-----------|--------|
| Rolloff | Maximally flat | Steeper rolloff | Linear phase |
| Ripple | No | Passband | No |
| Transient | Good | Moderate | Best |

**Standard -3 dB rolloff:**  |H(jωc)| = 1/√2

**Key uses in control:**
- **LPF** — anti-aliasing, sensor noise rejection
- **HPF** — derivative action, DC offset removal
- **BPF** — resonance isolation, fault detection
</div>""", unsafe_allow_html=True)

    with col_t2:
        st.markdown("### Stability Criteria")
        st.markdown("""<div class="theory-box">

**Bode Stability Margins:**
- **Phase Margin (PM)** = 180° + ∠L(jωgc)
  - PM > 45° → good robustness
  - PM ↔ damping ratio: ζ ≈ PM/100 (rough rule)
- **Gain Margin (GM)** = −|L(jωpc)| dB
  - GM > 6 dB → good robustness

**Routh–Hurwitz:** all coefficients of char. poly. must be positive *and* all Routh table entries positive.

**Discrete stability:** all CL poles inside unit circle |z| < 1.
</div>""", unsafe_allow_html=True)

        st.markdown("### Root Locus Rules")
        st.markdown("""<div class="theory-box">

For $1 + K \\cdot L(s) = 0$, where $L$ has $n$ poles and $m$ zeros:

1. **Branches** start at OL poles (K=0), end at OL zeros or ∞ (K→∞)
2. **Real-axis segments**: to the left of an *odd* number of poles+zeros
3. **Asymptotes**: $n − m$ branches go to ∞ at angles $(2k+1)180°/(n−m)$
4. **Centroid** of asymptotes: $\\sigma_a = (\\sum p_i − \\sum z_i)/(n−m)$
5. **Break-away**: solve $dK/ds = 0$
6. Imaginary-axis crossing: substitute $s = j\\omega$ → find K, ω
</div>""", unsafe_allow_html=True)

        st.markdown("### Quick Tuning Guide")
        st.markdown("""<div class="theory-box">

**Starting point (Ziegler–Nichols ultimate gain method):**
1. Set Ki = Kd = 0, increase Kp until sustained oscillation → Ku, Tu
2. PID: Kp = 0.6 Ku,  Ki = 2 Kp/Tu,  Kd = Kp·Tu/8

**Manual tuning rules of thumb:**
| Change | Effect |
|--------|--------|
| ↑ Kp   | Faster, more overshoot, less SS error |
| ↑ Ki   | Eliminates SS error, slower, can destabilise |
| ↑ Kd   | Reduces overshoot, faster settling, noise-sensitive |
| ↑ N    | Faster derivative response, higher noise pass-through |

**Target specs:** PM 45–60°, GM > 6 dB, OS < 10–20%.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**Coming soon:** LQR · LQG · MPC · Kalman Filter — same interactive approach,
showing state-space formulations, optimal gain computation, and observer design.
    """)
