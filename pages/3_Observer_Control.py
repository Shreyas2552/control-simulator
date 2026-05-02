"""
Observer-Based Control — Luenberger Observer + Full-State Feedback
==================================================================
Design deterministic state observers (Luenberger) via pole placement,
combine with state-feedback for closed-loop control, and compare against
the Kalman-based LQG approach already shown on the LQR/LQG page.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.linalg import eigvals, solve_continuous_are
from scipy.signal import place_poles

from modules.lqr_lqg import (
    SS_PLANTS, get_ss,
    controllability_rank, observability_rank,
    lqr_design, kalman_design, compute_nbar,
    simulate_lqr,
)

st.set_page_config(
    page_title="Observer-Based Control",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px; border-radius: 6px 6px 0 0; font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background: #1e2130; border-radius: 8px; padding: 10px 14px;
    }
    .theory-box {
        background: #1a1f2e;
        border-left: 4px solid #9C27B0;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .compare-box {
        background: #162032;
        border-left: 4px solid #FF9800;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

C = dict(
    true    = "#4CAF50",
    obs     = "#FF9800",
    meas    = "#F44336",
    ctrl    = "#9C27B0",
    lqg     = "#2196F3",
    ref     = "#90A4AE",
    stable  = "#4CAF50",
    unstable= "#F44336",
    grid    = "rgba(255,255,255,0.07)",
)
PLOT = dict(
    template     = "plotly_dark",
    plot_bgcolor  = "#0e1117",
    paper_bgcolor = "#0e1117",
    font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin=dict(l=60, r=30, t=40, b=50),
)
_DT = 0.005

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ Observer Control")
    st.markdown("---")

    plant_name = st.selectbox("Plant Model (State-Space)", list(SS_PLANTS.keys()))
    info = SS_PLANTS[plant_name]
    st.caption(f"**{info['desc']}**")
    st.caption(f"📌 {info['physical']}")

    plant_params = {}
    for pname, pm in info["params"].items():
        plant_params[pname] = st.slider(
            pm["label"], float(pm["min"]), float(pm["max"]),
            float(pm["default"]), float(pm["step"]),
        )
    st.markdown("---")

    n_states = info["n_states"]
    snames   = info["state_names"]

    # Controller poles (for full-state feedback via pole placement)
    st.markdown("### Controller Poles (A − BK)")
    st.caption("Place poles in left-half plane. More negative = faster.")
    ctrl_poles = []
    n_pairs = n_states // 2
    n_real  = n_states % 2
    for i in range(n_pairs):
        sigma = st.slider(f"Pole pair {i+1} — real part σ",
                          -50.0, -0.1, float(info["default_Q"][2*i] * -0.5 - 1.0), 0.1,
                          key=f"cp_sigma_{i}")
        omega = st.slider(f"Pole pair {i+1} — imag part ω",
                          0.0, 30.0, max(0.5, abs(sigma) * 0.5), 0.1,
                          key=f"cp_omega_{i}")
        ctrl_poles.extend([complex(sigma,  omega), complex(sigma, -omega)])
    for i in range(n_real):
        rp = st.slider(f"Real pole {n_pairs + i + 1}",
                       -50.0, -0.1, -2.0, 0.1, key=f"cp_real_{i}")
        ctrl_poles.append(complex(rp, 0.0))
    st.markdown("---")

    # Observer poles (A − LC) — typically 2–5× faster than controller
    st.markdown("### Observer Poles (A − LC)")
    st.caption("Observer poles should be 2–5× faster than controller poles.")
    obs_poles = []
    for i in range(n_pairs):
        o_sigma = st.slider(f"Obs pair {i+1} — real σ",
                            -100.0, -0.1,
                            float(info["default_Q"][2*i] * -1.5 - 2.0), 0.1,
                            key=f"op_sigma_{i}")
        o_omega = st.slider(f"Obs pair {i+1} — imag ω",
                            0.0, 50.0, max(0.5, abs(o_sigma) * 0.5), 0.1,
                            key=f"op_omega_{i}")
        obs_poles.extend([complex(o_sigma,  o_omega), complex(o_sigma, -o_omega)])
    for i in range(n_real):
        op = st.slider(f"Obs real pole {n_pairs + i + 1}",
                       -100.0, -0.1, -5.0, 0.1, key=f"op_real_{i}")
        obs_poles.append(complex(op, 0.0))
    st.markdown("---")

    # Noise (for comparison with Kalman)
    st.markdown("### Noise (for LQG comparison)")
    q_std = st.select_slider("Process noise σw",
        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], value=0.05)
    r_std = st.select_slider("Measurement noise σv",
        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], value=0.1)
    st.markdown("---")

    ref_val = st.number_input("Reference (setpoint)", -10.0, 10.0, 1.0, 0.1)
    t_end   = st.slider("Duration (s)", 2.0, 60.0, 15.0, 0.5)


# ── Compute plant matrices ─────────────────────────────────────────────────
A, B, C_out, D = get_ss(plant_name, plant_params)
C_obs = C_out.copy()
n = A.shape[0]
ol_eigs = eigvals(A)
ctrl_rank = controllability_rank(A, B)
obs_rank  = observability_rank(A, C_obs)

# ── Design controllers ─────────────────────────────────────────────────────
pp_ok = obs_ok = lqg_ok = False
K_pp = L_pp = K_lqr = L_kal = None
pp_err = obs_err = lqg_err = None

# Pole placement — controller
try:
    pp_result = place_poles(A, B, np.array(ctrl_poles))
    K_pp = pp_result.gain_matrix
    cl_eigs_pp = eigvals(A - B @ K_pp)
    pp_ok = True
except Exception as e:
    pp_err = f"Controller pole placement failed: {e}"

# Pole placement — observer
try:
    obs_result = place_poles(A.T, C_obs.T, np.array(obs_poles))
    L_pp = obs_result.gain_matrix.T
    obs_cl_eigs_pp = eigvals(A - L_pp @ C_obs)
    obs_ok = True
except Exception as e:
    obs_err = f"Observer pole placement failed: {e}"

# LQR + Kalman (for comparison)
try:
    Q_lqr = np.diag(info["default_Q"])
    R_lqr = np.array([[info["default_R"]]])
    K_lqr, _, cl_eigs_lqr = lqr_design(A, B, Q_lqr, R_lqr)
    Qn = np.eye(n) * q_std**2
    Rn = np.array([[r_std**2]])
    L_kal, _ = kalman_design(A, C_obs, Qn, Rn)
    lqg_ok = True
except Exception as e:
    lqg_err = f"LQG design failed: {e}"

# ── Simulation — Luenberger observer + state-feedback ─────────────────────
def simulate_observer(A, B, C_obs, K, L, Nbar, ref, t_end, ref_state,
                       q_std=0.0, r_std=0.0, seed=42):
    """Euler integration: true plant + Luenberger observer + state-feedback."""
    t   = np.arange(0., t_end, _DT)
    N   = len(t)
    n_s = A.shape[0]
    p   = C_obs.shape[0]

    x_true = np.zeros((N, n_s))
    x_est  = np.zeros((N, n_s))
    y_meas = np.zeros((N, p))
    u_hist = np.zeros(N)

    x_ref = np.zeros(n_s)
    x_ref[ref_state] = ref
    rng = np.random.default_rng(seed)

    for k in range(N - 1):
        u_k = (-K @ (x_est[k] - x_ref)).item()
        u_hist[k] = u_k

        w = rng.normal(0., q_std, n_s)
        x_true[k+1] = x_true[k] + _DT * ((A @ x_true[k]).flatten()
                                           + (B * u_k).flatten() + w)

        v = rng.normal(0., r_std, p)
        y_meas[k] = C_obs @ x_true[k] + v

        innov = y_meas[k] - C_obs @ x_est[k]
        x_est[k+1] = x_est[k] + _DT * ((A @ x_est[k]).flatten()
                                         + (B * u_k).flatten()
                                         + (L @ innov).flatten())

    y_meas[-1] = C_obs @ x_true[-1]
    u_hist[-1]  = u_hist[-2]
    return t, x_true, x_est, y_meas, u_hist


ref_state = info["ref_state"]
t_sim = x_true_pp = x_est_pp = y_meas_pp = u_pp = None
t_lqg = x_true_lqg = x_est_lqg = y_meas_lqg = u_lqg = None

if pp_ok and obs_ok:
    Nbar_pp = compute_nbar(A, B, C_out[0:1, :], K_pp)
    try:
        t_sim, x_true_pp, x_est_pp, y_meas_pp, u_pp = simulate_observer(
            A, B, C_obs, K_pp, L_pp, Nbar_pp, ref_val, t_end, ref_state,
            q_std=q_std, r_std=r_std,
        )
    except Exception as e:
        pp_err = f"Observer simulation failed: {e}"

if lqg_ok:
    Nbar_lqg = compute_nbar(A, B, C_out[0:1, :], K_lqr)
    try:
        t_lqg, x_true_lqg, x_est_lqg, y_meas_lqg, u_lqg = simulate_observer(
            A, B, C_obs, K_lqr, L_kal, Nbar_lqg, ref_val, t_end, ref_state,
            q_std=q_std, r_std=r_std,
        )
    except Exception as e:
        lqg_err = f"LQG simulation failed: {e}"

# ── Header ─────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("## 👁️ Observer-Based Control")
    st.markdown(
        f"**Plant:** {info['desc']}  |  **n={n}**  |  "
        f"Controllable: {'✅' if ctrl_rank==n else '❌'}  |  "
        f"Observable: {'✅' if obs_rank==n else '❌'}"
    )
with h2:
    ol_stable = all(e.real < 0 for e in ol_eigs)
    st.markdown(f"### {'🟢 Stable OL' if ol_stable else '🔴 Unstable OL'}")
    if pp_ok:
        cl_ok_pp = all(e.real < 0 for e in cl_eigs_pp)
        st.markdown(f"PP CL: {'🟢 Stable' if cl_ok_pp else '🔴 Unstable'}")

st.markdown("---")
if pp_err:  st.error(pp_err)
if obs_err: st.error(obs_err)
if lqg_err: st.warning(lqg_err)

# ── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔧 Design",
    "📈 Observer Response",
    "⚖️ Luenberger vs LQG",
    "📚 Theory",
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — Design
# ──────────────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Controller Gain K  (pole placement)")
        if pp_ok:
            st.dataframe(pd.DataFrame(np.round(K_pp, 5),
                         index=info["input_names"], columns=snames),
                         use_container_width=True)
            st.caption("u = −K·x̂  +  Nbar·r")

            st.markdown("#### Closed-Loop Poles  (A − BK)")
            cl_df = pd.DataFrame([{
                "Pole": f"{e.real:+.4f}" + (f" ± {abs(e.imag):.4f}j" if abs(e.imag) > 1e-4 else ""),
                "Re": f"{e.real:.4f}",
                "|λ|": f"{abs(e):.4f}",
                "ζ": f"{-e.real/abs(e):.3f}" if abs(e) > 1e-9 else "—",
                "Stable": "✅" if e.real < 0 else "❌",
            } for e in cl_eigs_pp])
            st.dataframe(cl_df, use_container_width=True, hide_index=True)
        else:
            st.error(pp_err)

    with c2:
        st.markdown("#### Observer Gain L  (Luenberger)")
        if obs_ok:
            st.dataframe(pd.DataFrame(np.round(L_pp, 5),
                         index=snames, columns=info["output_names"]),
                         use_container_width=True)
            st.caption("x̂̇ = Ax̂ + Bu + L(y − Cx̂)")

            st.markdown("#### Observer Poles  (A − LC)")
            obs_df = pd.DataFrame([{
                "Pole": f"{e.real:+.4f}" + (f" ± {abs(e.imag):.4f}j" if abs(e.imag) > 1e-4 else ""),
                "Re": f"{e.real:.4f}",
                "|λ|": f"{abs(e):.4f}",
                "Stable": "✅" if e.real < 0 else "❌",
            } for e in obs_cl_eigs_pp])
            st.dataframe(obs_df, use_container_width=True, hide_index=True)
        else:
            st.error(obs_err)

    # Pole map: OL vs CL vs Observer
    st.markdown("#### Pole Map — Open-Loop / CL Controller / Observer")
    fig_poles = go.Figure()
    fig_poles.add_shape(type="line", x0=0, x1=0, y0=-60, y1=60,
                        line=dict(color="rgba(255,255,255,0.25)", dash="dot"))
    fig_poles.add_trace(go.Scatter(
        x=[e.real for e in ol_eigs], y=[e.imag for e in ol_eigs],
        mode="markers", name="Open-Loop",
        marker=dict(symbol="x", size=16, color="#F44336", line=dict(width=3)),
    ))
    if pp_ok:
        fig_poles.add_trace(go.Scatter(
            x=[e.real for e in cl_eigs_pp], y=[e.imag for e in cl_eigs_pp],
            mode="markers", name="CL Controller (PP)",
            marker=dict(symbol="star", size=18, color="#FF9800",
                        line=dict(width=2, color="white")),
        ))
    if obs_ok:
        fig_poles.add_trace(go.Scatter(
            x=[e.real for e in obs_cl_eigs_pp], y=[e.imag for e in obs_cl_eigs_pp],
            mode="markers", name="Observer (PP)",
            marker=dict(symbol="diamond", size=14, color="#9C27B0",
                        line=dict(width=2, color="white")),
        ))
    if lqg_ok:
        fig_poles.add_trace(go.Scatter(
            x=[e.real for e in cl_eigs_lqr], y=[e.imag for e in cl_eigs_lqr],
            mode="markers", name="CL Controller (LQR)",
            marker=dict(symbol="circle", size=12, color="#2196F3",
                        line=dict(width=2, color="white")),
        ))
        obs_lqg_eigs = eigvals(A - L_kal @ C_obs)
        fig_poles.add_trace(go.Scatter(
            x=[e.real for e in obs_lqg_eigs], y=[e.imag for e in obs_lqg_eigs],
            mode="markers", name="Observer (Kalman)",
            marker=dict(symbol="diamond-open", size=14, color="#2196F3",
                        line=dict(width=2)),
        ))
    fig_poles.update_layout(xaxis_title="Re(s)", yaxis_title="Im(s)",
                            yaxis_scaleanchor="x", height=380, **PLOT)
    st.plotly_chart(fig_poles, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — Observer Response
# ──────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if t_sim is None:
        st.error(pp_err or "Simulation unavailable.")
    else:
        rs = ref_state
        y_true = x_true_pp[:, rs]
        y_est  = x_est_pp[:, rs]
        y_meas_flat = y_meas_pp.flatten()

        # Metrics
        est_rmse = float(np.sqrt(np.mean((y_est - y_true)**2)))
        meas_rmse = float(np.sqrt(np.mean((y_meas_flat - y_true)**2)))
        try:
            outside = np.where(np.abs(y_true - ref_val) > 0.02 * abs(ref_val))[0]
            ts_obs  = float(t_sim[outside[-1]]) if len(outside) else 0.0
        except Exception:
            ts_obs = None
        ss_err = float(ref_val - y_true[-1])

        mc = st.columns(4)
        mc[0].metric("SS Error",        f"{ss_err:.5f}")
        mc[1].metric("Settling (2%) s", f"{ts_obs:.3f}" if ts_obs else "N/A")
        mc[2].metric("Observer RMSE",   f"{est_rmse:.5f}")
        mc[3].metric("Meas. RMSE",      f"{meas_rmse:.5f}")

        # Plot: true / estimated / measurement / control
        n_plot = min(n, 4)
        fig_r = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Tracked state — True / Observer estimate / Noisy measurement",
                            "All states — True (solid) vs Estimated (dashed)",
                            "Control signal u(t)"),
        )
        fig_r.add_trace(go.Scatter(x=t_sim, y=np.full_like(t_sim, ref_val),
                                   name="Reference",
                                   line=dict(color=C["ref"], dash="dash", width=1)),
                        row=1, col=1)
        fig_r.add_trace(go.Scatter(x=t_sim, y=y_true, name="True state",
                                   line=dict(color=C["true"], width=2.5)), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=t_sim, y=y_est, name="Observer estimate",
                                   line=dict(color=C["obs"], width=2, dash="dash")),
                        row=1, col=1)
        fig_r.add_trace(go.Scatter(x=t_sim, y=y_meas_flat, name="Noisy measurement",
                                   line=dict(color=C["meas"], width=0.8, dash="dot"),
                                   opacity=0.6), row=1, col=1)

        palette = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
        for si in range(n_plot):
            fig_r.add_trace(go.Scatter(x=t_sim, y=x_true_pp[:, si],
                                       name=f"True {snames[si].split()[0]}",
                                       line=dict(color=palette[si % 4], width=2)),
                            row=2, col=1)
            fig_r.add_trace(go.Scatter(x=t_sim, y=x_est_pp[:, si],
                                       name=f"Est {snames[si].split()[0]}",
                                       line=dict(color=palette[si % 4], width=1.5, dash="dash"),
                                       showlegend=False), row=2, col=1)

        fig_r.add_trace(go.Scatter(x=t_sim, y=u_pp, name="u(t)",
                                   line=dict(color=C["ctrl"], width=2)), row=3, col=1)
        for r in [1, 2, 3]:
            fig_r.update_xaxes(gridcolor=C["grid"], row=r, col=1)
            fig_r.update_yaxes(gridcolor=C["grid"], row=r, col=1)
        fig_r.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_r.update_layout(height=680, **PLOT)
        st.plotly_chart(fig_r, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — Luenberger vs LQG
# ──────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if t_sim is None or t_lqg is None:
        st.warning("Both Luenberger and LQG simulations must succeed for comparison.")
    else:
        rs = ref_state
        fig_cmp = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("Tracked state — Luenberger vs LQG vs Noisy measurement",
                            "Observer estimation error  |x − x̂|"),
        )
        fig_cmp.add_trace(go.Scatter(x=t_sim, y=np.full_like(t_sim, ref_val),
                                     name="Reference",
                                     line=dict(color=C["ref"], dash="dash", width=1)),
                          row=1, col=1)
        fig_cmp.add_trace(go.Scatter(x=t_sim, y=x_true_pp[:, rs],
                                     name="True (Luenberger run)",
                                     line=dict(color=C["true"], width=2.5)), row=1, col=1)
        fig_cmp.add_trace(go.Scatter(x=t_sim, y=x_est_pp[:, rs],
                                     name="Luenberger estimate",
                                     line=dict(color=C["obs"], width=2, dash="dash")),
                          row=1, col=1)
        fig_cmp.add_trace(go.Scatter(x=t_lqg, y=x_est_lqg[:, rs],
                                     name="Kalman estimate (LQG)",
                                     line=dict(color=C["lqg"], width=2, dash="dash")),
                          row=1, col=1)
        fig_cmp.add_trace(go.Scatter(x=t_sim, y=y_meas_pp.flatten(),
                                     name="Noisy measurement",
                                     line=dict(color=C["meas"], width=0.7, dash="dot"),
                                     opacity=0.5), row=1, col=1)

        # Estimation error per state
        err_luen = np.sqrt(np.sum((x_true_pp - x_est_pp)**2, axis=1))
        err_kal  = np.sqrt(np.sum((x_true_lqg - x_est_lqg)**2, axis=1))
        fig_cmp.add_trace(go.Scatter(x=t_sim, y=err_luen,
                                     name="Luenberger ‖e‖",
                                     line=dict(color=C["obs"], width=2)), row=2, col=1)
        fig_cmp.add_trace(go.Scatter(x=t_lqg, y=err_kal,
                                     name="Kalman ‖e‖",
                                     line=dict(color=C["lqg"], width=2)), row=2, col=1)
        for r in [1, 2]:
            fig_cmp.update_xaxes(gridcolor=C["grid"], row=r, col=1)
            fig_cmp.update_yaxes(gridcolor=C["grid"], row=r, col=1)
        fig_cmp.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_cmp.update_layout(height=520, **PLOT)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Summary table
        luen_rmse = float(np.sqrt(np.mean((x_true_pp[:, rs] - x_est_pp[:, rs])**2)))
        kal_rmse  = float(np.sqrt(np.mean((x_true_lqg[:, rs] - x_est_lqg[:, rs])**2)))
        luen_ss = float(x_true_pp[-1, rs])
        kal_ss  = float(x_true_lqg[-1, rs])

        cmp_df = pd.DataFrame([
            {"Metric": "Est. RMSE (tracked state)", "Luenberger": f"{luen_rmse:.5f}", "Kalman (LQG)": f"{kal_rmse:.5f}"},
            {"Metric": "SS value",                  "Luenberger": f"{luen_ss:.5f}",   "Kalman (LQG)": f"{kal_ss:.5f}"},
            {"Metric": "Observer design method",    "Luenberger": "Pole placement",    "Kalman (LQG)": "Riccati / noise covariances"},
            {"Metric": "Requires noise model?",     "Luenberger": "No",                "Kalman (LQG)": "Yes (Qn, Rn)"},
            {"Metric": "Optimal (stochastic)?",     "Luenberger": "No",                "Kalman (LQG)": "Yes"},
        ])
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.markdown("""<div class="compare-box">

**When Luenberger beats Kalman (in practice)**
- You don't have a good noise model but want deterministic, predictable observer dynamics
- You need a specific transient shape (e.g., critically damped) and want to place observer poles exactly
- Real-time tuning is easier: directly specify desired bandwidth via pole locations

**When Kalman beats Luenberger**
- Sensor noise or disturbance statistics are known (e.g., from sensor datasheets)
- System is stochastic and you need the minimum-variance estimate
- Multiple sensors with different noise levels → Kalman automatically weights them optimally

**Rule of thumb:** Place observer poles 2–5× further left (faster) than controller poles so estimation errors decay faster than the closed-loop dynamics. Too fast → amplifies sensor noise; too slow → estimation lag degrades control.
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — Theory
# ──────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## Observer-Based Control — Theory")
    t1, t2 = st.columns(2)

    with t1:
        st.markdown("### The Observer Problem")
        st.markdown("""<div class="theory-box">

**Why do we need an observer?**
Full-state feedback `u = −Kx` requires all $n$ states. In practice only a few
outputs $y = Cx$ are measured. An observer reconstructs the unmeasured states.

**Luenberger Observer (deterministic)**
$$\\dot{\\hat{x}} = A\\hat{x} + Bu + L(y - C\\hat{x})$$

- Without L: open-loop predictor (drifts with model error)
- L drives the estimation error $e = x - \\hat{x}$ to zero:
$$\\dot{e} = (A - LC)\\, e$$

**Choose L so that eig(A − LC) are stable and fast enough.**

**Pole placement for L**
Desired observer poles $\\{p_1, p_2, \\ldots\\}$ → use `scipy.signal.place_poles(A.T, C.T, poles).gain_matrix.T`

This is the **dual** of placing controller poles with `place_poles(A, B, poles)`.
</div>""", unsafe_allow_html=True)

        st.markdown("### Separation Principle")
        st.markdown("""<div class="theory-box">

The combined plant + observer + controller system has eigenvalues:

$$\\text{eig}(A - BK) \\cup \\text{eig}(A - LC)$$

They are **completely decoupled** — you can design K and L independently.
This holds for any linear time-invariant system that is controllable and observable.

**Implication:** Design the controller (K) for performance, then design the
observer (L) for estimation speed. Neither design affects the other's poles.

**Combined system state-space (augmented form):**
$$\\begin{bmatrix} \\dot{x} \\\\ \\dot{e} \\end{bmatrix} =
\\begin{bmatrix} A-BK & BK \\\\ 0 & A-LC \\end{bmatrix}
\\begin{bmatrix} x \\\\ e \\end{bmatrix}$$
</div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("### Observer Pole Placement Rules")
        st.markdown("""<div class="theory-box">

| Guideline | Reason |
|-----------|--------|
| Observer poles 2–5× faster than controller | Error decays before it degrades control |
| Don't place observer poles too far left | Amplifies sensor noise (high L gains) |
| Use complex observer poles with ζ ≈ 0.7 | Avoid oscillatory estimation errors |
| Check condition number of observability matrix | Ill-conditioned → numerical issues in L |

**Observer bandwidth trade-off:**
- **Too slow** → estimation lag, controller acts on stale state → poor performance
- **Too fast** → large L gains, sensor noise is amplified into state estimate → chattering

The Kalman filter resolves this trade-off optimally when noise statistics are known.
</div>""", unsafe_allow_html=True)

        st.markdown("### Luenberger vs Kalman Summary")
        st.markdown("""<div class="theory-box">

| Property | Luenberger | Kalman |
|----------|------------|--------|
| Design input | Desired pole locations | Noise covariances Qn, Rn |
| Optimality | Not guaranteed | Minimum variance |
| Requires noise model | No | Yes |
| Easy to tune intuitively | Yes | Requires physical understanding of noise |
| Multiple sensors | Manual weighting | Automatic optimal fusion |
| Nonlinear extension | Sliding-mode observer | EKF / UKF |

**Practical advice**
For prototyping or deterministic plants → Luenberger (simpler, direct pole control).
For noisy sensors or stochastic systems → Kalman (statistically optimal, handles multi-sensor).
For nonlinear systems → Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF).
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Coming next:** Sliding-mode observers, Extended Kalman Filter (EKF) for nonlinear plants.")
