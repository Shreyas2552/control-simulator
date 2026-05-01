"""
LQR / Kalman Filter / LQG Interactive Simulator
================================================
State-space plants, optimal control via LQR, state estimation via Kalman filter,
and the full LQG loop — all interactive and annotated for learning.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.linalg import eigvals

from modules.lqr_lqg import (
    SS_PLANTS, get_ss,
    controllability_rank, observability_rank,
    lqr_design, kalman_design,
    compute_nbar,
    simulate_lqr, simulate_lqg,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LQR / LQG Simulator",
    page_icon="🎯",
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
        border-left: 4px solid #FF9800;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .info-box {
        background: #162032;
        border-left: 4px solid #2196F3;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour palette (consistent with PID page) ─────────────────────────────
C = dict(
    lqr_out   = "#4CAF50",
    lqg_out   = "#2196F3",
    x_true    = "#4CAF50",
    x_est     = "#FF9800",
    y_meas    = "#F44336",
    u_lqr     = "#9C27B0",
    u_lqg     = "#00BCD4",
    ref       = "#90A4AE",
    stable    = "#4CAF50",
    unstable  = "#F44336",
    grid      = "rgba(255,255,255,0.07)",
)

PLOT = dict(
    template    = "plotly_dark",
    plot_bgcolor = "#0e1117",
    paper_bgcolor= "#0e1117",
    font = dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend = dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin = dict(l=60, r=30, t=40, b=50),
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 LQR / LQG Simulator")
    st.markdown("---")

    # Plant
    st.markdown("### Plant Model (State-Space)")
    plant_name = st.selectbox("Plant", list(SS_PLANTS.keys()), label_visibility="collapsed")
    info = SS_PLANTS[plant_name]
    st.caption(f"**{info['desc']}**")
    st.caption(f"📌 {info['physical']}")

    plant_params = {}
    for pname, pm in info["params"].items():
        plant_params[pname] = st.slider(
            pm["label"],
            float(pm["min"]), float(pm["max"]),
            float(pm["default"]), float(pm["step"]),
        )
    st.markdown("---")

    # LQR weights
    st.markdown("### LQR Cost Weights")
    st.caption("J = ∫(x'Qx + u'Ru) dt")
    n_states = info["n_states"]
    snames   = info["state_names"]
    def_Q    = info["default_Q"]

    q_vals = []
    for i in range(n_states):
        q_vals.append(st.number_input(
            f"Q[{snames[i].split()[0]}]",
            min_value=0.0, max_value=10000.0,
            value=float(def_Q[i]), step=1.0, format="%.1f",
        ))
    Q_mat = np.diag(q_vals)

    R_val = st.number_input("R (input cost)", min_value=1e-4, max_value=1000.0,
                            value=float(info["default_R"]), step=0.01, format="%.3f")
    R_mat = np.array([[R_val]])
    st.markdown("---")

    # Kalman noise
    st.markdown("### Kalman Filter Noise")
    st.caption("Higher Qn → trust plant model less  |  Higher Rn → trust sensor less")
    q_std = st.select_slider(
        "Process noise std (σw)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        value=0.05,
    )
    r_std = st.select_slider(
        "Measurement noise std (σv)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        value=0.1,
    )
    Qn_mat = np.eye(n_states) * q_std**2
    Rn_mat = np.array([[r_std**2]])
    st.markdown("---")

    # Simulation
    st.markdown("### Simulation")
    ref_val = st.number_input("Reference (setpoint)", min_value=-10.0, max_value=10.0,
                              value=1.0, step=0.1)
    t_end   = st.slider("Duration (s)", 2.0, 60.0, 15.0, 0.5)

# ── Compute ────────────────────────────────────────────────────────────────
A, B, C_out, D = get_ss(plant_name, plant_params)
C_obs = C_out.copy()         # same output matrix used for observation

ol_eigs = eigvals(A)
ctrl_rank = controllability_rank(A, B)
obs_rank  = observability_rank(A, C_obs)
n = A.shape[0]

lqr_ok = kalman_ok = False
K = P_lqr = cl_eigs_lqr = None
L = Pe = None
t_lqr = x_lqr = y_lqr = u_lqr = None
t_lqg = x_true = x_est = y_meas = u_lqg_h = None

lqr_err = kalman_err = None

if ctrl_rank < n:
    lqr_err = f"System not fully controllable (rank {ctrl_rank} < {n}). LQR may not stabilise all modes."

try:
    K, P_lqr, cl_eigs_lqr = lqr_design(A, B, Q_mat, R_mat)
    lqr_ok = True
except Exception as e:
    lqr_err = f"LQR Riccati solver failed: {e}"

if lqr_ok:
    ref_state = info["ref_state"]
    Nbar = compute_nbar(A, B, C_out[0:1, :], K)
    try:
        t_lqr, x_lqr, y_lqr, u_lqr = simulate_lqr(
            A, B, C_out, K, Nbar, ref_val, t_end, ref_state=ref_state)
    except Exception as e:
        lqr_err = f"LQR simulation failed: {e}"

if obs_rank < n:
    kalman_err = f"System not fully observable (rank {obs_rank} < {n}). Kalman may not estimate all states."

try:
    L, Pe = kalman_design(A, C_obs, Qn_mat, Rn_mat)
    kalman_ok = True
except Exception as e:
    kalman_err = f"Kalman Riccati solver failed: {e}"

if lqr_ok and kalman_ok:
    try:
        t_lqg, x_true, x_est, y_meas, u_lqg_h = simulate_lqg(
            A, B, C_out, C_obs, K, L, Nbar,
            ref_val, t_end, q_std, r_std,
            ref_state=ref_state,
        )
    except Exception as e:
        kalman_err = f"LQG simulation failed: {e}"

# Observer closed-loop eigenvalues (A - L·C)
obs_cl_eigs = eigvals(A - L @ C_obs) if kalman_ok else None

# ── Helpers ────────────────────────────────────────────────────────────────
def eig_color(e):
    return C["stable"] if e.real < -1e-6 else C["unstable"]

def fmt_eig(e):
    if abs(e.imag) > 1e-4:
        return f"{e.real:+.4f} ± {abs(e.imag):.4f}j"
    return f"{e.real:+.4f}"

def make_pole_scatter(eigs, name, symbol="x", size=14, colour=None, dash_unit=False):
    colours = [eig_color(e) for e in eigs] if colour is None else colour
    return go.Scatter(
        x=[e.real for e in eigs], y=[e.imag for e in eigs],
        mode="markers", name=name,
        marker=dict(symbol=symbol, size=size, color=colours,
                    line=dict(width=2.5, color="white")),
    )

# ── Header ────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("## 🎯 LQR / Kalman Filter / LQG Simulator")
    st.markdown(
        f"**Plant:** {info['desc']}  |  "
        f"**States:** {n}  |  "
        f"**Controllable:** {'✅' if ctrl_rank==n else '❌'}  |  "
        f"**Observable:** {'✅' if obs_rank==n else '❌'}"
    )
with h2:
    ol_stable = all(e.real < 0 for e in ol_eigs)
    badge = ("🟢 **STABLE** (open-loop)" if ol_stable else "🔴 **UNSTABLE** (open-loop)")
    st.markdown(f"### {badge}")
    if lqr_ok and cl_eigs_lqr is not None:
        cl_stable = all(e.real < 0 for e in cl_eigs_lqr)
        st.markdown(f"LQR CL: {'🟢 Stable' if cl_stable else '🔴 Unstable'}")

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────--
tabs = st.tabs([
    "📐 System Model",
    "🎛️ LQR Design",
    "📈 LQR Response",
    "🔭 Kalman Filter",
    "🔗 LQG Full Loop",
    "📊 Pole Comparison",
    "📚 Theory",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — System Model
# ─────────────────────────────────────────────────────────────────────────
with tabs[0]:
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("#### System Matrices")

        def mat_df(M, rnames, cnames):
            return pd.DataFrame(
                np.round(M, 5), index=rnames, columns=cnames
            )

        st.markdown("**A  (system / state matrix)**")
        st.dataframe(mat_df(A, snames, snames), use_container_width=True)

        st.markdown("**B  (input matrix)**")
        st.dataframe(mat_df(B, snames, info["input_names"]), use_container_width=True)

        st.markdown("**C  (output matrix)**")
        st.dataframe(mat_df(C_out, info["output_names"], snames), use_container_width=True)

    with mc2:
        st.markdown("#### Open-Loop Eigenvalues of A")
        ol_df = pd.DataFrame([{
            "Eigenvalue":   fmt_eig(e),
            "Real Part":    f"{e.real:.5f}",
            "Imag Part":    f"{e.imag:.5f}",
            "Magnitude":    f"{abs(e):.5f}",
            "Stable":       "✅" if e.real < 0 else "❌",
        } for e in ol_eigs])
        st.dataframe(ol_df, use_container_width=True, hide_index=True)

        st.markdown("#### Controllability & Observability")
        co_df = pd.DataFrame([
            {"Property": "Controllability Rank",  "Value": ctrl_rank, "Full Rank?": "✅" if ctrl_rank==n else "❌"},
            {"Property": "Observability Rank",    "Value": obs_rank,  "Full Rank?": "✅" if obs_rank==n else "❌"},
            {"Property": "System Dimension (n)",  "Value": n,         "Full Rank?": "—"},
        ])
        st.dataframe(co_df, use_container_width=True, hide_index=True)

        # Pole-zero map (open-loop)
        st.markdown("#### Open-Loop Poles")
        fig_ol = go.Figure()
        fig_ol.add_shape(type="line", x0=0, x1=0, y0=-20, y1=20,
                         line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.5))
        fig_ol.add_trace(go.Scatter(
            x=[e.real for e in ol_eigs], y=[e.imag for e in ol_eigs],
            mode="markers", name="OL Poles",
            marker=dict(symbol="x", size=16,
                        color=[eig_color(e) for e in ol_eigs],
                        line=dict(width=3)),
        ))
        fig_ol.update_layout(xaxis_title="Re", yaxis_title="Im",
                             yaxis_scaleanchor="x", height=280, **PLOT)
        st.plotly_chart(fig_ol, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — LQR Design
# ─────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if not lqr_ok:
        st.error(lqr_err or "LQR design failed.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### LQR Gain Matrix  K")
            K_df = pd.DataFrame(
                np.round(K, 6),
                index=info["input_names"],
                columns=snames,
            )
            st.dataframe(K_df, use_container_width=True)
            st.caption("Control law:  **u = −K·x**  (or  u = −K·(x−x_ref) for setpoint)")

            st.markdown("#### Riccati Solution  P")
            P_df = pd.DataFrame(np.round(P_lqr, 4), index=snames, columns=snames)
            st.dataframe(P_df, use_container_width=True)
            st.caption("P solves:  A'P + PA − P·B·R⁻¹·B'·P + Q = 0")

        with c2:
            st.markdown("#### Closed-Loop Eigenvalues  (A − BK)")
            cl_df = pd.DataFrame([{
                "Eigenvalue":  fmt_eig(e),
                "Re":          f"{e.real:.5f}",
                "Im":          f"{e.imag:.5f}",
                "|λ|":         f"{abs(e):.5f}",
                "ζ":           f"{-e.real/abs(e):.4f}" if abs(e) > 1e-10 else "—",
                "Stable":      "✅" if e.real < 0 else "❌",
            } for e in cl_eigs_lqr])
            st.dataframe(cl_df, use_container_width=True, hide_index=True)

            # OL vs CL pole comparison plot
            st.markdown("#### Pole Map: Open-Loop vs Closed-Loop")
            fig_pz = go.Figure()
            fig_pz.add_shape(type="line", x0=0, x1=0, y0=-50, y1=50,
                             line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.5))
            fig_pz.add_trace(go.Scatter(
                x=[e.real for e in ol_eigs], y=[e.imag for e in ol_eigs],
                mode="markers", name="OL Poles",
                marker=dict(symbol="x", size=16, color="#F44336", line=dict(width=3)),
            ))
            fig_pz.add_trace(go.Scatter(
                x=[e.real for e in cl_eigs_lqr], y=[e.imag for e in cl_eigs_lqr],
                mode="markers", name="LQR CL Poles",
                marker=dict(symbol="star", size=18, color="#FF9800", line=dict(width=2, color="white")),
            ))
            # Damping lines
            for zeta_v in [0.2, 0.4, 0.7, 1.0]:
                ang  = np.arccos(zeta_v)
                r_m  = max(abs(cl_eigs_lqr).max(), abs(ol_eigs).max()) * 1.5 + 1
                xv   = [-r_m*zeta_v, 0, -r_m*zeta_v]
                yv   = [ r_m*np.sin(ang), 0, -r_m*np.sin(ang)]
                fig_pz.add_trace(go.Scatter(x=xv, y=yv, mode="lines", showlegend=False,
                                            line=dict(color="rgba(255,255,255,0.1)", dash="dot")))
                fig_pz.add_annotation(x=xv[0]*0.85, y=yv[0]*0.85, text=f"ζ={zeta_v}",
                                      showarrow=False, font=dict(size=9, color="#aaa"))

            fig_pz.update_layout(xaxis_title="Re(s)", yaxis_title="Im(s)",
                                 yaxis_scaleanchor="x", height=350, **PLOT)
            st.plotly_chart(fig_pz, use_container_width=True)

            # Q/R cost summary
            st.markdown("#### Cost Matrices")
            st.caption(f"Q diagonal = {np.round(np.diag(Q_mat), 3).tolist()}")
            st.caption(f"R = [[{R_val}]]")
            opt_cost = float(np.trace(P_lqr))
            st.metric("Optimal cost trace(P)", f"{opt_cost:.4f}",
                      help="Proportional to the closed-loop performance: lower = better")

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — LQR Response
# ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if not lqr_ok or t_lqr is None:
        st.error(lqr_err or "LQR simulation unavailable.")
    else:
        # Performance metrics on the primary output
        ref_s  = info["ref_state"]
        y_prim = x_lqr[:, ref_s]
        try:
            i10 = np.where(y_prim >= 0.1 * ref_val)[0][0] if ref_val > 0 else 0
            i90 = np.where(y_prim >= 0.9 * ref_val)[0][0] if ref_val > 0 else 0
            tr  = float(t_lqr[i90] - t_lqr[i10])
        except Exception:
            tr  = None
        try:
            outside = np.where(np.abs(y_prim - ref_val) > 0.02 * abs(ref_val))[0]
            ts = float(t_lqr[outside[-1]]) if len(outside) else 0.
        except Exception:
            ts = None

        os_pct = max(0., (y_prim.max() - ref_val) / (abs(ref_val) + 1e-15) * 100) if ref_val != 0 else 0.
        ss_err = float(ref_val - y_prim[-1])

        mc = st.columns(4)
        mc[0].metric("Overshoot (%)",    f"{os_pct:.1f}")
        mc[1].metric("Rise Time (s)",    f"{tr:.3f}" if tr is not None else "N/A")
        mc[2].metric("Settling (2%) s",  f"{ts:.3f}" if ts is not None else "N/A")
        mc[3].metric("SS Error",         f"{ss_err:.5f}")

        # Output + states plot
        n_plot_states = min(n, 4)
        fig_r = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Tracked State  (vs reference)", "All States", "Control Input  u(t)"),
        )

        # Row 1: reference + tracked state
        fig_r.add_trace(go.Scatter(x=t_lqr, y=np.full_like(t_lqr, ref_val),
                                   name="Reference", line=dict(color=C["ref"], dash="dash", width=1.5)),
                        row=1, col=1)
        fig_r.add_trace(go.Scatter(x=t_lqr, y=y_prim, name=snames[ref_s].split()[0],
                                   line=dict(color=C["lqr_out"], width=2.5)), row=1, col=1)

        # Row 2: all states
        palette = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
        for si in range(n_plot_states):
            fig_r.add_trace(go.Scatter(x=t_lqr, y=x_lqr[:, si],
                                       name=snames[si].split()[0],
                                       line=dict(color=palette[si % 4], width=1.5)),
                            row=2, col=1)

        # Row 3: control
        fig_r.add_trace(go.Scatter(x=t_lqr, y=u_lqr, name="u(t) — LQR",
                                   line=dict(color=C["u_lqr"], width=2)), row=3, col=1)

        for r in [1, 2, 3]:
            fig_r.update_xaxes(gridcolor=C["grid"], row=r, col=1)
            fig_r.update_yaxes(gridcolor=C["grid"], row=r, col=1)
        fig_r.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_r.update_layout(height=680, **PLOT)
        st.plotly_chart(fig_r, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Kalman Filter
# ─────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if not kalman_ok or t_lqg is None:
        st.error(kalman_err or "Kalman filter unavailable.")
    else:
        st.markdown(
            "The Kalman filter continuously estimates the full state vector from "
            "a single noisy measurement. Below: true state (green) vs estimated (orange) "
            "vs raw measurement (red)."
        )

        # Observer gain and error covariance
        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown("#### Observer Gain  L  (Kalman)")
            L_df = pd.DataFrame(np.round(L, 6), index=snames,
                                columns=info["output_names"])
            st.dataframe(L_df, use_container_width=True)
            st.caption("Observer: x̂̇ = Ax̂ + Bu + L·(y − Cx̂)")

        with kc2:
            st.markdown("#### Observer CL Eigenvalues  (A − LC)")
            if obs_cl_eigs is not None:
                obs_df = pd.DataFrame([{
                    "Eigenvalue": fmt_eig(e),
                    "Re":         f"{e.real:.5f}",
                    "|λ|":        f"{abs(e):.5f}",
                    "Stable":     "✅" if e.real < 0 else "❌",
                } for e in obs_cl_eigs])
                st.dataframe(obs_df, use_container_width=True, hide_index=True)

        # State estimation plots (all states, 2-column grid)
        n_rows = int(np.ceil(n / 2))
        fig_k  = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=[s.split("(")[0].strip() for s in snames],
            vertical_spacing=0.1, horizontal_spacing=0.08,
        )
        palette = ["#4CAF50", "#FF9800", "#2196F3", "#E91E63"]
        for si in range(n):
            r_idx = si // 2 + 1
            c_idx = si % 2 + 1
            fig_k.add_trace(go.Scatter(x=t_lqg, y=x_true[:, si], name=f"True {si}",
                                       line=dict(color=palette[0], width=2),
                                       showlegend=(si == 0)), row=r_idx, col=c_idx)
            fig_k.add_trace(go.Scatter(x=t_lqg, y=x_est[:, si], name=f"Estimated {si}",
                                       line=dict(color=palette[1], width=2, dash="dash"),
                                       showlegend=(si == 0)), row=r_idx, col=c_idx)

        # Measurement overlay on the observed state
        obs_state = np.argmax(C_obs[0])
        fig_k.add_trace(go.Scatter(x=t_lqg, y=y_meas.flatten(),
                                   name="Noisy measurement", mode="lines",
                                   line=dict(color=C["y_meas"], width=0.8, dash="dot"),
                                   opacity=0.7,
                                   showlegend=True),
                        row=obs_state // 2 + 1, col=obs_state % 2 + 1)

        for r in range(1, n_rows + 1):
            for c in range(1, 3):
                fig_k.update_xaxes(gridcolor=C["grid"], row=r, col=c)
                fig_k.update_yaxes(gridcolor=C["grid"], row=r, col=c)
        fig_k.update_layout(height=max(350, 200 * n_rows), **PLOT,
                            legend=dict(orientation="h", y=1.04))
        st.plotly_chart(fig_k, use_container_width=True)

        # Estimation RMSE
        rmse_vals = np.sqrt(np.mean((x_true - x_est)**2, axis=0))
        rmse_df = pd.DataFrame({
            "State":       snames,
            "Est. RMSE":   [f"{v:.5f}" for v in rmse_vals],
            "σw noise":    [f"{q_std:.4f}"] * n,
        })
        st.markdown("#### Estimation Accuracy")
        st.dataframe(rmse_df, use_container_width=True, hide_index=True)
        st.caption(f"Measurement noise σv = {r_std:.4f}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 5 — LQG Full Loop
# ─────────────────────────────────────────────────────────────────────────
with tabs[4]:
    if not lqr_ok or not kalman_ok or t_lqg is None:
        st.error("LQG unavailable — check LQR and Kalman tabs for errors.")
    else:
        st.markdown(
            "**LQG = LQR gain + Kalman estimator.**  "
            "The controller never sees the true state — it only uses the Kalman estimate. "
            "The Separation Principle guarantees that the LQR and Kalman designs can be "
            "combined independently without loss of optimality."
        )

        ref_s  = info["ref_state"]
        y_true_lqg = x_true[:, ref_s]
        y_est_lqg  = x_est[:,  ref_s]

        fig_lqg = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Tracked State: True vs Estimated vs Noisy Measurement",
                            "State Estimation (Tracked State)", "Control Signal  u(t)"),
        )

        # Row 1: true output, estimated, measurement, reference
        fig_lqg.add_trace(go.Scatter(x=t_lqg, y=np.full(len(t_lqg), ref_val),
                                     name="Reference", line=dict(color=C["ref"], dash="dash", width=1.5)),
                          row=1, col=1)
        fig_lqg.add_trace(go.Scatter(x=t_lqg, y=y_true_lqg, name="True state",
                                     line=dict(color=C["x_true"], width=2.5)), row=1, col=1)
        fig_lqg.add_trace(go.Scatter(x=t_lqg, y=y_meas.flatten(), name="Measurement (noisy)",
                                     line=dict(color=C["y_meas"], width=0.8, dash="dot"),
                                     opacity=0.6), row=1, col=1)

        # Row 2: true vs estimated (tracked state)
        fig_lqg.add_trace(go.Scatter(x=t_lqg, y=y_true_lqg, name="True",
                                     line=dict(color=C["x_true"], width=2)), row=2, col=1)
        fig_lqg.add_trace(go.Scatter(x=t_lqg, y=y_est_lqg, name="Estimated",
                                     line=dict(color=C["x_est"], width=2, dash="dash")), row=2, col=1)

        # Row 3: control
        fig_lqg.add_trace(go.Scatter(x=t_lqg, y=u_lqg_h, name="u(t) — LQG",
                                     line=dict(color=C["u_lqg"], width=2)), row=3, col=1)

        for r in [1, 2, 3]:
            fig_lqg.update_xaxes(gridcolor=C["grid"], row=r, col=1)
            fig_lqg.update_yaxes(gridcolor=C["grid"], row=r, col=1)
        fig_lqg.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_lqg.update_layout(height=680, **PLOT)
        st.plotly_chart(fig_lqg, use_container_width=True)

        # LQR vs LQG comparison (noiseless LQR vs noisy LQG on tracked state)
        if t_lqr is not None:
            st.markdown("#### LQR (ideal) vs LQG (with noise) — tracked state")
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=t_lqr, y=np.full(len(t_lqr), ref_val),
                                         name="Reference", line=dict(color=C["ref"], dash="dash", width=1)))
            fig_cmp.add_trace(go.Scatter(x=t_lqr, y=x_lqr[:, ref_s],
                                         name="LQR (no noise)", line=dict(color=C["lqr_out"], width=2.5)))
            fig_cmp.add_trace(go.Scatter(x=t_lqg, y=x_true[:, ref_s],
                                         name="LQG true state", line=dict(color=C["lqg_out"], width=2.5)))
            fig_cmp.update_layout(xaxis_title="Time (s)", height=320, **PLOT)
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Summary table
        lqr_ss = float(x_lqr[-1, ref_s]) if x_lqr is not None else None
        lqg_ss = float(x_true[-1, ref_s])
        lqg_rmse = float(np.sqrt(np.mean((x_true[:, ref_s] - ref_val)**2)))
        lqr_rmse = float(np.sqrt(np.mean((x_lqr[:, ref_s] - ref_val)**2))) if x_lqr is not None else None

        cmp_df = pd.DataFrame([
            {"Metric": "SS Value", "LQR":     f"{lqr_ss:.5f}" if lqr_ss else "—",
                                   "LQG":     f"{lqg_ss:.5f}"},
            {"Metric": "RMSE vs ref", "LQR":  f"{lqr_rmse:.5f}" if lqr_rmse else "—",
                                      "LQG":  f"{lqg_rmse:.5f}"},
            {"Metric": "σw", "LQR": "0 (ideal)", "LQG": f"{q_std}"},
            {"Metric": "σv", "LQR": "0 (ideal)", "LQG": f"{r_std}"},
        ])
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 6 — Pole Comparison (Q/R sensitivity)
# ─────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(
        "How do the **closed-loop poles move** as you change the Q/R trade-off?  "
        "Use the slider below to sweep the cost ratio and see the LQR pole locus."
    )
    if not lqr_ok:
        st.warning("LQR unavailable — adjust weights in the sidebar.")
    else:
        ratio_range = st.slider(
            "Q/R ratio sweep (log scale, applied uniformly to all Q weights)",
            min_value=-3, max_value=3, value=(-2, 2), step=1,
        )
        n_sweep = 60
        ratios  = np.logspace(ratio_range[0], ratio_range[1], n_sweep)
        base_Q  = np.diag(q_vals)
        branches = [[] for _ in range(n)]

        for ratio in ratios:
            try:
                K_s, _, eigs_s = lqr_design(A, B, base_Q * ratio, R_mat)
                for i, e in enumerate(eigs_s):
                    branches[i].append((e.real, e.imag))
            except Exception:
                for i in range(n):
                    branches[i].append((np.nan, np.nan))

        fig_qr = go.Figure()
        fig_qr.add_shape(type="line", x0=0, x1=0, y0=-100, y1=100,
                         line=dict(color="rgba(255,255,255,0.25)", dash="dot"))
        palette_b = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
        for bi, br in enumerate(branches):
            xs = [p[0] for p in br]
            ys = [p[1] for p in br]
            fig_qr.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                        name=f"Pole {bi+1}",
                                        line=dict(color=palette_b[bi % 4], width=2),
                                        marker=dict(size=4)))

        # Mark current Q/R operating point
        for e in cl_eigs_lqr:
            fig_qr.add_trace(go.Scatter(x=[e.real], y=[e.imag], mode="markers",
                                        name="Current", showlegend=False,
                                        marker=dict(symbol="star", size=18, color="white",
                                                    line=dict(color="#FF9800", width=2))))

        fig_qr.update_layout(
            xaxis_title="Re(s)", yaxis_title="Im(s)",
            yaxis_scaleanchor="x", height=500,
            title="LQR Pole Locus as Q/R ratio varies",
            **PLOT
        )
        st.plotly_chart(fig_qr, use_container_width=True)
        st.caption(
            "**Small Q/R** → poles near open-loop (cheap control, slow response).  "
            "**Large Q/R** → poles pushed far left (aggressive control, fast response, high effort)."
        )

# ─────────────────────────────────────────────────────────────────────────
# TAB 7 — Theory
# ─────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("## LQR / Kalman Filter / LQG — Theory Guide")

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("### Linear Quadratic Regulator (LQR)")
        st.markdown("""<div class="theory-box">

**System:**  $\\dot{x} = Ax + Bu$

**Cost function:**
$$J = \\int_0^\\infty \\left( x^\\top Q x + u^\\top R u \\right) dt$$

- **Q** penalises state deviation — make $Q_{ii}$ large to quickly drive state $i$ to zero
- **R** penalises control effort — large $R$ → gentle inputs, slow response

**Optimal control law:**  $u^* = -Kx$  where  $K = R^{-1} B^\\top P$

**Algebraic Riccati Equation (ARE):**
$$A^\\top P + PA - P B R^{-1} B^\\top P + Q = 0$$

**Closed-loop:** $\\dot{x} = (A - BK)x$ — always stable if (A,B) is stabilisable and Q ≥ 0, R > 0.

**Bryson's tuning rule:**  $Q_{ii} = 1/x_{i,\\max}^2$,  $R = 1/u_{\\max}^2$
</div>""", unsafe_allow_html=True)

        st.markdown("### Q vs R Tuning")
        st.markdown("""<div class="theory-box">

| Q/R ratio | Effect |
|-----------|--------|
| ↑ Q / fix R | State errors penalised more → faster response, more control effort |
| ↑ R / fix Q | Input penalised more → gentler control, slower settling |
| Q diagonal | Each $Q_{ii}$ weights state $i$ independently |

**Physical interpretation:**
If state $i$ is position (m) and limit is 0.5 m → $Q_{ii} = 1/0.5^2 = 4$.
If actuator limit is 10 N → $R = 1/100$.

The pole locus (Tab 6) shows how CL poles migrate as Q/R grows.
</div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("### Kalman Filter (Optimal Observer)")
        st.markdown("""<div class="theory-box">

**Stochastic system:**
$$\\dot{x} = Ax + Bu + w, \\quad w \\sim \\mathcal{N}(0, Q_n)$$
$$y = Cx + v, \\quad v \\sim \\mathcal{N}(0, R_n)$$

**Observer:**
$$\\dot{\\hat{x}} = A\\hat{x} + Bu + L(y - C\\hat{x})$$

**Kalman gain:**  $L = P_e C^\\top R_n^{-1}$  where $P_e$ solves:
$$A P_e + P_e A^\\top - P_e C^\\top R_n^{-1} C P_e + Q_n = 0$$

(This is the **dual** of the LQR ARE with $A \\to A^\\top$, $B \\to C^\\top$.)

**Tuning:**
- Large $Q_n$ → trust plant model less → faster observer, more noise tracking
- Large $R_n$ → trust sensor less → slower observer, smoother estimates
</div>""", unsafe_allow_html=True)

        st.markdown("### Separation Principle (LQG)")
        st.markdown("""<div class="theory-box">

**LQG controller:**
$$u = -K \\hat{x}$$

The magic: design LQR (choose K) and Kalman (choose L) **independently**,
then combine — stability and optimality are preserved.

**Why?** The combined system is:
$$\\begin{bmatrix} \\dot{x} \\\\ \\dot{e} \\end{bmatrix}
= \\begin{bmatrix} A - BK & BK \\\\ 0 & A - LC \\end{bmatrix}
\\begin{bmatrix} x \\\\ e \\end{bmatrix}$$

where $e = x - \\hat{x}$.  Eigenvalues = eig(A−BK) ∪ eig(A−LC) — **decoupled!**

**Practical notes:**
- More process noise → increase $Q_n$ to make Kalman respond faster
- Noisy sensors → increase $R_n$, accept slower estimation
- LQG robustness is weaker than LQR — check gain/phase margins separately
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**Coming next on this simulator:** MPC (Model Predictive Control) and LQG with loop-transfer recovery (LTR).
    """)
