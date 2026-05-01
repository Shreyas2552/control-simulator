# Codebase Reference — Control Systems Simulator

## What This Is

An interactive educational tool for control systems, built with Streamlit.
Covers classical control (PID, Bode, root locus, filters) and modern control
(LQR optimal control, Kalman filter, full LQG loop).

- **Live URL:** https://control-simulator-shreyas.streamlit.app/ *(deploy via Streamlit Share)*
- **GitHub repo:** https://github.com/Shreyas2552/control-simulator
- **Entry point:** `app.py`
- **Local launch:** `streamlit run app.py` or double-click `run_simulator.bat`

---

## File Structure

```
app.py                      Main page — PID simulator (6 plants, 6 tabs, theory guide)
run_simulator.bat           Windows launcher — uses python -m pip, keeps window open

pages/
  2_LQR_LQG.py             LQR + Kalman Filter + LQG full loop (7 tabs)

modules/                    Pure computation — no Streamlit code here
  __init__.py
  plants.py                 6 TF plant models + PLANT_MODELS catalogue dict
  pid_controller.py         PID transfer function + Tustin discretisation
  analysis.py               TF algebra, time responses, Bode, root locus,
                             stability margins, pole analysis, performance metrics
  filters.py                Analog/digital filter design (Butterworth, Chebyshev, Bessel)
  lqr_lqg.py                State-space plants, LQR/Kalman Riccati solvers, Euler sim

requirements.txt            streamlit, numpy, scipy, plotly, pandas
CLAUDE.md                   This file
```

### Streamlit multi-page convention
- `app.py` is the home page (shown first).
- Everything in `pages/` is auto-discovered and shown in the sidebar.
- File prefix number sets order: `2_LQR_LQG.py` → sidebar position 2.
- Each page calls `st.set_page_config()` as its **first** `st.*` call — this is required.
- All `from modules.X import Y` imports resolve relative to the repo root (where Streamlit runs from).

---

## Module APIs

### `modules/plants.py`
- `PLANT_MODELS` — dict keyed by display name. Each entry has `tf_display`, `physical_context`, `params` (default/min/max/step for sliders). Used directly by the sidebar loop in `app.py`.
- `get_plant_tf(plant_name, params) -> (num, den)` — coefficient lists, highest power first.
- **6 models:** First Order, Second Order, DC Motor (Position), Integrating Plant, Unstable Plant, Third Order.

### `modules/pid_controller.py`
- `get_pid_tf(Kp, Ki, Kd, N) -> (num, den)` — continuous PID: `C(s) = Kp + Ki/s + Kd·N·s/(s+N)`. Handles P/PI/PD as special cases.
- `discretize_tf(num, den, Ts, method='tustin') -> (num_d, den_d)` — wraps `scipy.signal.cont2discrete`.

### `modules/analysis.py`
- `build_ol_cl(plant_num, plant_den, ctrl_num, ctrl_den)` → `(ol_num, ol_den, cl_num, cl_den)`
- `step_response(num, den, t_end, n, discrete, Ts)` → `(t, y)`
- `ramp_response(num, den, t_end, n)` → `(t, y)`
- `control_signal_step(plant_num, plant_den, ctrl_num, ctrl_den, t_end, n)` → `(t, u)`
- `bode_data(num, den, n, discrete, Ts)` → `(omega, mag_dB, phase_deg)` — auto-ranges frequency.
- `stability_margins(ol_num, ol_den, discrete, Ts)` → `{gm_db, pm_deg, wgc, wpc}` — zero-crossing on unwrapped phase.
- `root_locus_data(ol_num, ol_den, n_gains)` → `(locus, ol_poles, ol_zeros)` — nearest-neighbour branch assignment.
- `cl_pole_analysis(cl_den)` → list of dicts with `{pole, kind, real, imag, wn, zeta, wd, stable, settling_time, overshoot_pct}`.
- `performance_metrics(t, y, ref)` → dict with overshoot, rise time (10→90%), settling (2%), SS error.

### `modules/filters.py`
- `design_analog(ftype, family, order, wc, wc2, ripple_db)` → `(b, a)`
- `design_digital(ftype, family, order, wn, wn2, ripple_db)` → `(b, a)` — `wn` is normalised [0,1].
- `filter_bode(b, a, analog)` → `(omega, mag_dB, phase_deg)`
- `filter_step(b, a, analog, t_end)` → `(t, y)`
- `FILTER_TYPES` = `["Low Pass", "High Pass", "Band Pass", "Band Stop"]`
- `FILTER_FAMILIES` = `["Butterworth", "Chebyshev Type I", "Bessel"]`

### `modules/lqr_lqg.py`
- `SS_PLANTS` — dict of 4 state-space plants with params, default Q/R weights, `ref_state` index.
- `get_ss(plant_name, params)` → `(A, B, C, D)` numpy matrices.
- `controllability_rank(A, B)` → int, `observability_rank(A, C)` → int.
- `lqr_design(A, B, Q, R)` → `(K, P, cl_eigs)` — solves continuous-time ARE.
- `kalman_design(A, C, Qn, Rn)` → `(L, Pe)` — observer Riccati, dual of LQR.
- `compute_nbar(A, B, C_row, K)` → float — DC pre-compensator for unity steady-state.
- `simulate_lqr(A, B, C_out, K, Nbar, ref, t_end, ref_state)` → `(t, x, y, u)` — Euler at 200 Hz.
- `simulate_lqg(A, B, C_out, C_obs, K, L, Nbar, ref, t_end, q_std, r_std, ref_state)` → `(t, x_true, x_est, y_meas, u)`.
- **4 plants:** Mass-Spring-Damper, DC Motor (Full Electrical + Mechanical), Inverted Pendulum on Cart, Double Integrator.

---

## Known Gotchas

### 1. NumPy 2.x — use `.item()` not `float()` on arrays
Running **Python 3.14 + NumPy 2.4**. In NumPy 2.0+, `float(arr)` raises `TypeError` when `arr` is 1-dimensional, even with one element.

```python
# WRONG — breaks with NumPy 2.x
u_k = float(-K @ x)        # K is (1,n) → result is shape (1,) → TypeError

# CORRECT
u_k = (-K @ x).item()      # .item() works for any single-element array
```

Fixed in `lqr_lqg.py` at lines 231, 237 (`simulate_lqr`) and 272 (`simulate_lqg`). Always use `.item()` or `.flat[0]` when extracting a scalar from a matrix-vector product.

### 2. `pip` not in PATH — use `python -m pip`
Only `python` is in PATH on this machine, not `pip.exe`. The `run_simulator.bat` already uses `python -m pip install`. Don't use bare `pip` in any scripts.

### 3. `set_page_config` must be first
In each page file, `st.set_page_config()` must come before any other `st.*` call. Don't add sidebar or write calls above it.

---

## Deploying to Streamlit Share

1. Go to https://share.streamlit.io
2. Click **New app**
3. Select repo: `Shreyas2552/control-simulator`
4. Branch: `main`
5. Main file: `app.py`
6. Click **Deploy**

Streamlit Share auto-redeploys on every push to `main`.

---

## Adding a New Page

1. Create `pages/N_Page_Name.py` (number sets sidebar order).
2. First line must be `st.set_page_config(...)`.
3. Import from `modules/` as needed.
4. Add new pip packages to `requirements.txt`.
5. Commit and push → auto-redeploys on Streamlit Share.

## Adding a New Transfer-Function Plant (PID page)

1. Add entry to `PLANT_MODELS` dict in `modules/plants.py` — follow existing pattern.
2. Add `if plant_name == "Your Plant":` branch in `get_plant_tf()`.
3. Nothing else changes — sidebar and tabs are fully data-driven.

## Adding a New State-Space Plant (LQR/LQG page)

1. Add entry to `SS_PLANTS` dict in `modules/lqr_lqg.py`.
2. Add `if plant_name == "Your Plant":` branch in `get_ss()`.
3. Provide `default_Q` (list, one value per state), `default_R` (float), `ref_state` (int index).
