"""
Interactive local UI for BSREOptions using Streamlit.

Run with:
    streamlit run ui_app.py
"""

import numpy as np
import streamlit as st
from bsre_options import BSREOptions
from visuals import (
    plot_paths,
    plot_final_distribution,
    plot_price_surface_3d,
    plot_sensitivity_bars,
)

st.set_page_config(
    page_title="BSREOptions Explorer",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* main app background */
    .stApp {
        background-color: #121212;
    }

    /* top bar / header */
    header[data-testid="stHeader"] {
        background-color: #121212;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    div[data-baseweb="slider"] {
        padding-top: 0.1rem;
        padding-bottom: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def simulate_and_price(model: BSREOptions) -> dict:
    V, r = model.simulate_paths()

    dt = model.T / model.M
    integral_r = np.sum(r[:-1], axis=0) * dt
    discount_factor = np.exp(-integral_r)

    payoff = np.maximum(V[-1] - model.K, 0.0)
    price_paths = discount_factor * payoff
    price = float(np.mean(price_paths))

    result: dict[str, float | np.ndarray] = {
        "V": V,
        "r": r,
        "price": price,
    }

    if model.I > 1:
        sample_std = float(np.std(price_paths, ddof=1))
        stderr = sample_std / np.sqrt(model.I)

        z_95 = 1.96
        ci_low = price - z_95 * stderr
        ci_high = price + z_95 * stderr

        result["stderr"] = stderr
        result["ci_low"] = ci_low
        result["ci_high"] = ci_high
    else:
        result["stderr"] = float("nan")
        result["ci_low"] = float("nan")
        result["ci_high"] = float("nan")

    return result


def compute_sensitivity_data(params: dict) -> tuple[list[str], list[float], list[float]]:
    """
    Compute one-way sensitivities of the option price to key parameters.

    For each parameter, bump it down/up and reprice with a lighter Monte Carlo config.
    Returns:
        names        : list of parameter names
        prices_low   : prices with parameter bumped down
        prices_high  : prices with parameter bumped up
    """
    sens_I = max(2_000, params["I"] // 5)
    sens_M = max(20, params["M"] // 2)

    specs: list[tuple[str, str, float]] = [
        ("V0", "relative", 0.10),
        ("K", "relative", 0.10),
        ("T", "relative", 0.10),
        ("r0", "absolute", 0.01),
        ("kappa_r", "relative", 0.20),
        ("theta_r", "absolute", 0.01),
        ("sigma_r", "absolute", 0.005),
        ("v0", "relative", 0.25),
        ("kappa_v", "relative", 0.20),
        ("theta_v", "relative", 0.25),
        ("sigma_v", "relative", 0.20),
        ("rho", "absolute", 0.10),
        ("q", "absolute", 0.005),
        ("carry_cost", "absolute", 0.005),
    ]

    names: list[str] = []
    prices_low: list[float] = []
    prices_high: list[float] = []

    base_seed = int(params["seed"])

    for name, mode, bump in specs:
        val = params[name]

        if mode == "relative":
            delta = max(abs(val) * bump, 1e-6)
        else:
            delta = bump

        if name in {"v0", "theta_v", "sigma_v"}:
            low_val = max(0.0, val - delta)
            high_val = max(0.0, val + delta)
        elif name in {"T"}:
            low_val = max(0.01, val - delta)
            high_val = val + delta
        elif name in {"r0", "theta_r", "q", "carry_cost"}:
            low_val = max(-0.5, val - delta)
            high_val = min(0.5, val + delta)
        elif name == "sigma_r":
            low_val = max(0.0, val - delta)
            high_val = max(0.0, val + delta)
        elif name == "rho":
            low_val = max(-1.0, val - delta)
            high_val = min(1.0, val + delta)
        else:
            low_val = val - delta
            high_val = val + delta

        params_low = dict(params)
        params_low[name] = low_val
        params_low["M"] = sens_M
        params_low["I"] = sens_I
        params_low["seed"] = base_seed

        model_low = BSREOptions(
            V0=params_low["V0"],
            K=params_low["K"],
            T=params_low["T"],
            r0=params_low["r0"],
            kappa_r=params_low["kappa_r"],
            theta_r=params_low["theta_r"],
            sigma_r=params_low["sigma_r"],
            v0=params_low["v0"],
            kappa_v=params_low["kappa_v"],
            theta_v=params_low["theta_v"],
            sigma_v=params_low["sigma_v"],
            rho=params_low["rho"],
            q=params_low["q"],
            carry_cost=params_low["carry_cost"],
            M=params_low["M"],
            I=params_low["I"],
            seed=params_low["seed"],
        )
        price_low = float(simulate_and_price(model_low)["price"])

        params_high = dict(params)
        params_high[name] = high_val
        params_high["M"] = sens_M
        params_high["I"] = sens_I
        params_high["seed"] = base_seed

        model_high = BSREOptions(
            V0=params_high["V0"],
            K=params_high["K"],
            T=params_high["T"],
            r0=params_high["r0"],
            kappa_r=params_high["kappa_r"],
            theta_r=params_high["theta_r"],
            sigma_r=params_high["sigma_r"],
            v0=params_high["v0"],
            kappa_v=params_high["kappa_v"],
            theta_v=params_high["theta_v"],
            sigma_v=params_high["sigma_v"],
            rho=params_high["rho"],
            q=params_high["q"],
            carry_cost=params_high["carry_cost"],
            M=params_high["M"],
            I=params_high["I"],
            seed=params_high["seed"],
        )
        price_high = float(simulate_and_price(model_high)["price"])

        names.append(name)
        prices_low.append(price_low)
        prices_high.append(price_high)

    return names, prices_low, prices_high


def controls() -> dict:
    s = st.sidebar

    # --- Property and option ---
    s.markdown("**Property and option**")
    V0 = s.slider("V0", min_value=50_000.0, max_value=10_000_000.0, value=200_000.0, step=10_000.0, format="$%0.0f", help="Initial property value at t = 0 (dollars).")
    K = s.slider("K (strike)", min_value=50_000.0, max_value=10_000_000.0, value=220_000.0, step=10_000.0, format="$%0.0f", help="Strike price / exercise price of the option (dollars).")
    T = s.slider("T (years)", min_value=0.05, max_value=30.0, value=10.0, step=0.05, format="%.2f", help="Time to maturity in years.")

    # --- Yield and carry ---
    s.markdown("**Yield and carry**")
    q_pct = s.slider("q", min_value=0.0, max_value=50.0, value=2.0, step=0.1, format="%.2f%%", help="Lease/dividend yield as percent of property value per year.")
    carry_pct = s.slider("carry_cost", min_value=-50.0, max_value=50.0, value=0.0, step=0.1, format="%.1f%%", help="Additional carry in the drift (percent per year). Negative values are extra cost.")

    # --- Rates ---
    s.markdown("**Short-rate process**")
    r0_pct = s.slider("r0", min_value=0.0, max_value=20.0, value=3.0, step=0.1, format="%.2f%%", help="Initial short rate (annual, percent).")
    kappa_r = s.slider("kappa_r", min_value=0.0, max_value=5.0, value=0.3, step=0.1, format="%.2f", help="Mean reversion speed of the short rate r(t) (per year).")
    theta_r_pct = s.slider("theta_r", min_value=0.0, max_value=20.0, value=3.0, step=0.1, format="%.2f%%", help="Long-run mean short rate (annual, percent).")
    sigma_r_pct = s.slider("sigma_r", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f%%", help="Volatility of the short rate (annual, percent).")

    # --- Variance process ---
    s.markdown("**Variance process**")
    v0 = s.slider("v0", min_value=0.0, max_value=1.0, value=0.02, step=0.005, format="%.3f", help="Initial variance of the property value process (annual).")
    kappa_v = s.slider("kappa_v", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.2f", help="Mean reversion speed of the variance (per year).")
    theta_v = s.slider("theta_v", min_value=0.0, max_value=1.0, value=0.02, step=0.005, format="%.3f", help="Long-run variance level (annual).")
    sigma_v = s.slider("sigma_v", min_value=0.0, max_value=2.0, value=0.1, step=0.01, format="%.3f", help="Volatility of the variance process.")
    rho = s.slider("rho", min_value=-1.0, max_value=1.0, value=-0.5, step=0.05, format="%.2f", help="Correlation between property-value shocks and rate shocks.")

    # --- Monte Carlo controls ---
    s.markdown("**Monte Carlo**")
    M = s.slider("M (time steps)", min_value=10, max_value=1_000, value=100, step=10, format="%d", help="Number of time steps per simulated path.")
    I = s.slider("I (paths)", min_value=500, max_value=50_000, value=10_000, step=500, format="%d", help="Number of Monte Carlo paths.")
    seed = s.slider("Random seed", min_value=0, max_value=1_000_000, value=123, step=1, format="%d", help="Fixed seed for reproducible simulations.")
    max_paths_plot = s.slider("Paths in plot", min_value=10, max_value=500, value=100, step=10, format="%d", help="Number of simulated paths drawn in the path chart.")

    q = q_pct / 100.0
    carry_cost = carry_pct / 100.0
    r0 = r0_pct / 100.0
    theta_r = theta_r_pct / 100.0
    sigma_r = sigma_r_pct / 100.0

    params = {
        "V0": V0,
        "K": K,
        "T": T,
        "r0": r0,
        "kappa_r": kappa_r,
        "theta_r": theta_r,
        "sigma_r": sigma_r,
        "v0": v0,
        "kappa_v": kappa_v,
        "theta_v": theta_v,
        "sigma_v": sigma_v,
        "rho": rho,
        "q": q,
        "carry_cost": carry_cost,
        "M": M,
        "I": I,
        "seed": int(seed),
        "max_paths_plot": max_paths_plot,
    }
    return params



def main() -> None:
    st.title("Interactive Explorer")

    st.markdown(
        """
        Real estate option priced with a Black–Scholes-style Monte Carlo engine,
        with stochastic short rates and stochastic variance.
        """
    )
    st.markdown("")
    st.markdown("")

    params = controls()

    model = BSREOptions(
        V0=params["V0"],
        K=params["K"],
        T=params["T"],
        r0=params["r0"],
        kappa_r=params["kappa_r"],
        theta_r=params["theta_r"],
        sigma_r=params["sigma_r"],
        v0=params["v0"],
        kappa_v=params["kappa_v"],
        theta_v=params["theta_v"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        q=params["q"],
        carry_cost=params["carry_cost"],
        M=params["M"],
        I=params["I"],
        seed=params["seed"],
    )

    with st.spinner("Running Monte Carlo simulation..."):
        result = simulate_and_price(model)

    price = result["price"]
    stderr = result["stderr"]
    ci_low = result["ci_low"]
    ci_high = result["ci_high"]
    V = result["V"]
    r = result["r"]

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated option value", f"${price:,.2f}")
    col2.metric("Std. error", f"{stderr:,.4f}")
    col3.metric("95% CI", f"[${ci_low:,.2f}, ${ci_high:,.2f}]")

    st.markdown("")
    st.markdown("")
    st.markdown("")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_paths = plot_paths(
            V,
            r=r,
            max_paths=int(params["max_paths_plot"]),
            show=False,
            save_path=None,
        )
        st.pyplot(fig_paths, clear_figure=True)

    with col_right:
        fig_hist = plot_final_distribution(
            V,
            bins=100,
            show=False,
            save_path=None,
        )
        st.pyplot(fig_hist, clear_figure=True)

    with st.spinner("Computing price surface and parameter sensitivities..."):
        surface_I = max(1_000, params["I"] // 10)
        surface_M = max(20, params["M"] // 2)

        n_K = 10
        n_T = 10

        K_center = params["K"]
        K_low = 0.7 * K_center
        K_high = 1.3 * K_center

        T_center = params["T"]
        T_low = max(0.05, 0.5 * T_center)
        T_high = max(T_low + 1e-6, 1.5 * T_center)

        K_vals = np.linspace(K_low, K_high, n_K)
        T_vals = np.linspace(T_low, T_high, n_T)
        KK, TT = np.meshgrid(K_vals, T_vals, indexing="xy")
        P = np.zeros_like(KK)

        for i in range(n_T):
            for j in range(n_K):
                m = BSREOptions(
                    V0=params["V0"],
                    K=float(KK[i, j]),
                    T=float(TT[i, j]),
                    r0=params["r0"],
                    kappa_r=params["kappa_r"],
                    theta_r=params["theta_r"],
                    sigma_r=params["sigma_r"],
                    v0=params["v0"],
                    kappa_v=params["kappa_v"],
                    theta_v=params["theta_v"],
                    sigma_v=params["sigma_v"],
                    rho=params["rho"],
                    q=params["q"],
                    carry_cost=params["carry_cost"],
                    M=surface_M,
                    I=surface_I,
                    seed=None,
                )
                V_s, r_s = m.simulate_paths()
                dt_s = m.T / m.M
                disc_s = np.exp(-np.sum(r_s[:-1, :], axis=0) * dt_s)
                payoff_s = np.maximum(V_s[-1] - m.K, 0.0)
                P[i, j] = np.mean(disc_s * payoff_s)

        sens_names, sens_low, sens_high = compute_sensitivity_data(params)

    col_surface, col_sens = st.columns(2)

    with col_surface:
        fig_surface = plot_price_surface_3d(
            KK,
            TT,
            P,
            show=False,
            save_path=None,
        )
        st.pyplot(fig_surface, clear_figure=True)

    with col_sens:
        fig_sens = plot_sensitivity_bars(
            sens_names,
            sens_low,
            sens_high,
            base_price=price,
            show=False,
            save_path=None,
        )
        st.pyplot(fig_sens, clear_figure=True)

    st.title("")

    V_T = V[-1, :]
    r_T = r[-1, :]

    quantiles = [5, 25, 50, 75, 95]
    V_q = np.percentile(V_T, quantiles)
    r_q = np.percentile(r_T, quantiles)

    dt = model.T / model.M
    integral_r = np.sum(r[:-1, :], axis=0) * dt
    discount_factor = np.exp(-integral_r)

    payoff_T = np.maximum(V_T - model.K, 0.0)
    discounted_payoff = discount_factor * payoff_T

    prob_itm = float(np.mean(V_T > model.K))
    mean_payoff = float(np.mean(payoff_T))
    mean_disc_payoff = float(price)
    disc_q5, disc_q95 = np.percentile(discounted_payoff, [5, 95])

    stats_cols = st.columns(3)

    with stats_cols[0]:
        st.markdown("**Terminal property value V_T (percentiles)**")
        st.table(
            {
                "Percentile": [f"{q}%" for q in quantiles],
                "V_T": [f"${x:,.0f}" for x in V_q],
            }
        )

    with stats_cols[1]:
        st.markdown("**Terminal short rate r_T (percentiles)**")
        st.table(
            {
                "Percentile": [f"{q}%" for q in quantiles],
                "r_T": [f"{100 * x:.2f}%" for x in r_q],
            }
        )

    with stats_cols[2]:
        st.markdown("**Option payoff and price metrics**")

        price_pct_strike = mean_disc_payoff / model.K

        st.table(
            {
                "Metric": [
                    "Prob. finishes in-the-money",
                    "Mean payoff (undiscounted)",
                    "Mean discounted payoff (price)",
                    "Option price as % of strike",
                    "5% to 95% disc payoff range",
                ],
                "Value": [
                    f"{prob_itm:.1%}",
                    f"${mean_payoff:,.2f}",
                    f"${mean_disc_payoff:,.2f}",
                    f"{100 * price_pct_strike:.2f}%",
                    f"[${disc_q5:,.1f}, ${disc_q95:,.0f}]",
                ],
            }
        )




if __name__ == "__main__":
    main()
