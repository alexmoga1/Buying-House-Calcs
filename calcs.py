# app.py
# Streamlit version of your interactive vacation rental mortgage coverage model.
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------
# Defaults (from your sheet)
# ----------------------------
RENTABLE_NIGHTS_PER_YEAR = 356
COGS_ANNUAL_Y1_DEFAULT = 9_300.00

DEFAULTS = dict(
    home_price=1_200_000.00,
    down_payment_pct=0.20,
    mortgage_rate_apr=0.06,
    term_years=30,
    mgmt_fee=0.35,
    nightly_rate=500.0,
    vacancy_rate=0.50,     # % vacant
    rent_growth=0.03,
    cogs_inflation=0.02,
    cogs_annual_y1=COGS_ANNUAL_Y1_DEFAULT,
)

# ----------------------------
# Math
# ----------------------------
def annual_mortgage_payment(home_price: float, down_payment_pct: float, apr: float, term_years: int) -> float:
    """Annual P+I payment for an amortizing mortgage. Returns positive magnitude."""
    principal = home_price * (1.0 - down_payment_pct)
    n = int(round(term_years * 12))
    if n <= 0:
        return float("nan")
    r = apr / 12.0
    if abs(r) < 1e-12:
        monthly = principal / n
    else:
        monthly = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return monthly * 12.0


def build_df(
    home_price: float,
    down_payment_pct: float,
    apr: float,
    term_years: int,
    nightly_rate: float,
    vacancy_rate: float,
    mgmt_fee: float,
    rent_growth: float,
    cogs_inflation: float,
    cogs_annual_y1: float,
) -> pd.DataFrame:
    years = int(max(1, round(term_years)))
    y = np.arange(1, years + 1)

    mortgage_annual = annual_mortgage_payment(home_price, down_payment_pct, apr, term_years)

    # Year 1 rent BEFORE management fee (matches your sheet logic)
    total_rent_y1 = nightly_rate * RENTABLE_NIGHTS_PER_YEAR * (1 - vacancy_rate)

    total_rent = total_rent_y1 * (1 + rent_growth) ** (y - 1)
    mgmt_fee_cost = total_rent * mgmt_fee
    net_after_fee = total_rent - mgmt_fee_cost

    cogs = cogs_annual_y1 * (1 + cogs_inflation) ** (y - 1)

    cash_available = net_after_fee - cogs
    out_of_pocket = mortgage_annual - cash_available
    cumulative_out_of_pocket = np.cumsum(out_of_pocket)

    df = pd.DataFrame(
        {
            "Year": y,
            "Total Rent": total_rent,
            "Mgmt Fee Cost": mgmt_fee_cost,
            "COGS": cogs,
            "Cash Available": cash_available,
            "Mortgage (Annual P+I)": mortgage_annual,
            "Out of Pocket": out_of_pocket,
            "Cumulative Out of Pocket": cumulative_out_of_pocket,
        }
    )
    return df


def first_breakeven_year(df: pd.DataFrame):
    hit = df.loc[df["Out of Pocket"] <= 0, "Year"]
    return int(hit.iloc[0]) if len(hit) else None


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Vacation Rental Mortgage Coverage", layout="wide")
st.title("Vacation Rental Mortgage Coverage Simulator")

with st.sidebar:
    st.subheader("Typable inputs")
    home_price = st.number_input("House Price ($)", min_value=0.0, value=float(DEFAULTS["home_price"]), step=10_000.0, format="%.2f")
    mortgage_rate_apr = st.number_input("Mortgage Rate (APR %)", min_value=0.0, max_value=30.0, value=float(DEFAULTS["mortgage_rate_apr"] * 100), step=0.25, format="%.3f") / 100.0
    term_years = st.number_input("Term (years)", min_value=1, max_value=50, value=int(DEFAULTS["term_years"]), step=1)
    mgmt_fee = st.number_input("Mgmt Fee (%)", min_value=0.0, max_value=80.0, value=float(DEFAULTS["mgmt_fee"] * 100), step=0.5, format="%.2f") / 100.0

    st.divider()
    st.subheader("Compact knobs (sliders)")
    nightly_rate = st.slider("Nightly Rate ($)", min_value=50, max_value=1500, value=int(DEFAULTS["nightly_rate"]), step=10)
    vacancy_rate = st.slider("Vacancy (%)", min_value=0, max_value=95, value=int(DEFAULTS["vacancy_rate"] * 100), step=1) / 100.0
    rent_growth = st.slider("Rent Growth (%/yr)", min_value=-5.0, max_value=15.0, value=float(DEFAULTS["rent_growth"] * 100), step=0.5) / 100.0
    cogs_inflation = st.slider("COGS Inflation (%/yr)", min_value=0.0, max_value=15.0, value=float(DEFAULTS["cogs_inflation"] * 100), step=0.5) / 100.0

    # Optional: make this typable too (helps reality)
    cogs_annual_y1 = st.number_input("COGS Year 1 ($/yr)", min_value=0.0, value=float(DEFAULTS["cogs_annual_y1"]), step=500.0, format="%.2f")

    down_payment_pct = st.slider("Down Payment (%)", min_value=0, max_value=80, value=int(DEFAULTS["down_payment_pct"] * 100), step=1) / 100.0


df = build_df(
    home_price=home_price,
    down_payment_pct=down_payment_pct,
    apr=mortgage_rate_apr,
    term_years=term_years,
    nightly_rate=float(nightly_rate),
    vacancy_rate=vacancy_rate,
    mgmt_fee=mgmt_fee,
    rent_growth=rent_growth,
    cogs_inflation=cogs_inflation,
    cogs_annual_y1=cogs_annual_y1,
)

breakeven = first_breakeven_year(df)

# Key metrics (Year 1)
y1 = df.iloc[0]
loan_amount = home_price * (1 - down_payment_pct)
mortgage_annual = float(y1["Mortgage (Annual P+I)"])
total_rent_y1 = float(y1["Total Rent"])
cash_available_y1 = float(y1["Cash Available"])
oop_y1 = float(y1["Out of Pocket"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Loan Amount", f"${loan_amount:,.0f}")
c2.metric("Mortgage (Annual P+I)", f"${mortgage_annual:,.0f}")
c3.metric("Total Rent (Year 1)", f"${total_rent_y1:,.0f}")
c4.metric("Cash Available (Year 1)", f"${cash_available_y1:,.0f}")
c5.metric("Break-even Year", "Not within term" if breakeven is None else f"Year {breakeven}")

tabs = st.tabs(["Summary Charts", "Year-by-Year View"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Out-of-Pocket Per Year")
        chart1 = (
            alt.Chart(df)
            .mark_line()
            .encode(x="Year:Q", y=alt.Y("Out of Pocket:Q", title="$/year"), tooltip=["Year", "Out of Pocket"])
        )
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
        st.altair_chart((chart1 + zero_line).interactive(), use_container_width=True)

    with right:
        st.subheader("Cumulative Out-of-Pocket")
        chart2 = (
            alt.Chart(df)
            .mark_line()
            .encode(x="Year:Q", y=alt.Y("Cumulative Out of Pocket:Q", title="Cumulative $"), tooltip=["Year", "Cumulative Out of Pocket"])
        )
        st.altair_chart((chart2 + zero_line).interactive(), use_container_width=True)

with tabs[1]:
    st.subheader("Breakdown & Table")

    top1, top2 = st.columns([1, 1])

    # Rent breakdown stacked (Rent positive, costs negative)
    df_stack = df[["Year", "Total Rent", "Mgmt Fee Cost", "COGS"]].copy()
    df_stack = df_stack.melt("Year", var_name="Component", value_name="Amount")
    # Make costs negative for intuitive stacking
    df_stack.loc[df_stack["Component"].isin(["Mgmt Fee Cost", "COGS"]), "Amount"] *= -1

    with top1:
        st.caption("Stacked: Total Rent, -Mgmt Fee, -COGS")
        chart_stack = (
            alt.Chart(df_stack)
            .mark_area()
            .encode(
                x="Year:Q",
                y=alt.Y("Amount:Q", title="$/year"),
                color="Component:N",
                tooltip=["Year", "Component", alt.Tooltip("Amount:Q", format=",.0f")],
            )
        )
        st.altair_chart((chart_stack + zero_line).interactive(), use_container_width=True)

    with top2:
        st.caption("Coverage: Cash Available vs Mortgage")
        df_cov = df[["Year", "Cash Available", "Mortgage (Annual P+I)"]].melt("Year", var_name="Series", value_name="Amount")
        chart_cov = (
            alt.Chart(df_cov)
            .mark_line()
            .encode(
                x="Year:Q",
                y=alt.Y("Amount:Q", title="$/year"),
                color="Series:N",
                tooltip=["Year", "Series", alt.Tooltip("Amount:Q", format=",.0f")],
            )
        )
        st.altair_chart((chart_cov + zero_line).interactive(), use_container_width=True)

    st.divider()
    st.dataframe(
        df.style.format(
            {
                "Total Rent": "${:,.0f}",
                "Mgmt Fee Cost": "${:,.0f}",
                "COGS": "${:,.0f}",
                "Cash Available": "${:,.0f}",
                "Mortgage (Annual P+I)": "${:,.0f}",
                "Out of Pocket": "${:,.0f}",
                "Cumulative Out of Pocket": "${:,.0f}",
            }
        ),
        use_container_width=True,
        height=420,
    )

st.caption(
    "Interpretation: Out-of-Pocket > 0 means you cover the gap; < 0 means surplus. "
    "Break-even year is the first year Out-of-Pocket goes <= 0."
)
