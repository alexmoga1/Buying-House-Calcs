# app.py
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------
# Defaults (your sheet)
# ----------------------------
RENTABLE_NIGHTS_PER_YEAR = 356

DEFAULTS = dict(
    home_price=1_200_000.00,
    down_payment_pct=0.20,
    mortgage_rate_apr=0.06,
    term_years=30,
    mgmt_fee=0.35,

    nightly_rate=500.0,
    vacancy_rate=0.50,     # % vacant
    rent_growth=0.03,      # annual
    cogs_inflation=0.02,   # annual
    cogs_annual_y1=9_300.00,

    reinvest_pct=1.00,     # 100% -> all surplus to extra principal
)

# ----------------------------
# Core mortgage helpers
# ----------------------------
def monthly_payment(principal: float, apr: float, term_years: int) -> float:
    """Standard amortizing mortgage monthly payment (P+I)."""
    n = int(term_years * 12)
    if n <= 0:
        return float("nan")
    r = apr / 12.0
    if abs(r) < 1e-12:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def simulate(
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
    reinvest_pct: float,
) -> pd.DataFrame:
    """
    Month-by-month simulation:
    - rent & cogs are constant within a year, then step up each year by growth/inflation
    - mortgage is fixed-rate minimum payment, plus extra principal from reinvestment
    """
    years = int(max(1, round(term_years)))
    months = years * 12

    loan0 = home_price * (1 - down_payment_pct)
    bal = loan0

    pmt_min = monthly_payment(loan0, apr, years)

    out_of_pocket_year = np.zeros(years)
    takehome_profit_year = np.zeros(years)
    extra_principal_year = np.zeros(years)
    interest_paid_year = np.zeros(years)
    principal_paid_year = np.zeros(years)
    end_balance_year = np.zeros(years)
    cash_available_year = np.zeros(years)
    total_rent_year = np.zeros(years)

    # Precompute year-level rent/cogs based on your formula
    # Year1 Total Rent (before mgmt fee): nightly * rentable_nights * (1 - vacancy)
    total_rent_y1 = nightly_rate * RENTABLE_NIGHTS_PER_YEAR * (1 - vacancy_rate)

    for year_idx in range(years):
        # Year-level (growing each year)
        yr = year_idx + 1
        total_rent = total_rent_y1 * (1 + rent_growth) ** year_idx
        cogs = cogs_annual_y1 * (1 + cogs_inflation) ** year_idx

        mgmt_cost = total_rent * mgmt_fee
        net_after_fee = total_rent - mgmt_cost
        cash_available = net_after_fee - cogs

        total_rent_year[year_idx] = total_rent
        cash_available_year[year_idx] = cash_available

        cash_m = cash_available / 12.0

        # Simulate 12 months for this year (or until paid off)
        for _m in range(12):
            if bal <= 1e-6:
                break

            r_m = apr / 12.0
            interest = bal * r_m
            principal_min = max(0.0, pmt_min - interest)

            # If final payment would overshoot, cap it
            if principal_min > bal:
                principal_min = bal
                actual_pmt = interest + principal_min
            else:
                actual_pmt = pmt_min

            # Monthly surplus after paying minimum (if any)
            monthly_surplus = max(0.0, cash_m - actual_pmt)
            monthly_shortfall = max(0.0, actual_pmt - cash_m)

            # Reinvestment: use a fraction of surplus as extra principal
            extra_principal = monthly_surplus * reinvest_pct
            # Takehome profit: remainder of surplus
            takehome_profit = monthly_surplus * (1.0 - reinvest_pct)

            # Extra principal cannot exceed remaining balance after min principal
            extra_principal = min(extra_principal, max(0.0, bal - principal_min))

            # Update totals
            out_of_pocket_year[year_idx] += monthly_shortfall
            takehome_profit_year[year_idx] += takehome_profit
            extra_principal_year[year_idx] += extra_principal
            interest_paid_year[year_idx] += interest
            principal_paid_year[year_idx] += principal_min + extra_principal

            # Reduce balance
            bal -= (principal_min + extra_principal)

        end_balance_year[year_idx] = max(0.0, bal)

    # Mortgage paid each year (minimum + extra principal; not including out-of-pocket coverage logic)
    # For reporting: "Mortgage P+I paid from property cash + out-of-pocket" is less intuitive here,
    # so we expose interest/principal and balance directly.
    df = pd.DataFrame({
        "Year": np.arange(1, years + 1),
        "Total Rent": total_rent_year,
        "Cash Available": cash_available_year,
        "Out of Pocket": out_of_pocket_year,
        "Takehome Profit": takehome_profit_year,
        "Extra Principal Paid": extra_principal_year,
        "Interest Paid": interest_paid_year,
        "Principal Paid": principal_paid_year,
        "Ending Balance": end_balance_year,
    })

    df["Cumulative Out of Pocket"] = df["Out of Pocket"].cumsum()

    # Payoff year (first year ending balance hits 0)
    payoff = df.index[df["Ending Balance"] <= 1e-6]
    df.attrs["payoff_year"] = int(df.loc[payoff[0], "Year"]) if len(payoff) else None
    df.attrs["loan_amount"] = loan0
    df.attrs["monthly_payment_min"] = pmt_min
    return df

def first_breakeven_year(df: pd.DataFrame):
    hit = df.loc[df["Out of Pocket"] <= 1e-6, "Year"]
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
    st.subheader("Compact controls (sliders)")
    nightly_rate = st.slider("Nightly Rate ($)", min_value=50, max_value=1500, value=int(DEFAULTS["nightly_rate"]), step=10)
    vacancy_rate = st.slider("Vacancy (%)", min_value=0, max_value=95, value=int(DEFAULTS["vacancy_rate"] * 100), step=1) / 100.0
    rent_growth = st.slider("Rent Growth (%/yr)", min_value=-5.0, max_value=15.0, value=float(DEFAULTS["rent_growth"] * 100), step=0.5) / 100.0
    cogs_inflation = st.slider("COGS Inflation (%/yr)", min_value=0.0, max_value=15.0, value=float(DEFAULTS["cogs_inflation"] * 100), step=0.5) / 100.0
    cogs_annual_y1 = st.number_input("COGS Year 1 ($/yr)", min_value=0.0, value=float(DEFAULTS["cogs_annual_y1"]), step=500.0, format="%.2f")
    down_payment_pct = st.slider("Down Payment (%)", min_value=0, max_value=80, value=int(DEFAULTS["down_payment_pct"] * 100), step=1) / 100.0

    reinvest_pct = st.slider(
        "Reinvest Surplus to Principal (%)",
        min_value=0, max_value=100,
        value=int(DEFAULTS["reinvest_pct"] * 100),
        step=5
    ) / 100.0

df = simulate(
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
    reinvest_pct=reinvest_pct,
)

breakeven = first_breakeven_year(df)
payoff_year = df.attrs.get("payoff_year")
loan_amount = df.attrs.get("loan_amount")
pmt_min = df.attrs.get("monthly_payment_min")

y1 = df.iloc[0]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Loan Amount", f"${loan_amount:,.0f}")
c2.metric("Min Mortgage Pmt (monthly)", f"${pmt_min:,.0f}")
c3.metric("Cash Available (Year 1)", f"${y1['Cash Available']:,.0f}")
c4.metric("Out-of-Pocket (Year 1)", f"${y1['Out of Pocket']:,.0f}")
c5.metric("Takehome Profit (Year 1)", f"${y1['Takehome Profit']:,.0f}")
c6.metric("Payoff Year", "No payoff" if payoff_year is None else f"Year {payoff_year}")

tabs = st.tabs(["Summary Charts", "Year-by-Year View"])

zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")

with tabs[0]:
    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("Out-of-Pocket Per Year")
        chart1 = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="Year:Q",
                y=alt.Y("Out of Pocket:Q", title="$/year (+you pay, -surplus)"),
                tooltip=["Year", alt.Tooltip("Out of Pocket:Q", format=",.0f")]
            )
        )
        st.altair_chart((chart1 + zero_line).interactive(), use_container_width=True)

        st.subheader("Takehome Profit Per Year")
        # Bar chart is best here
        profit_bar = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Takehome Profit:Q", title="$/year"),
                tooltip=["Year", alt.Tooltip("Takehome Profit:Q", format=",.0f")]
            )
        )
        st.altair_chart(profit_bar.interactive(), use_container_width=True)

    with right:
        st.subheader("Cumulative Out-of-Pocket")
        chart2 = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="Year:Q",
                y=alt.Y("Cumulative Out of Pocket:Q", title="Cumulative $"),
                tooltip=["Year", alt.Tooltip("Cumulative Out of Pocket:Q", format=",.0f")]
            )
        )
        st.altair_chart((chart2 + zero_line).interactive(), use_container_width=True)

        st.subheader("Ending Mortgage Balance")
        bal = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="Year:Q",
                y=alt.Y("Ending Balance:Q", title="Balance ($)"),
                tooltip=["Year", alt.Tooltip("Ending Balance:Q", format=",.0f")]
            )
        )
        st.altair_chart(bal.interactive(), use_container_width=True)

with tabs[1]:
    st.subheader("Year-by-Year Table")
    st.dataframe(
        df.style.format({
            "Total Rent": "${:,.0f}",
            "Cash Available": "${:,.0f}",
            "Out of Pocket": "${:,.0f}",
            "Takehome Profit": "${:,.0f}",
            "Extra Principal Paid": "${:,.0f}",
            "Interest Paid": "${:,.0f}",
            "Principal Paid": "${:,.0f}",
            "Ending Balance": "${:,.0f}",
            "Cumulative Out of Pocket": "${:,.0f}",
        }),
        use_container_width=True,
        height=520,
    )

st.caption(
    "Definitions: Cash Available = (Rent after mgmt fee) − COGS. "
    "Out-of-Pocket is the gap when cash available can’t cover the mortgage payment. "
    "Takehome Profit is the portion of surplus not reinvested into extra principal. "
    "Reinvest slider controls what fraction of surplus becomes extra principal (0% = keep it, 100% = pay down faster)."
)