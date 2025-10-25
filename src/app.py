import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / 'data'

RISK_FREE_RATE = 0.036

@st.cache_data
def get_tickers():
    """
    Returns a dataframe containing ticker and upper bound prices.
    The upper bound variable is to set the rightmost value of the underlying price slider. It is given by 2 times the max high price during the timeframe of the data.
    """

    tickers = [] # stores (ticker, max high price)
    upper_bounds = []
    avg_closes = []
    for filename in os.listdir(data_dir):
        try:
            df = pd.read_csv(data_dir / filename)
            high_prices = df.loc[:, 'High']
            high = np.max(high_prices)
            ticker = filename[:-4]
            avg_last_5_closes = np.round(np.mean(df.loc[:, 'Close'][-5:]))

            tickers.append(ticker)
            upper_bounds.append(np.ceil(2 * high))
            avg_closes.append(avg_last_5_closes)

        except pd.errors.ParserError:
            continue

    res = pd.DataFrame({
        'Ticker': tickers,
        'Upper Bound': upper_bounds,
        '5-day Average Close': avg_closes,
    })

    return res

@st.cache_data
def black_scholes(underlying, strike, vol, tte, rfr):
    """
    Estimates the price of the call and put options based on the 5 factors.
    """

    asset_or_nothing = (np.log(underlying / strike) + (rfr + vol * vol / 2) * tte) / (vol * np.sqrt(tte))
    cash_or_nothing = asset_or_nothing - (vol * np.sqrt(tte))
    call = norm.cdf(asset_or_nothing) * underlying - norm.cdf(cash_or_nothing) * strike * (np.e ** (-rfr * tte))
    put = norm.cdf(-cash_or_nothing) * strike * (np.e ** (-rfr * tte)) - norm.cdf(-asset_or_nothing) * underlying

    return (call, put)

@st.cache_data
def create_heatmap_tte(underlying, vol, lower_strike, upper_strike, lower_tte, upper_tte):
    """
    Creates a 2d-heatmap of option prices with the horizontal axis being time-to-expiry and vertical axis being strike price.
    """
    strike = np.round(np.linspace(lower_strike, upper_strike, 11, endpoint=True), 2)
    tte = np.round(np.linspace(lower_tte, upper_tte, 11, endpoint=True), 2)
    strike_mesh, tte_mesh = np.meshgrid(strike, tte)
    call_grid, put_grid = black_scholes(underlying, strike_mesh, vol, tte_mesh, RISK_FREE_RATE)
    
    call_grid = pd.DataFrame(call_grid).fillna(0)
    put_grid = pd.DataFrame(put_grid).fillna(0)

    call_grid.index = tte
    call_grid.columns = strike
    put_grid.index = tte
    put_grid.columns = strike

    call_grid = call_grid.transpose().sort_index(axis=0, ascending=False)
    put_grid = put_grid.transpose().sort_index(axis=0, ascending=False)

    return (call_grid, put_grid)

@st.cache_data
def create_heatmap_vol(underlying, tte, lower_strike, upper_strike, lower_vol, upper_vol):
    """
    Creates a 2d-heatmap of option prices with the horizontal axis being volatilty and vertical axis being strike price.
    """
    strike = np.round(np.linspace(lower_strike, upper_strike, 11, endpoint=True), 2)
    vol = np.round(np.linspace(lower_vol, upper_vol, 11, endpoint=True), 2)
    strike_mesh, vol_mesh = np.meshgrid(strike, vol)
    call_grid, put_grid = black_scholes(underlying, strike_mesh, vol_mesh, tte, RISK_FREE_RATE)
    
    call_grid = pd.DataFrame(call_grid).fillna(0)
    put_grid = pd.DataFrame(put_grid).fillna(0)

    call_grid.index = vol
    call_grid.columns = strike
    put_grid.index = vol
    put_grid.columns = strike

    call_grid = call_grid.transpose().sort_index(axis=0, ascending=False)
    put_grid = put_grid.transpose().sort_index(axis=0, ascending=False)

    return (call_grid, put_grid)

def style_df(styler):
    styler.background_gradient(axis=None, cmap='plasma')
    styler.format(precision=2)
    styler.format_index(precision=2)
    return styler

def style_df_pnl(styler):
    styler.background_gradient(axis=None, cmap='RdYlGn', vmin=-10, vmax=10)
    styler.format(precision=2)
    styler.format_index(precision=2)
    return styler

def show_heatmap(call_grid, put_grid, pnl: bool, cost):
    """
    Displays the Option Price Heatmap onto the screen.
    If pnl = True, displays the Profit and Loss instead of Option Prices
    """
    try:
        col1, col2 = st.columns(2, gap='large')
        if not pnl:
            col1.subheader("Call Option")
            col2.subheader("Put Option")
            with col1:
                st.dataframe(
                    call_grid.style.pipe(style_df),
                    row_height=45,
                    height=532
                )
            
            with col2:
                st.dataframe(
                    put_grid.style.pipe(style_df),
                    row_height=45,
                    height=532
                )

        else:
            col1.subheader("Call Option")
            col2.subheader("Put Option")
            with col1:
                call_grid = call_grid - cost
                st.dataframe(
                    call_grid.style.pipe(style_df_pnl),
                    row_height=45,
                    height=532
                )

            with col2:
                put_grid = put_grid - cost
                st.dataframe(
                    put_grid.style.pipe(style_df_pnl),
                    row_height=45,
                    height=532
                )
    except KeyError:
        col1.markdown("Range too small! :cry:")
        col2.markdown("Range not big enough! :sob:")

def set_page_config():
    """
    Sets the page options on load
    """
    st.title("Black-Scholes Option Heatmap")
    st.set_page_config(
        page_title = "Option Pricer",
        layout = 'wide',
    )

def main():
    
    set_page_config()

    # SIDEBAR UI
    ticker_df = get_tickers()
    tickers = pd.concat([pd.Series(["None"]), ticker_df.loc[:, 'Ticker']])
    ticker_df = ticker_df.set_index('Ticker')
    st.sidebar.subheader("Parameters", divider=True)
    ticker_select = st.sidebar.selectbox(
        'Ticker',
        tickers,
    )

    # close = ticker_df.loc[ticker_select, '5-day Average Close']
    underlying = st.sidebar.number_input("Underlying Price", min_value = 0.0, value=50.0, step=0.01, disabled=(ticker_select != "None"))
    lower_strike, upper_strike = st.sidebar.slider("Strike Range", min_value=0.0, max_value=100.0, value=(25.0, 75.0))
    
    st.sidebar.subheader("TTE against Strike", divider=True)
    volatility = st.sidebar.number_input("Implied Volatility", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    lower_tte, upper_tte = st.sidebar.slider("TTE Range", min_value=0.0, max_value=5.0, value=(0.0, 1.0))
    
    st.sidebar.subheader("Vol (σ) against Strike", divider=True)
    tte = st.sidebar.number_input("Time To Expiry (in years)", min_value=0.01, value=0.5, step=0.01)
    lower_vol, upper_vol = st.sidebar.slider("Volatility Range", min_value=0.0, max_value=1.0, value=(0.25, 0.75))

    pnl = st.sidebar.checkbox("PnL Heatmap", value=False)
    cost = st.sidebar.slider("Option Premium", min_value=0.0, max_value=100.0, value=(0.0), disabled=not pnl)

    # MAIN UI
    # STRIKE AGAINST TTE
    tte_container = st.container()
    with tte_container:
        st.header("TTE against Strike Price", divider='gray')
        call_grid_tte, put_grid_tte = create_heatmap_tte(underlying, volatility, lower_strike, upper_strike, lower_tte, upper_tte)
        show_heatmap(call_grid_tte, put_grid_tte, pnl, cost)

    # STRIKE AGAINST VOLATILITY
    vol_container = st.container()
    with vol_container:
        st.header("Volatility against Strike Price", divider='gray')
        call_grid_vol, put_grid_vol = create_heatmap_vol(underlying, tte, lower_strike, upper_strike, lower_vol, upper_vol)
        show_heatmap(call_grid_vol, put_grid_vol, pnl, cost)


if __name__ == "__main__":
    main()