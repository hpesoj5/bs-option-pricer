import streamlit as st
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
def create_heatmap(underlying, vol, close):
    """
    Creates a 2d-heatmap of option prices with the horizontal axis being time-to-expiry and vertical axis being strike price.
    """
    lower = max(0, close - 5)
    strike = np.round(np.linspace(lower, close + 5, int(close + 5 - lower + 1), endpoint=True), 2)
    tte = np.round(np.linspace(0, 1, 11, endpoint=True), 2)
    strike_mesh, tte_mesh = np.meshgrid(strike, tte)
    call_grid, put_grid = black_scholes(underlying, strike_mesh, vol, tte_mesh, RISK_FREE_RATE)
    
    call_grid = pd.DataFrame(call_grid).fillna(0)
    put_grid = pd.DataFrame(put_grid).fillna(0)

    call_grid.index = tte
    call_grid.columns = strike
    put_grid.index = tte
    put_grid.columns = strike

    call_grid = call_grid.transpose()
    put_grid = put_grid.transpose()

    return (call_grid, put_grid)

def style_df(styler):
    styler.background_gradient(axis=None, cmap='plasma')
    styler.format(precision=2)
    return styler

def style_df_pnl(styler):
    styler.background_gradient(axis=None, cmap='RdYlGn', vmin=-10, vmax=10)
    styler.format(precision=2)
    return styler

def show_heatmap(call_grid, put_grid):
    pd.set_option('display.float_format', '{:.2f}'.format)
    st.subheader("Call Option")
    st.dataframe(call_grid.style.pipe(style_df), selection_mode='single-cell')
    st.subheader("Put Option")
    st.dataframe(put_grid.style.pipe(style_df), selection_mode='single-cell')

def main():

    st.title('Black-Scholes Option Heatmap')
    ticker_df = get_tickers()
    tickers = ticker_df.loc[:, 'Ticker']
    ticker_df = ticker_df.set_index('Ticker')

    ticker_select = st.sidebar.selectbox(
        'Ticker',
        tickers,
    )
    close = ticker_df.loc[ticker_select, '5-day Average Close']
    underlying = st.sidebar.slider('Underlying Price', 0.0, ticker_df.loc[ticker_select, 'Upper Bound'], (close))
    volatility = st.sidebar.slider('Implied Volatility', 0.0, 3.0, (1.0))
    call_grid, put_grid = create_heatmap(underlying, volatility, close)
    
    show_heatmap(call_grid, put_grid)

if __name__ == "__main__":
    main()