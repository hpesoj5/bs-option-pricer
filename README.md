# Black Scholes Option Pricing Model

This project is solely for me to **understand more about the Black-Scholes equation** and create an app to **visualise** how option price are affected by each of the five factors:

- underlying price
- strike price
- time to expiry (TTE)
- volatility
- risk-free rate

The app is available on https://hpesoj-bs-option-pricer.streamlit.app/. The Python version used in this repo is 3.12.12.

## App Usage

The main UI consists of a main body and a sidebar. Users are given the option to choose from a stock (yet to implement) or select their own underlying price.

There will be two heatmaps showing the option price against **TTE and strike price**, and against **volatility and strike price**. 

In the sidebar, users can change the general parameters such as the underlying price and strike price range for the heatmap, and individual parameters for each of the two heatmaps.

There is also an option to display a **profit and loss (PnL)** heatmap given the option premium.

## Conclusions

Conclusions and graphs are available in the main.ipynb Jupyter notebook file.

## Disclaimer

This app isn't meant to be an accurate representation or prediction of option prices.
