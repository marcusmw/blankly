"""
    Oscillator wrappers
    Copyright (C) 2021 Brandon Fan

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Any

import numpy as np
import pandas as pd
import tulipy as ti

from blankly.indicators.utils import check_series, convert_to_numpy


def rsi(data: Any, period: int = 14, round_rsi: bool = False, use_series=False) -> np.array:
    """ Implements RSI Indicator """
    if period >= len(data):
        return pd.Series() if use_series else []
    if check_series(data):
        use_series = True
    data = convert_to_numpy(data)
    rsi_values = ti.rsi(data, period)
    if round_rsi:
        rsi_values = np.round(rsi_values, 2)
    return pd.Series(rsi_values) if use_series else rsi_values

def cci(high_data: Any, low_data: Any, close_data: Any, period: int = 5, use_series=False) -> np.array:
    """ Implements CCI Indicator """
    if check_series(high_data) or check_series(low_data) or check_series(close_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    close_data = convert_to_numpy(close_data)
    cci_values = ti.cci(high_data, low_data, close_data, period)
    return pd.Series(cci_values) if use_series else cci_values

def adx(high_data, low_data, close_data, period=14, use_series=False):
    if check_series(high_data) or check_series(low_data) or check_series(close_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    close_data = convert_to_numpy(close_data)
    adx_output = ti.adx(high_data, low_data, close_data, period)
    di_output = ti.di(high_data, low_data, close_data, period)
    plus_di = di_output[0]
    minus_di = di_output[1]
    if use_series:
        adx_output = pd.Series(adx_output)
        plus_di = pd.Series(plus_di)
        minus_di = pd.Series(minus_di)
    return adx_output, plus_di, minus_di

def awesome_oscillator(high_data, low_data, use_series=False):
    if check_series(high_data) or check_series(low_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    ao = ti.ao(high_data, low_data)
    return pd.Series(ao) if use_series else ao

def momentum_oscillator(input_data, period=5, use_series=False):
    if check_series(input_data):
        use_series = True
    input_data = convert_to_numpy(input_data)
    mom_output = ti.mom(input_data, period)
    return pd.Series(mom_output) if use_series else mom_output

#Problem, this seems to be wrong by a little bit.
def macd(input_data, short_period=12, long_period=26, signal_period=9, use_series=False):
    if check_series(input_data):
        use_series = True
    input_data = convert_to_numpy(input_data)
    macd, macd_signal, macd_histogram = ti.macd(input_data, short_period, long_period, signal_period)
    if use_series:
        return pd.Series(macd), pd.Series(macd_signal), pd.Series(macd_histogram)
    else:
        return macd, macd_signal, macd_histogram
    
def stochrsi(input_data, period=14, use_series=False):
    if check_series(input_data):
        use_series = True
    input_data = convert_to_numpy(input_data)
    stochrsi_values = ti.stochrsi(input_data, period)
    stochrsi_K = np.convolve(stochrsi_values, np.ones(3)/3, mode='valid')
    stochrsi_D = np.convolve(stochrsi_K, np.ones(3)/3, mode='valid')
    if use_series:
        return pd.Series(stochrsi_values), pd.Series(stochrsi_K), pd.Series(stochrsi_D)
    else:
        return stochrsi_values, stochrsi_K, stochrsi_D

def williams_r(high_data, low_data, close_data, period=14, use_series=False):
    if check_series(high_data) or check_series(low_data) or check_series(close_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    close_data = convert_to_numpy(close_data)
    willr = ti.willr(high_data, low_data, close_data, period)
    return pd.Series(willr) if use_series else willr

def ultimate_oscillator(high_data, low_data, close_data, short_period=2, medium_period=3, long_period=5, use_series=False):
    if check_series(high_data) or check_series(low_data) or check_series(close_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    close_data = convert_to_numpy(close_data)
    ultosc = ti.ultosc(high_data, low_data, close_data, short_period, medium_period, long_period)
    return pd.Series(ultosc) if use_series else ultosc

def aroon_oscillator(high_data: Any, low_data: Any, period=14, use_series=False):
    if check_series(high_data) or check_series(low_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    aroonsc = ti.aroonosc(high_data, low_data, period=period)
    return pd.Series(aroonsc) if use_series else aroonsc

def chande_momentum_oscillator(data, period=14, use_series=False):
    if check_series(data):
        use_series = True
    data = convert_to_numpy(data)
    cmo = ti.cmo(data, period)
    return pd.Series(cmo) if use_series else cmo


def absolute_price_oscillator(data, short_period=12, long_period=26, use_series=False):
    if check_series(data):
        use_series = True
    data = convert_to_numpy(data)
    apo = ti.apo(data, short_period, long_period)
    return pd.Series(apo) if use_series else apo


def percentage_price_oscillator(data, short_period=12, long_period=26, use_series=False):
    if check_series(data):
        use_series = True
    data = convert_to_numpy(data)
    ppo = ti.ppo(data, short_period, long_period)
    return pd.Series(ppo) if use_series else ppo


def stochastic_oscillator(high_data, low_data, close_data, pct_k_period=14, pct_k_slowing_period=3, pct_d_period=3,
                          use_series=False):
    if check_series(high_data) or check_series(low_data) or check_series(close_data):
        use_series = True
    high_data = convert_to_numpy(high_data)
    low_data = convert_to_numpy(low_data)
    close_data = convert_to_numpy(close_data)
    stoch = ti.stoch(high_data, low_data, close_data, pct_k_period, pct_k_slowing_period, pct_d_period)
    return pd.Series(stoch) if use_series else stoch

def cm_ult_macd_mtf(data):
    "Runs on scraped data"
    if len(data) >= 1:
        histogram = data[-1].get('CM_histogram')
        cross = data[-1].get('CM_cross')
    else:
        return 0
    
    if histogram is not None and cross is not None:
        if cross == 0 or cross == 1e+100:
            return 0
        # Cross is active therefore buy/sell
        if histogram > 0:
            return 1
        elif histogram < 0:
            return -1
        else:
            return 0
    else:
        return 0
    

def squeeze_mom(data):
    "Runs on scraped data"
    if len(data) >= 2:
        #cross colour: gray is 6, black is 5
        mom = data[-1].get('Squeeze_mom_plot') 
        cross = data[-1].get('Squeeze_mom_plot2') 
        last_mom = data[-2].get('Squeeze_mom_plot') 
        last_cross = data[-2].get('Squeeze_mom_plot2')
        
        if mom is not None and cross is not None and last_mom is not None and last_cross is not None:
            if cross  == 6 and last_cross == 5 and mom> 0:
                return 1
            elif last_mom >= 0 and mom < 0:
                return -1
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
def super_trend(data):
    "Runs on scraped data"
    if len(data) >= 1:
        buy = data[-1].get("Super_Trend_Buy2")
        sell = data[-1].get("Super_Trend_Sell2")
    else:
        return 0

    if buy is not None and sell is not None:
        if buy == 1:
            return 1
        elif sell == 1:
            return -1
        else:
            return 0
    else:
        return 0

def ult_ma_mtf_v2(data):
    "Runs on scraped data"
    if len(data) >= 2:
        color = data[-1].get("Ult_MA_MTF_colorer")
        prev_color = data[-2].get("Ult_MA_MTF_colorer")
    else:
        return 0

    if color is not None and prev_color is not None:
        if color == prev_color:
            return 0
        elif color == 0:
            return 1
        else:
            return -1
    else:
        return 0

def wave_trend_osc(data):
    "Runs on scraped data"
    if len(data) >= 1:
        osc = data[-1].get("Wave_Trend_Osc")
        signal = data[-1].get("Wave_Trend_Signal")
        overbought = data[-1].get("Wave_Trend_Over")
        underbought = data[-1].get("Wave_Trend_Under")
    else:
        return 0

    if osc is not None and signal is not None and overbought is not None and underbought is not None:
        if osc > overbought and osc <= signal:
            return -1
        elif osc < underbought and osc >= signal:
            return 1
        else:
            return 0
    else:
        return 0
    
def williams_vix_fix(data):
    "Runs on scraped data"
    if len(data) >= 1:
        w_vf = data[-1].get('Williams_VF_colorer')
    else:
        return 0

    if w_vf is not None:
        if w_vf == 0:
            return 1
        else:
            return 0
    else:
        return 0