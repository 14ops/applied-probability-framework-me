import math
import pandas as pd
from src.python.indicators.ta import rsi, atr


def test_rsi_basic_levels():
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    out = rsi(close, 14)
    # strictly increasing series approaches RSI ~ 100 after warmup
    assert out.iloc[-1] > 70


def test_atr_non_negative():
    high = pd.Series([10, 11, 12, 13, 14])
    low = pd.Series([9, 9.5, 10, 11, 12])
    close = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5])
    a = atr(high, low, close, 3)
    assert (a.dropna() >= 0).all()


