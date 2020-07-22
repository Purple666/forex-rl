import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import ta
from ta.momentum import stoch, stoch_signal
from ta.trend import ema
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore

# symbol=["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "USDCHF", "USDCAD", "EURGBP"]
# symbol=["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD", "AUDJPY"]
symbol=["GBPJPY"]
#

def fast_stochastic(lowp, highp, closep, period=14, smoothing=3):
    """ calculate slow stochastic
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K
    """
    low_min = lowp.rolling(window=period).min()
    high_max = highp.rolling(window=period).max()
    k_fast = 100 * (closep - low_min)/(high_max - low_min)
    d_fast = k_fast.rolling(window = 3).mean()
    d_slow = d_fast.rolling(window=smoothing).mean()
    return d_fast, d_slow


def gen_data(symbol=symbol):
    mt5.initialize()
    x_list = []
    y_list = []

    for s in symbol:

        # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M30, 0, 250000 // 2)
        # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, 250000)
        # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 250000 * 3)
        # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H1, 0,62500)
        r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H4, 0,15000)
        # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0,2600)
        df = pd.DataFrame(r)
        try:
            print(s)
            point = mt5.symbol_info(s).point
            str_point = str(point)
            if "e-" in str_point:
                point = int(str_point[-1])
                print(point)
                df *= 10 ** point
            else:
                point = len(str_point.rsplit(".")[1])
                if point == 1:
                    point = 0
                df *= 10 ** point
            df = np.round(df, 0)
        except:
            pass

        df["sig1"] = ta.momentum.rsi(df.close, 9)
        f, d = fast_stochastic(df.low, df.high, df.close, 5)
        df["sig2"] = d - f
        exp1 = df.close.ewm(span=16, adjust=False).mean()
        exp2 = df.close.ewm(span=8, adjust=False).mean()
        macd = exp1 - exp2
        macd_sig = macd.rolling(window=6).mean() - macd
        df["sig3"] = macd_sig
        df["sig4"] = df.close.ewm(span=10, adjust=False).mean() - df.close.ewm(span=5, adjust=False).mean()
        df["sig5"] = df.close - df.close.shift(1)
        # df["sig5"] = np.log(df.close / df.close.shift(1))

        df = df.dropna()
        # df[["sig3", "sig5", "sig2", "sig4"]] = df[["sig3", "sig5", "sig2", "sig4"]].apply(zscore)
        # df[["sig5"]] = df[["sig5"]].apply(zscore)
        # x = df[["sig3"]]
        x = df[["sig3"]].apply(zscore)
        # x = MinMaxScaler().fit_transform(df[["sig5"]].values)
        x = np.array(x)
        # x = np.clip(x, -10, 10)


        y = np.array(df[["open"]])
        volatility = np.array(df[["open"]])
        # scale_y = np.array(StandardScaler().fit_transform(df[["open"]].vau))
        atr = np.array(ta.volatility.average_true_range(df["high"], df["low"], df["close"]))
        high = np.array(df[["high"]])
        low = np.array(df[["low"]])

        print("gen time series data")
        # x = x[100:]
        # y = y[100:]

        window_size = 20
        time_x = []
        time_y = []

        for i in range(len(y) - window_size):
            time_x.append(x[i:i + window_size])
            i += window_size
            time_y.append(y[i])

        x = np.array(time_x).reshape((-1, window_size, x.shape[-1]))
        y = np.array(time_y).reshape((-1, y.shape[-1]))
        #
        # x = [(x * 10 ** 3).astype(np.int32) * (10 ** -3)]

        atr = atr[-len(y):].reshape((-1, 1))
        high = high[-len(y):].reshape((-1, 1))
        low = low[-len(y):].reshape((-1, 1))
        volatility = volatility[-len(y):].reshape((-1, 1))
        y = [[y, volatility, atr, high, low]]

        # y = [[y]]
        x_list.append(x)
        y_list.append(y)

    np.save("x1", np.array(x_list))
    np.save("target1", np.array(y_list))

    print("done\n")


if __name__ == "__main__":
    gen_data()
