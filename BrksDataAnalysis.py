import warnings

import matplotlib.pyplot as plt
import pandas as p
import numpy as np

from DataRead.GoogleStockReader import GoogleStockReader
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
import datetime
from dateutil import parser


class Analysis:

    class StopLossLim:
        def __init__(self, price):
            self.price = price

    class DataBlock:
        def __init__(self, series, date):
            self.series = series
            self.date = date

        def getRollingDataForDay(self, window, operation='mean'):
            s = self.series.resample("1D").fillna('ffill').rolling(window=window)
            if (operation == 'mean'):
                return s.mean()[self.date]
            if (operation == 'max'):
                return s.max()[self.date]
            if (operation == 'min'):
                return s.min()[self.date]
            if (operation == 'var'):
                return s.var()[self.date]

        def getRollingData(self, window, operation='mean'):
            s = self.series.resample("1D").fillna('ffill').rolling(window=window)
            if (operation == 'mean'):
                return s.mean()
            if (operation == 'max'):
                return s.max()
            if (operation == 'min'):
                return s.min()
            if (operation == 'var'):
                return s.var()

        def peaked(self, distBtwnMinAndMax, dataPieces, minWindow=30, maxWindow=20, peakWidthMax=20, MinToMaxRatio=.9):
            max = self.series.resample("1D").fillna('ffill').rolling(window=maxWindow).max()[self.date]
            min = self.series.resample("1D").fillna('ffill').rolling(window=minWindow).min()[self.date]

            index = 0  # index of current datablock
            for i in range(0, len(dataPieces)):
                if (dataPieces[i].date == self.date):
                    index = i
                    break

            # Peak is tall enough
            diff = min / max
            if diff > MinToMaxRatio:
                return False

            # If is min
            if min == self.open:
                a = False
                dist = index - distBtwnMinAndMax
                for i in range(dist, index):
                    if dataPieces[i].open == max:  # If max was within distBtwnMinAndMax
                        a = True
                if not a:
                    return False
            else:
                return False

            # Peak is skinny enough
            dist = index - distBtwnMinAndMax

            for i in range(dist, index):
                if dataPieces[i].open <= min <= dataPieces[i + 1].open:
                    return True
            return False

    def __init__(self, company, start=datetime.datetime(2014, 1, 1), end=datetime.date.today()):

        self.company = company
        data = p.read_csv(GoogleStockReader.getStockUrl(company, start, end), parse_dates=True, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        d = [x for x in data['Date'].values]
        o = [x for x in data['Open'].values]
        h = [x for x in data['High'].values]
        l = [x for x in data['Low'].values]
        c = [x for x in data['Close'].values]

        dates = []
        open = []
        high = []
        low = []
        close = []

        for x in range(1, len(d)):
            dates.append(parser.parse(d[x]))
            open.append(float(o[x]))
            high.append(float(h[x]))
            low.append(float(l[x]))
            close.append(float(c[x]))

        dataPieces = []
        s = p.Series(open, dates)
        s.cumsum()
        # Initialize all basic data
        for dayIndex in range(0, len(dates)):
            db = Analysis.DataBlock(s, dates[dayIndex])
            db.index = dayIndex
            db.open = open[dayIndex]
            db.high = high[dayIndex]
            db.low = low[dayIndex]
            db.close = close[dayIndex]
            dataPieces.append(db)

        dataPieces.reverse()
        dates.reverse()
        open.reverse()
        self.dataPoints = dataPieces;
        self.dates = dates
        self.open = open

    def run(self):

        s = p.Series(self.open, self.dates)
        s.cumsum()

        movAvg5 = s.resample("1D").fillna('ffill').rolling(window=5,)
        movAvg10 = s.resample("1D").fillna('ffill').rolling(window=10,)
        movAvg20 = s.resample("1D").fillna('ffill').rolling(window=20,)
        movAvg40 = s.resample("1D").fillna('ffill').rolling(window=40,)
        movAvg100 = s.resample("1D").fillna('ffill').rolling(window=100,)
        movAvg160 = s.resample("1D").fillna('ffill').rolling(window=160,)
        movAvg320 = s.resample("1D").fillna('ffill').rolling(window=320,)

        print(self.dataPoints[50].getRollingDataForDay(5))
        print(len(self.dates))



        s.plot(style ='k', markersize=5)
        # movAvg5.mean().plot(style='b')
        # movAvg5.min().plot(style='g')
        # movAvg20.max().plot(style='orange')
        movAvg100.mean().plot(style='r')
        # movAvg80.min().plot(style='g')

        initialBalance = 20000
        balance = initialBalance
        multipler = 1
        equity=0
        stockOwned = 0
        onlyBuyWhenNumIsAbove = 0
        onlySellWhenNumIsAbove = 0

        sll = Analysis.StopLossLim(0)

        lastHigh = 0

        priceSoldAt = 0

        for dayIndex in range(len(self.dataPoints)):
            db = self.dataPoints[dayIndex]

            # Adjust StopLossLim
            if (stockOwned > 0):
                if (db.open > lastHigh):
                    sll.price = db.open * .90
                    lastHigh = db.open



            # Double peak analysis,
            # if price just peaked and dropped


            # see likelihood that it will go up again,
            # if it seems likely to go up,
            #  buy some stocks,
            # if it seems unlikely dont.
            #  Put a stop limit at some percent loss below buy price.
            #  If reaches some threshold compared to peak,
            # put stop limit,
            # determine risk of not selling for each time amount held on for,
            # then sell when risk outweighs reward



            # Does nothing right now was just curious how volatility affects the market
            recentVolatily = db.getRollingDataForDay(40, 'var')
            longTermVolatily = db.getRollingDataForDay(100, 'var')

            # if (not math.isnan(longTermVolatily)):
            #     if recentVolatily*.9 > longTermVolatily:
            #      plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=10)

            # peaked = db.peaked(60, dataPieces, minWindow=5, maxWindow=40)
            # if peaked:
            #     plt.plot(db.date, db.open, marker='*', linestyle='--', color='pink', markersize=10)

            plt.plot(db.date, sll.price, marker='o', linestyle='--', color='pink', markersize=5)

            # Within range to 20 day min, and its starting to pick up
            goodBuyShortTerm = db.open*.98 <= db.getRollingDataForDay(20, 'min') and db.getRollingDataForDay(50) > db.getRollingDataForDay(100)

            # Within range to 100 day min, and its starting to pick up
            goodBuyLongTerm = db.open*.98 <= db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(100) > db.getRollingDataForDay(150)

            # Price is well below usual
            wellBelowUsual = (db.open * .97 < db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(100) > db.getRollingDataForDay(150))\
                             or db.open*.98 < db.getRollingDataForDay(150, 'min')

            # Most expensive its been in 20 days, and its slowing down
            goodSellShortTerm = db.open*.99 >= db.getRollingDataForDay(20, 'max') and db.getRollingDataForDay(5) < db.getRollingDataForDay(50)

            # Most expensive its been in 50 days, and its slowing down
            goodSellLongTerm = db.open*.98 >= db.getRollingDataForDay(50, 'max') and db.getRollingDataForDay(10) < db.getRollingDataForDay(100)


            # belowMarks = db.open <= db.getRollingData(60) #and db.getRollingData(3) < db.getRollingData(10)

            # Buy
            if (goodBuyShortTerm or goodBuyLongTerm or wellBelowUsual) and not (db.open < sll.price) or db.open < priceSoldAt*.98:
                if (balance > db.open):
                    stockToBuy = 1

                    # A number that correlates with how strong the buy will be
                    num = ((db.getRollingDataForDay(50) / db.getRollingDataForDay(4)) - 1) * 100
                    if (num > 1):
                        stockToBuy = int(num)
                    if (num < onlyBuyWhenNumIsAbove):
                        continue
                    plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=stockToBuy*2)
                    stockToBuy*=multipler
                    print('buy', stockToBuy, 'at', db.open, db.date)
                    if (stockOwned == 0):
                        sll.price = db.open * .90
                    lastHigh = db.open
                    stockOwned+=stockToBuy
                    balance-=stockToBuy*db.open


                    priceSoldAt = sll.price
                else:
                    print('No moneys')
            # Sell
            elif (goodSellLongTerm or goodSellShortTerm) and not (db.open < sll.price):
                if stockOwned > 0:
                    stockToSell = 1

                    # A number that correlates with how strong the sell will be
                    num = (db.getRollingDataForDay(15) / db.getRollingDataForDay(50) - 1) * 100
                    if (num > 1):
                        stockToSell = int(num)
                    if (num < onlySellWhenNumIsAbove):
                        continue
                    if (stockToSell*multipler > stockOwned):
                        stockToSell = (stockOwned)%multipler + 1

                    plt.plot(db.date, db.open, marker='o', linestyle='--', color='b', markersize=stockToSell*5)
                    stockToSell*=multipler
                    print('sell', stockToSell, 'at', db.open, db.date)
                    stockOwned -= stockToSell
                    balance += db.open*stockToSell
            # Stop limit reached, sell everything
            elif(db.open <= sll.price):
                print("Reached Limit Selling ", stockOwned, "on", db.date)
                sellAmt = stockOwned * sll.price
                balance += sellAmt
                stockOwned = 0
                if self.peakedRecently(db, self.dataPoints, dayIndex):
                    plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=30)
                else:
                    plt.plot(db.date, db.open, marker='*', linestyle='--', color='g', markersize=30)

                sll.price = 0
            else:
                    print("no stock")

        currentPrice = self.dataPoints[len(self.dataPoints)-1].open
        stockMoney=stockOwned*currentPrice
        equity = balance + stockMoney
        profit = equity - initialBalance
        percent = (100 - abs(((equity/initialBalance)*100)))*-1
        print('profit:',profit)
        print('stock money:',stockMoney)
        print('equity:',equity)
        print('gain percent:', percent)

        print('balance', balance)

        print(stockOwned)
        # plt.plot(dates, movAvg80, "b")
        plt.ylabel('some numbers')
        plt.show()

    def movingAvgIndex(self, db):
        # Within range to 20 day min, and its starting to pick up
        goodBuyShortTerm = db.open * .98 <= db.getRollingDataForDay(20, 'min') and db.getRollingDataForDay(
            50) > db.getRollingDataForDay(100)

        # Within range to 100 day min, and its starting to pick up
        goodBuyLongTerm = db.open * .98 <= db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(
            100) > db.getRollingDataForDay(150)

        # Price is well below usual
        wellBelowUsual = (db.open * .97 < db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(
            100) > db.getRollingDataForDay(150)) \
                         or db.open * .98 < db.getRollingDataForDay(150, 'min')

        # Most expensive its been in 20 days, and its slowing down
        goodSellShortTerm = db.open * .99 >= db.getRollingDataForDay(20, 'max') and db.getRollingDataForDay(
            5) < db.getRollingDataForDay(50)

        # Most expensive its been in 50 days, and its slowing down
        goodSellLongTerm = db.open * .98 >= db.getRollingDataForDay(50, 'max') and db.getRollingDataForDay(
            10) < db.getRollingDataForDay(100)

        # belowMarks = db.open <= db.getRollingData(60) #and db.getRollingData(3) < db.getRollingData(10)

        num = 0
        # Buy
        if goodBuyShortTerm or goodBuyLongTerm or wellBelowUsual:
            # A number that correlates with how strong the buy will be
            num = ((db.getRollingDataForDay(50) / db.getRollingDataForDay(4)) - 1) * 100
        # Sell
        elif goodSellLongTerm or goodSellShortTerm:
            # A number that correlates with how strong the sell will be
            num = -1*(db.getRollingDataForDay(15) / db.getRollingDataForDay(50) - 1) * 100
        return num

    def diff_from_moving_avg(self, db, period):
        movAvg = db.getRollingDataForDay(period)
        return db.open/movAvg


    # Using this to get best markers for certain things
    def bare_run (self):

        # Plot actual open data
        s = p.Series(self.open, self.dates)
        s.cumsum()
        s.plot(style ='k', markersize=5)

        for dayIndex in range(len(self.dataPoints)):
            db = self.dataPoints[dayIndex]
            if (dayIndex > 5):
                slope = self.calculateSlope(5, db.getRollingData(5), db.date)
                print(slope)
            num = self.movingAvgIndex(db)
            # if num > 0:
            #     plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=num)
            # elif num < 0:
            #     plt.plot(db.date, db.open, marker='o', linestyle='--', color='b', markersize=num*-1)
            if db.peaked(150, self.dataPoints, minWindow=40, maxWindow=100):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=4)
            if db.peaked(250, self.dataPoints, minWindow=40, maxWindow=50):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='y', markersize=4)
            if db.peaked(30, self.dataPoints, minWindow=10, maxWindow=20, MinToMaxRatio=.94):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=4)
            if db.peaked(80, self.dataPoints):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=8)

        plt.show()


    def csv_indices(self):
        columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10','x11','x12','x13', 'x14','x15', 'y1','y2','y3']
        index = self.dates
        df = p.DataFrame(index=index, columns=columns)
        # TODO add DOW && S&P index
        # TODO add PercentDiffBetweenMovAvg
        # TODO add PE ratio

        # TODO add  5, 10, 20, 40, 80, 160  days later actual db.open


        for dayIndex in range(len(self.dataPoints)):
            db = self.dataPoints[dayIndex]
            df.at[db.date, 'x1'] = db.open

            num = self.movingAvgIndex(db)
            if num > 0:
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=num)
            elif num < 0:
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='b', markersize=num*-1)

            df.at[db.date, 'x2'] = num
            if db.peaked(150, self.dataPoints, minWindow=40, maxWindow=100):
                df.at[db.date, 'x3'] = 1
            else:
                df.at[db.date, 'x3'] = 0
            if db.peaked(250, self.dataPoints, minWindow=40, maxWindow=50):
                df.at[db.date, 'x4'] = 1
            else:
                df.at[db.date, 'x4'] = 0
            if db.peaked(30, self.dataPoints, minWindow=10, maxWindow=20, MinToMaxRatio=.94):
                df.at[db.date, 'x5'] = 1
            else:
                df.at[db.date, 'x5'] = 0
            if (dayIndex >= 5):
                df.at[db.date, 'x6'] = self.calculateSlope(5, db.getRollingData(5), db.date)
            if (dayIndex >= 10):
                 df.at[db.date, 'x7'] = self.calculateSlope(10, db.getRollingData(10), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'x8'] = self.calculateSlope(20, db.getRollingData(20), db.date)
            if (dayIndex >= 40):
                df.at[db.date, 'x9'] = self.calculateSlope(40, db.getRollingData(40), db.date)
            if (dayIndex >= 80):
                df.at[db.date, 'x10'] = self.calculateSlope(80, db.getRollingData(80), db.date)
            if (dayIndex >= 160):
                df.at[db.date, 'x11'] = self.calculateSlope(160, db.getRollingData(160), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'x12'] = self.diff_from_moving_avg(db, 20)
            if (dayIndex >= 40):
                df.at[db.date, 'x13'] = self.diff_from_moving_avg(db, 40)
            if (dayIndex >= 80):
                df.at[db.date, 'x14'] = self.diff_from_moving_avg(db, 80)
            if (dayIndex >= 160):
                df.at[db.date, 'x15'] = self.diff_from_moving_avg(db, 160)
            if dayIndex + 5 < len(self.open):
                df.at[db.date, 'y1'] = self.open[dayIndex + 5]
            if dayIndex + 10 < len(self.open):
                df.at[db.date, 'y2'] = self.open[dayIndex + 10]
            if dayIndex + 50 < len(self.open):
                df.at[db.date, 'y3'] = self.open[dayIndex + 50]


            # if db.peaked(80, self.dataPoints):
            #     plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=8)
        df.to_csv("C:\\Users\\cole\\Desktop\\Brks\\" + self.company + "_processed.csv")
    def peakedRecently(self, db, index, peakLookBack=3):

        if index >= peakLookBack and db.getRollingDataForDay(2, 'mean') > db.getRollingDataForDay(5, 'mean'):
            for indexOfRecentDays in range(0, peakLookBack):
                if self.dataPoints[index - indexOfRecentDays].peaked(80, self.dataPoints, minWindow=20, maxWindow=40):
                    return True
        return False
    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def calculateSlope(self, numDays, series, date):
        xs = []
        ys = []
        for i in range(0, numDays):
            newDate = date - datetime.timedelta(days=i)
            ys.append(series[newDate])
            xs.append(i)
        # dy = (np.roll(ys, -1, axis=1) - ys)[:,:-1]
        # dx = (np.roll(xs, -1, axis=0) - xs)[:-1]
        ys.reverse()
        from scipy.stats import linregress
        return linregress(xs, ys)[0]


if __name__ == "__main__":
    A = Analysis("PFPT", start=datetime.datetime(2012, 1, 1))
    # A.bare_run()
    A.csv_indices()