import warnings

import matplotlib.pyplot as plt
import pandas as p

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

        def getRollingData(self, window, operation='mean'):
            s = self.series.resample("1D").fillna('ffill').rolling(window=window)
            if (operation == 'mean'):
                return s.mean()[self.date]
            if (operation == 'max'):
                return s.max()[self.date]
            if (operation == 'min'):
                return s.min()[self.date]
            if (operation == 'var'):
                return s.var()[self.date]

        def peaked(self, distBtwnMinAndMax, dataPieces, minWindow=30, maxWindow=20, peakWidthMax=20, peakHeightMin=5):
            max = self.series.resample("1D").fillna('ffill').rolling(window=maxWindow).max()[self.date]
            min = self.series.resample("1D").fillna('ffill').rolling(window=minWindow).min()[self.date]

            index = 0  # index of current datablock
            for i in range(0, len(dataPieces)):
                if (dataPieces[i].date == self.date):
                    index = i
                    break

            # Peak is tall enough
            diff = max - min
            if diff < peakHeightMin:
                return False

            # If is min
            if min == self.open:
                a = False
                dist = index - distBtwnMinAndMax
                for i in range(dist, index):
                    if dataPieces[i].open == max:  # If max was within distBtwnMinAndMax
                        a = True
                if a == False:
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

        print(self.dataPoints[50].getRollingData(5))
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
            recentVolatily = db.getRollingData(40,'var')
            longTermVolatily = db.getRollingData(100,'var')

            # if (not math.isnan(longTermVolatily)):
            #     if recentVolatily*.9 > longTermVolatily:
            #      plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=10)

            # peaked = db.peaked(60, dataPieces, minWindow=5, maxWindow=40)
            # if peaked:
            #     plt.plot(db.date, db.open, marker='*', linestyle='--', color='pink', markersize=10)

            plt.plot(db.date, sll.price, marker='o', linestyle='--', color='pink', markersize=5)

            # Within range to 20 day min, and its starting to pick up
            goodBuyShortTerm = db.open*.98 <= db.getRollingData(20, 'min') and db.getRollingData(50) > db.getRollingData(100)

            # Within range to 100 day min, and its starting to pick up
            goodBuyLongTerm = db.open*.98 <= db.getRollingData(100, 'min') and db.getRollingData(100) > db.getRollingData(150)

            # Price is well below usual
            wellBelowUsual = (db.open*.97 < db.getRollingData(100, 'min') and db.getRollingData(100) > db.getRollingData(150))\
                             or db.open*.98 < db.getRollingData(150, 'min')

            # Most expensive its been in 20 days, and its slowing down
            goodSellShortTerm = db.open*.99 >= db.getRollingData(20, 'max') and db.getRollingData(5) < db.getRollingData(50)

            # Most expensive its been in 50 days, and its slowing down
            goodSellLongTerm = db.open*.98 >= db.getRollingData(50, 'max') and db.getRollingData(10) < db.getRollingData(100)


            # belowMarks = db.open <= db.getRollingData(60) #and db.getRollingData(3) < db.getRollingData(10)

            # Buy
            if (goodBuyShortTerm or goodBuyLongTerm or wellBelowUsual) and not (db.open < sll.price) or db.open < priceSoldAt*.98:
                if (balance > db.open):
                    stockToBuy = 1

                    # A number that correlates with how strong the buy will be
                    num = ((db.getRollingData(50)/db.getRollingData(4)) - 1)*100
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
                    num = (db.getRollingData(15)/ db.getRollingData(50) - 1) * 100
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
        goodBuyShortTerm = db.open * .98 <= db.getRollingData(20, 'min') and db.getRollingData(
            50) > db.getRollingData(100)

        # Within range to 100 day min, and its starting to pick up
        goodBuyLongTerm = db.open * .98 <= db.getRollingData(100, 'min') and db.getRollingData(
            100) > db.getRollingData(150)

        # Price is well below usual
        wellBelowUsual = (db.open * .97 < db.getRollingData(100, 'min') and db.getRollingData(
            100) > db.getRollingData(150)) \
                         or db.open * .98 < db.getRollingData(150, 'min')

        # Most expensive its been in 20 days, and its slowing down
        goodSellShortTerm = db.open * .99 >= db.getRollingData(20, 'max') and db.getRollingData(
            5) < db.getRollingData(50)

        # Most expensive its been in 50 days, and its slowing down
        goodSellLongTerm = db.open * .98 >= db.getRollingData(50, 'max') and db.getRollingData(
            10) < db.getRollingData(100)

        # belowMarks = db.open <= db.getRollingData(60) #and db.getRollingData(3) < db.getRollingData(10)

        num = 0
        # Buy
        if goodBuyShortTerm or goodBuyLongTerm or wellBelowUsual:
            # A number that correlates with how strong the buy will be
            num = ((db.getRollingData(50) / db.getRollingData(4)) - 1) * 100
        # Sell
        elif goodSellLongTerm or goodSellShortTerm:
            # A number that correlates with how strong the sell will be
            num = (db.getRollingData(15) / db.getRollingData(50) - 1) * 100
        return num

    # Stripped down run
    def bare_run (self):
        pass

    def csv_indices(self):
        # Take multiple indices and add them to a csv. This csv can then be used by neural network
        pass

    def peakedRecently(self, db, dataPieces, index, peakLookBack=3):

        if index >= peakLookBack and db.getRollingData(2, 'mean') > db.getRollingData(5, 'mean'):
            for indexOfRecentDays in range(0, peakLookBack):
                if dataPieces[index - indexOfRecentDays].peaked(80, dataPieces, minWindow=20, maxWindow=40):
                    return True
        return False
    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False



if __name__ == "__main__":
    A = Analysis("MSI")
    A.run()