from pandas.io.common import urlencode

class StockReader():

    @staticmethod
    def getStockUrlGoogle(sym, start, end):
        _HISTORICAL_GOOGLE_URL = 'http://www.google.com/finance/historical?'
        url = "%s%s" % (_HISTORICAL_GOOGLE_URL,
                        urlencode({"q": sym,
                                   "startdate": start.strftime('%b %d, ' '%Y'),
                                   "enddate": end.strftime('%b %d, %Y'),
                                   "output": "csv"}))
        return url

    @staticmethod
    def getStockUrlYahoo(sym, start, end, interval='d'):
        _HISTORICAL_YAHOO_URL = 'http://ichart.finance.yahoo.com/table.csv?'
        url = (_HISTORICAL_YAHOO_URL + 's=%s' % sym +
               '&a=%s' % (start.month - 1) +
               '&b=%s' % start.day +
               '&c=%s' % start.year +
               '&d=%s' % (end.month - 1) +
               '&e=%s' % end.day +
               '&f=%s' % end.year +
               '&g=%s' % interval +
               '&ignore=.csv')
        return url

    # TODO make one from http://quotes.wsj.com/index/DJIA/historical-prices, it actually has data I want
    @staticmethod
    def getStockUrlWSJ(sym,start,end):
        _HISTORICAL_WSJ_URL1 = "http://quotes.wsj.com/index/"
        _HISTORICAL_WSJ_URL2 = "/historical-prices/download?MOD_VIEW=page"
        numDays = (end - start).days
        startString = start.strftime("%m/%d/%y")
        endString = end.strftime("%m/%d/%y")
        url = (_HISTORICAL_WSJ_URL1 + sym + _HISTORICAL_WSJ_URL2 +
               '&num_rows=%s' % numDays +
               '&range_rows=%s' % numDays +
               '&startDate=%s' % startString +
               '&endDate=%s' % endString +
               ' HTTP/1.1')
        print(url)
        return url