from pandas.io.common import urlencode

class GoogleStockReader():

    @staticmethod
    def getStockUrl(sym, start, end):
        _HISTORICAL_GOOGLE_URL = 'http://www.google.com/finance/historical?'
        url = "%s%s" % (_HISTORICAL_GOOGLE_URL,
                        urlencode({"q": sym,
                                   "startdate": start.strftime('%b %d, ' '%Y'),
                                   "enddate": end.strftime('%b %d, %Y'),
                                   "output": "csv"}))
        return url
