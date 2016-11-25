from datetime import datetime

from Analyzer import Analyzer
from PreProcessor import PreProcessor
import threading

# Used to process multiple stocks at once, and make a model out of multiple stocks at once. Tried to multi thread it,
# But I could not figure it out
class MultiStockPreProcessor:


    @staticmethod
    def PreProcessCompanies(companies, start=datetime(2011,1,1),end =datetime.today()):
        threads = []
        for company in companies:
            print("Processing:", company)
            MultiStockPreProcessor.preProcessCompany(company,start,end)

    @staticmethod
    def preProcessCompany(company, start,end):
        PP = PreProcessor(company, start=start, end=end)
        PP.csv_indices()

    @staticmethod
    def GetAnalyzerForCompanies(companies, xColumns, yColumns, valColumn, cvPercent=.1, cvSelection='top', degree=1,valcolumnstart=datetime(2014, 1, 1), end=datetime.today()):
        csvFiles = []
        for company in companies:
            csvFiles.append("Processed_csv\\"+company+"_processed.csv")
        return Analyzer(csvFiles=csvFiles,xColumns=xColumns,yColumns=yColumns,cvPercent=cvPercent,cvSelection=cvSelection,valColumn=valColumn)

if __name__ == "__main__":
    file = open('company_lists\\BigTechs', 'r')

    companies = []
    for line in file:
        companies.append(line.strip('\n'))
    # MultiStockPreProcessor.PreProcessCompanies(companies)
    model = MultiStockPreProcessor.GetAnalyzerForCompanies(companies,
                                                           xColumns=['open','percent300Avg', 'growth5', 'growth10', 'growth20','growth40','growth80','growth160','growth300',
                   'num', 'x3', 'x4', 'x5', 'slope5', 'slope10', 'slope20', 'slope40', 'slope80',
                   'slope160','movAvgDiff20','movAvgDiff40', 'movAvgDiff80','movAvgDiff160','var20','var40','var80','var160'],
                                                           yColumns=['5DayActual','10DayActual','50DayActual'], valColumn='open', cvPercent=0, cvSelection='top')

    model.run(degrees=[1, 2]
          ,rowToPredict='last')

