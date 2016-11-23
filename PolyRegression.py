import matplotlib as plt
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
import  pandas as p
import random
import numbers
import math

class PolyRegression:

    def run(self, degrees, rowToPredict):

        for i in range(len(degrees)):
            polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                     include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                                 ("linear_regression", linear_regression)])
            pipeline.fit(self.xVectors, self.yVectors)

            print("------------------------------------------------")
            print("degree:", degrees[i])
            avgSquaredE = 0.0
            avgError = 0.0
            length = len(self.yVectors_cv)
            bought = 0
            sold = 0
            for x in range(0, length):
                current = self.xVectors_cv[x][0]
                guess = pipeline.predict(self.xVectors_cv[x])
                actual = self.yVectors_cv[x]
                # if(guess[0][0] > current * 1.01):
                #     bought += actual[0] - current
                # if(guess[0][0] < current *.99):
                #     sold += current - actual[0]
                error = abs(guess - actual)
                squaredE = error ** 2
                avgSquaredE += squaredE
                avgError += error
            avgSquaredE /= length
            avgError /= length
            print("Avg Squared Error:")
            print(avgSquaredE)
            print("Avg Error:")
            print(avgError)
            # print(bought)
            # print(sold)
            print("Prediction:")
            print(pipeline.predict(rowToPredict))
            print("------------------------------------------------")

    def __init__(self, csvFile, xColumns, yColumns, cvPercent=.2):
        self.data = p.read_csv(csvFile)

        length = len(self.data[xColumns[0]].values)

        self.xVectors = []
        self.yVectors = []
        self.xVectors_cv = []
        self.yVectors_cv = []

        # For each row
        for i in range(0, length):

            skipRow = False
            # Create x and y row
            xRow = []
            for xColumn in xColumns:
                val = self.data[xColumn].values[i]
                xRow.append(val)

                if not isinstance(val, numbers.Real) or math.isnan(val):
                    skipRow = True
            yRow = []
            for yColumn in yColumns:
                val = self.data[yColumn].values[i]
                yRow.append(val)
                if not isinstance(val, numbers.Real) or math.isnan(val):
                    skipRow = True

            # Skip invalid rows
            if skipRow:
                continue

            # Determine if it should be a cross validate value or not
            r = random.random()
            if r >= cvPercent:
                self.xVectors.append(xRow)
                self.yVectors.append(yRow)
            else:
                self.xVectors_cv.append(xRow)
                self.yVectors_cv.append(yRow)

if __name__ == "__main__":
    P = PolyRegression(csvFile="C:\\Users\\cole\\Desktop\\Brks\\PFPT_Processed.csv",
                       xColumns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12' ,'x13' , 'x14', 'x15'],
                       yColumns=['y1', 'y2', 'y3'])
    P.run(degrees=[1, 2, 3]
          ,rowToPredict=[82.33,0,0,0,0,0.7994,0.368612121212,0.199587593985,-0.048517565666,0.0734299314346,0.13835385333,1.07334689195,1.10019410081,1.08749995872,1.13834374649])