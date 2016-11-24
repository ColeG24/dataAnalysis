import math
import numbers
import random
import warnings

import numpy as np
import  pandas as p

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline


# Analyzes pre processed data, and can run linear/polynomial regression and gradient descent on the data
class Analyzer:

    def run(self, degrees, rowToPredict):

        for i in range(len(degrees)):
            polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                     include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                                 ("linear_regression", linear_regression)])
            pipeline.fit(self.xVectors, self.yVectors)

            if (rowToPredict == 'last'):
                rowToPredict = self.lastRow

            print("------------------------------------------------")
            print("degree:", degrees[i])
            avgSquaredE = 0.0
            avgError = 0.0
            length = len(self.yVectors_cv)

            # If there is any cross validation data
            if (length != 0):
                bought = 0
                sold = 0
                for x in range(0, length):
                    current = self.xVectors_cv[x][0]
                    guess = pipeline.predict(self.xVectors_cv[x])
                    actual = self.yVectors_cv[x]
                    if(guess[0][0] > current * 1.02):
                        bought += actual[0] - current
                    if(guess[0][0] < current *.98):
                        sold += current - actual[0]
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
                print("bought", bought)
                print("sold", sold)
            prediction = pipeline.predict(rowToPredict)
            print("Prediction:", prediction)
            predictionMoney = prediction * self.lastRowMovAvg
            print("Prediction as money", predictionMoney)

            print("------------------------------------------------")

    def __init__(self, csvFile, xColumns, yColumns, cvPercent=.2, cvSelection = 'random', valColumn='k'):
        self.data = p.read_csv(csvFile)

        length = len(self.data[xColumns[0]].values)

        self.xVectors = []
        self.yVectors = []
        self.valColumns = []
        self.xVectors_cv = []
        self.yVectors_cv = []
        self.valColumns_cv = []

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
            if (i == length - 1):
                self.lastRow = xRow
                self.lastRowMovAvg = self.data[valColumn].values[i]

            # Skip invalid rows
            if skipRow:
                continue

            # Determine if it should be a cross validate value or not
            r = random.random()
            useForCV = (r < cvPercent and cvSelection == 'random') or (i > (1-cvPercent) * length and cvSelection == 'top')

            if not useForCV:
                self.xVectors.append(xRow)
                self.yVectors.append(yRow)
            else:
                self.xVectors_cv.append(xRow)
                self.yVectors_cv.append(yRow)

    def grad_desc(self, rowToPredict):
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(alpha=0.001, average=False, class_weight=None, epsilon=0.1,
                      eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                      learning_rate='optimal', loss='hinge', n_iter=1000, n_jobs=1,
                      penalty='l2', power_t=0.5, random_state=None, shuffle=False,
                      verbose=0, warm_start=False)
        Ys = []
        # X_train = self.data.drop('y1', axis=1)
        Y_train = np.asarray(self.data['y1'], dtype="|S6")
        for vector in self.yVectors:
            Ys.append(vector[0])
        clf.fit(self.xVectors, np.asarray(Ys, dtype="|S6"))
        avgSquaredE = 0.0
        avgError = 0.0
        if (rowToPredict == 'last'):
            rowToPredict = self.lastRow

        length = len(self.yVectors_cv)
        for x in range(0, length):
            current = self.xVectors_cv[x][0]
            guess = float(clf.predict(self.xVectors_cv[x])[0])
            actual = self.yVectors_cv[x][0]
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
        prediction = clf.predict(rowToPredict)
        print("Prediction",prediction)
        # predictionMoney = prediction*self.lastRowMovAvg
        # print("Prediction as money",predictionMoney)





if __name__ == "__main__":
    row = [0.895984320557,0,0,0,0,-0.1374,0.316993939394,0.296665037594,-0.00738881332082,0.0655642595523,0.0165954231916,1.02211227209,1.05718063239,1.05766414952,1.09807109928]
    P = Analyzer(csvFile="Processed_csv\\X_Processed.csv",
                 xColumns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12' ,'x13' , 'x14', 'x15','x16','x17','x18','x19'],
                 yColumns=['y1', 'y2', 'y3'], valColumn='x0',cvPercent=.2, cvSelection='top')
    P.run(degrees=[1, 2, 3]
          ,rowToPredict='last')
    # data = PolyRegression(csvFile="D4.csv", xColumns=['x1','x2','x3'], yColumns=['y1'], cvPercent=0)
    P.grad_desc(rowToPredict='last')

