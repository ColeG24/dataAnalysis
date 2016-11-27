import math
import numbers
import random
import warnings

import numpy as np
import pandas as p

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

# Analyzes pre processed data, and can run linear/polynomial regression and gradient descent on the data
class Analyzer:


    def make_model(self, degree):
        polynomial_features = PolynomialFeatures(degree=degree,
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(self.xVectors, self.yVectors)
        return pipeline

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
            print("Cross validating with %d entries" % length)
            # If there is any cross validation data
            if (length != 0):
                bought5D = 0
                sold5D = 0
                avg5DErrorAbove = 0
                above5DCount = 0
                avg5DErrorBelow = 0
                below5DCount = 0

                bought10D = 0
                sold10D = 0
                avg10DErrorAbove = 0
                above10DCount = 0
                avg10DErrorBelow = 0
                below10DCount = 0

                bought50D = 0
                sold50D = 0
                avg50DErrorAbove = 0
                above50DCount = 0
                avg50DErrorBelow = 0
                below50DCount = 0

                for x in range(0, length):
                    current = self.xVectors_cv[x][0]
                    guess = pipeline.predict(self.xVectors_cv[x])
                    actual = self.yVectors_cv[x]
                    if(guess[0][0] > 1.02):
                        bought5D += actual[0] - 1
                    if(guess[0][0] < .98):
                        sold5D += 1 - actual[0]

                    if (guess[0][1] > 1.04):
                        bought10D += actual[1] - 1
                    if (guess[0][1] < .96):
                        sold10D += 1 - actual[1]

                    if (guess[0][2] > 1.06):
                        bought50D += actual[2] - 1
                    if (guess[0][2] < .94):
                        sold50D += 1 - actual[2]

                    error = guess - actual
                    squaredE = error ** 2
                    avgSquaredE += squaredE
                    avgError += abs(error)

                    # See error above and below for 5 days
                    error5D = error[0][0]
                    if error5D > 0:
                        avg5DErrorAbove += error5D
                        above5DCount += 1
                    else:
                        avg5DErrorBelow += error5D
                        below5DCount += 1

                    # See error above and below for 10 days
                    error10D = error[0][1]
                    if error10D > 0:
                        avg10DErrorAbove += error10D
                        above10DCount += 1
                    else:
                        avg10DErrorBelow += error10D
                        below10DCount += 1

                    # See error above and below for 50 days
                    error50D = error[0][2]
                    if error50D > 0:
                        avg50DErrorAbove += error50D
                        above50DCount += 1
                    else:
                        avg50DErrorBelow += error50D
                        below50DCount += 1


                avgSquaredE /= length
                avgError /= length
                if above5DCount > 0:
                    avg5DErrorAbove /= above5DCount
                if below5DCount > 0:
                    avg5DErrorBelow /= below5DCount
                if above10DCount > 0:
                    avg10DErrorAbove /= above10DCount
                if below10DCount > 0:
                    avg10DErrorBelow /= below10DCount
                if above50DCount > 0:
                    avg50DErrorAbove /= above50DCount
                if below50DCount > 0:
                    avg50DErrorBelow /= below50DCount

                print("Avg Squared Error:")
                print(avgSquaredE)
                print("Avg Error:", avgError)
                print("")
                print("Avg Above Error for 5 days:", avg5DErrorAbove, 'with %d guesses above' % above5DCount)
                print("Avg Below Error for 5 days:", avg5DErrorBelow, 'with %d guesses below' % below5DCount)

                print("Avg Above Error for 10 days:", avg10DErrorAbove, 'with %d guesses above' % above10DCount)
                print("Avg Below Error for 10 days:", avg10DErrorBelow, 'with %d guesses below' % below10DCount)

                print("Avg Above Error for 50 days:", avg50DErrorAbove, 'with %d guesses above' % above50DCount)
                print("Avg Below Error for 50 days:", avg50DErrorBelow, 'with %d guesses below' % below50DCount)
                print("")
                print("Gain\loss when prediction is above 2% for 5 days", bought5D)
                print("Gain\loss when prediction is below 2% for 5 days", sold5D)

                print("Gain\loss when prediction is above 4% for 10 days", bought10D)
                print("Gain\loss when prediction is below 4% for 10 days", sold10D)

                print("Gain\loss when prediction is above 6% for 50 days", bought50D)
                print("Gain\loss when prediction is above 6% for 50 days", sold50D)

            if rowToPredict == 'last':
                for csvFile in self.csvFiles:
                    prediction = pipeline.predict(self.lastRows[csvFile])
                    print("")
                    print("Predicting last row of ", csvFile)
                    print("Prediction:", prediction)
                    predictionMoney = prediction * self.lastRowMovAvg[csvFile]
                    print("Prediction as money", predictionMoney)
                    print("")


            print("------------------------------------------------")

    def __init__(self, csvFiles, xColumns, yColumns, cvPercent=.2, cvSelection ='random', valColumn='k'):
        self.csvFiles = csvFiles
        datas = []
        # Longest csv
        longestLength = 0;
        for csvFile in csvFiles:
            data = p.read_csv(csvFile)
            datas.append(data)
            longestLength = len(data) if len(data) > longestLength else longestLength

        self.xVectors = []
        self.yVectors = []
        self.valColumns = []
        self.xVectors_cv = []
        self.yVectors_cv = []
        self.valColumns_cv = []
        self.lastRows = {}
        self.lastRowMovAvg = {}

        # For each row
        for i in range(0, longestLength):
            csvFileIndex = 0
            for data in datas:
                csvFileName = csvFiles[csvFileIndex]
                csvFileIndex += 1
                csvLength = len(data)
                if csvLength-1 < i:  # if this csv is shorter then the index
                    continue

                skipRow = False
                # Create x and y row
                xRow = []
                for xColumn in xColumns:
                    val = data[xColumn].values[i]
                    xRow.append(val)

                    if not isinstance(val, numbers.Real) or math.isnan(val):
                        skipRow = True
                yRow = []
                for yColumn in yColumns:
                    val = data[yColumn].values[i]
                    yRow.append(val)
                    if not isinstance(val, numbers.Real) or math.isnan(val):
                        skipRow = True

                # If last row of csv
                if i == csvLength - 1:
                    self.lastRows[csvFileName] = xRow
                    self.lastRowMovAvg[csvFileName] = data[valColumn].values[i]

                # Skip invalid rows
                if skipRow:
                    continue

                # Determine if it should be a cross validate value or not
                r = random.random()
                useForCV = (r < cvPercent and cvSelection == 'random') or \
                           (i > (1-cvPercent) * longestLength and cvSelection == 'top')

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
    P = Analyzer(csvFiles=["Processed_csv\\X_Processed.csv"],
                 xColumns=['x0','x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12' ,'x13' , 'x14', 'x15','x16','x17','x18','x19'],
                 yColumns=['y1', 'y2', 'y3'], valColumn='x0', cvPercent=.2, cvSelection='top')
    P.run(degrees=[1, 2, 3]
          ,rowToPredict='last')
    # data = PolyRegression(csvFile="D4.csv", xColumns=['x1','x2','x3'], yColumns=['y1'], cvPercent=0)
    P.grad_desc(rowToPredict='last')

