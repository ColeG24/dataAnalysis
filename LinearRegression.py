import pandas as p
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

if __name__ == "__main__":


    data = p.read_csv("C:\\Users\\cole\\Desktop\\hw3\\D3_minus15.csv")
    cvData = p.read_csv("C:\\Users\\cole\\Desktop\\hw3\\crossValidationData.csv")

    X1 = [x for x in data['x'].values]
    X2 = [x for x in data['x1'].values]
    X3 = [x for x in data['x2'].values]
    Y = [x for x in data['y'].values]

    cvX1 = [[x] for x in data['x'].values]
    cvX2 = [[x] for x in data['x1'].values]
    cvX3 = [[x] for x in data['x2'].values]
    cvY = [[x] for x in data['y'].values]


    class T:
        pass


    rows = []

    for i in range(0, len(X1)):
        t = T()
        t.y = Y[i]
        t.x1 = X1[i]
        t.x2 = X2[i]
        t.x3 = X3[i]
        rows.append(t)

    cvRows = []

    for i in range(0, len(X1)):
        t = T()
        t.y = Y[i]
        t.x1 = X1[i]
        t.x2 = X2[i]
        t.x3 = X3[i]
        cvRows.append(t)
    X = [[x] for x in data['x'].values]

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    # regr.fit([[getattr(t, 'x%d' % i) for i in range(1, 4)] for t in rows], [t.y for t in rows])
    # avgE = 0
    # length = len(cvRows)
    avgE = 0.0
    length = len(cvX1)
    for x in range(0, length):
        e = (regr.predict(cvX1[x]) - cvY[x]) ** 2
        avgE += e
    avgE /= length
    # for x in range(0, length):
    #     a = cvRows[i]
    #     e = (regr.predict([a.x1, a.x2, a.x3]) - a.y) ** 2
    #     avgE += e
    # avgE /= length
    print(avgE)

    # print(X2)
    # print(X3)





    # avgE = 0.0
    # length = len(cvX)
    # for x in range(0, length):
    #     e =(regr.predict(cvX[x])-cvY[x])**2
    #     avgE += e
    # avgE /= length

    #print(avgE)
    # print(poly(1))
    # print(poly.predict(2))
    # print(poly.predict(3))


    # plt.scatter(X, Y, color='black')
    # plt.plot(X, regr.predict(X), color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

    # degrees = [2, 3, 4, 5]
    # plt.figure(figsize=(14, 5))
    # for i in range(len(degrees)):
    #     ax = plt.subplot(1, len(degrees), i + 1)
    #     plt.setp(ax, xticks=(), yticks=())
    #
    #     polynomial_features = PolynomialFeatures(degree=degrees[i],
    #                                              include_bias=False)
    #     linear_regression = LinearRegression()
    #     pipeline = Pipeline([("polynomial_features", polynomial_features),
    #                          ("linear_regression", linear_regression)])
    #     pipeline.fit(X, Y)
    #
    #     print("------------------------------------------------")
    #     print(degrees[i])
    #     avgE = 0.0
    #     length = len(cvX)
    #     for x in range(0, length):
    #         e = (pipeline.predict(cvX[x]) - cvY[x]) ** 2
    #         avgE += e
    #     avgE /= length
    #     print(avgE)
    #     print("------------------------------------------------")

        # X_test = np.linspace(0, 4, 1000)
        # plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        # plt.scatter(X, Y, label="Samples")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.xlim((0, 4))
        # plt.ylim((-5,5))
        # plt.legend(loc="best")
    # plt.show()

    # np.random.seed(0)
    #
    # n_samples = 30
    # degrees = [1, 4, 15]
    #
    # true_fun = lambda X: np.cos(1.5 * np.pi * X)
    # X = np.sort(np.random.rand(n_samples))
    # y = true_fun(X) + np.random.randn(n_samples) * 0.1
    #
    # plt.figure(figsize=(14, 5))
    # for i in range(len(degrees)):
    #     ax = plt.subplot(1, len(degrees), i + 1)
    #     plt.setp(ax, xticks=(), yticks=())
    #
    #     polynomial_features = PolynomialFeatures(degree=degrees[i],
    #                                              include_bias=False)
    #     linear_regression = LinearRegression()
    #     pipeline = Pipeline([("polynomial_features", polynomial_features),
    #                          ("linear_regression", linear_regression)])
    #     pipeline.fit(X[:, np.newaxis], y)
    #
    #     # Evaluate the models using crossvalidation
    #
    #     X_test = np.linspace(0, 1, 100)
    #     plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    #     plt.plot(X_test, true_fun(X_test), label="True function")
    #     plt.scatter(X, y, label="Samples")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.xlim((0, 1))
    #     plt.ylim((-2, 2))
    #     plt.legend(loc="best")
    #     plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})")
    # plt.show()
