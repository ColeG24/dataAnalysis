import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import  pandas as p

degrees = [2, 3, 4, 5]
data = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\MSITrain.csv")
# cvData = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\MSICV.csv")

X1 = [x for x in data['x1'].values]
X2 = [x for x in data['x2'].values]
X3 = [x for x in data['x3'].values]
X4 = [x for x in data['x4'].values]
X5 = [x for x in data['x5'].values]
X6 = [x for x in data['x6'].values]
X7 = [x for x in data['x7'].values]
X8 = [x for x in data['x8'].values]
X9 = [x for x in data['x9'].values]
X10 = [x for x in data['x10'].values]
X11 = [x for x in data['x11'].values]

Y1 = [x for x in data['y1'].values]
Y2 = [x for x in data['y2'].values]


class T:
    pass


xVectors = []

for i in range(0, len(X1)):
    # t = T()
    # t.x1 = X1[i]
    # t.x2 = X2[i]
    # t.x3 = X3[i]
    # t.x4 = X4[i]
    # t.x5 = X5[i]
    # t.x6 = X6[i]
    # t.x7 = X7[i]
    # t.x8 = X8[i]
    # t.x9 = X9[i]
    # t.x10 = X10[i]
    # t.x11 = X11[i]
    xVectors.append(X1)

    # xVectors.append(t)

yVectors = []
for i in range(0, len(X1)):
    # t = T()
    # t.y1 = Y1[i]
    # t.y2 = Y1[i]
    yVectors.append(Y1)
degrees = [2]
for i in range(len(degrees)):



    # cvRows = []
    #
    # for i in range(0, len(X1)):
    #     t = T()
    #     t.y = Y[i]
    #     t.x1 = X1[i]
    #     t.x2 = X2[i]
    #     t.x3 = X3[i]
    #     cvRows.append(t)

    # ax = plt.subplot(1, len(degrees), i + 1)
    # plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(xVectors, yVectors)

    # print("------------------------------------------------")
    # print(degrees[i])
    # avgE = 0.0
    # length = len(cvX)
    # for x in range(0, length):
    #     e = (pipeline.predict(cvX[x]) - cvY[x]) ** 2
    #     avgE += e
    # avgE /= length
    # print(avgE)
    # print("------------------------------------------------")