import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import  pandas as p

degrees = [2, 3, 4, 5]
data = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\MSITrain.csv")
cvData = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\MSI_CV.csv")

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

cvX1 = [x for x in cvData['x1'].values]
cvX2 = [x for x in cvData['x2'].values]
cvX3 = [x for x in cvData['x3'].values]
cvX4 = [x for x in cvData['x4'].values]
cvX5 = [x for x in cvData['x5'].values]
cvX6 = [x for x in cvData['x6'].values]
cvX7 = [x for x in cvData['x7'].values]
cvX8 = [x for x in cvData['x8'].values]
cvX9 = [x for x in cvData['x9'].values]
cvX10 = [x for x in cvData['x10'].values]
cvX11 = [x for x in cvData['x11'].values]

cvY1 = [x for x in cvData['y1'].values]
cvY2 = [x for x in cvData['y2'].values]

xVectors = []
for i in range(0, len(X1)):
    xVectors.append([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11])

yVectors = []
for i in range(0, len(X1)):
    yVectors.append([Y1,Y2])

cv_xVectors = []
for i in range(0, len(cvX1)):
    cv_xVectors.append([cvX1,cvX2,cvX3,cvX4,cvX5,cvX6,cvX7,cvX8,cvX9,cvX10,cvX11])

cv_yVectors = []
for i in range(0, len(X1)):
    cv_yVectors.append([cvY1,cvY2])

degrees = [2]
for i in range(len(degrees)):


    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(xVectors, yVectors)

    print("------------------------------------------------")
    print(degrees[i])
    avgE = 0.0
    length = len()
    for x in range(0, length):
        e = (pipeline.predict(cv_xVectors) - cv_yVectors) ** 2
        avgE += e
    avgE /= length
    print(avgE)
    print("------------------------------------------------")