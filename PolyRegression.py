import matplotlib as plt
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
import  pandas as p

degrees = [2, 3, 4, 5]
# data = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\MSITrain.csv")
# cvData = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\MSI_CV.csv")
data = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\CDETrain.csv")
cvData = p.read_csv("C:\\Users\\cole\\Desktop\\Brks\\CDE_CV.csv")

# TODO Make mathod that takes in csv, and do this automatically
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
    xVectors.append([X1[i],X2[i],X3[i],X4[i],X5[i],X6[i],X7[i],X8[i],X9[i],X10[i],X11[i]])

yVectors = []
for i in range(0, len(X1)):
    yVectors.append([Y1[i],Y2[i]])

cv_xVectors = []
for i in range(0, len(cvX1)):
    cv_xVectors.append([cvX1[i],cvX2[i],cvX3[i],cvX4[i],cvX5[i],cvX6[i],cvX7[i],cvX8[i],cvX9[i],cvX10[i],cvX11[i]])

cv_yVectors = []
for i in range(0, len(cvX1)):
    cv_yVectors.append([cvY1[i],cvY2[i]])

degrees = [1,2,3]
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
    length = len(cv_yVectors)
    # bought = 0
    # sold = 0
    for x in range(0, length):
        current = cv_xVectors[x][0]
        guess = pipeline.predict(cv_xVectors[x])
        actual = cv_yVectors[x]
        # if(guess[0][0] > current * 1.01):
        #     bought += actual[0] - current
        # if(guess[0][0] < current *.99):
        #     sold += current - actual[0]
        e = (guess - actual) ** 2
        avgE += e
    avgE /= length
    print(avgE)
    # print(bought)
    # print(sold)
    print(pipeline.predict([8.87,15.7334716101,1,1,0,-0.3208,-0.0628909090909,0.00373984962406,-0.0458340525328,-0.0140111404126,0.0499829469047]))


    print("------------------------------------------------")