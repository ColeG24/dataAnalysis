import matplotlib as mpl
mpl.use('pdf')
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sympy import *

class descent:

    def __init__(self):
        self.xmax = 20
        self.ymax = 20
        self.x = Symbol('x')
        self.y = Symbol('y')
        self.f = self.func(self.x, self.y)
        self.df_dx = self.f.diff(self.x)
        self.df_dy = self.f.diff(self.y)

    def func(self, x, y):
        return (1-(y-3))**2 + 20*((x + 3)-(y - 3)**2)**2
        # return (x - 2)**2 + (y - 3)**2
        # return (x - 2)**3 + (y - 3)**3




    def func_grad(self, vx, vy):

        val1 = np.float64(self.df_dx.evalf(64, subs={self.x: vx, self.y: vy}))
        val2 = np.float64(self.df_dy.evalf(64, subs={self.x: vx, self.y: vy}))

        valX = 40*vx-40*((vy-3)**2)+120
        valY = 2*vy-80*(vy-3)*(vx-((vy-3)**2)+3)-8

        # if (abs(val1 - valX) > .00000000001):
        #     print("----------[[]]------[[]]-------!!!!!!!!![[]]!!!!!!!-------[[]]-------")
        #
        # if (abs(val2 - valY) > .00000000001):
        #     print("-------[[]]---------[[]]-------!!!!!!!![[]]!!!!!!!!-------[[]]-------")

        # dfdx = 2*vx - 4
        # dfdy = 2.0*vy - 6
        return np.array([valX, valY])


    def closest_pair(self, points):
        goodPoints = []
        for point in points:
            if abs(point[0]) < self.xmax and abs(point[1]) < self.ymax:
                goodPoints.append(point)
        distx = goodPoints[0][0] - goodPoints[1][0]
        disty = goodPoints[0][1] - goodPoints[1][1]
        minDist = (distx ** 2 + disty ** 2) ** .5
        pt = points[0]
        for x in range(0, len(goodPoints) - 1):
            pointA = goodPoints[x]
            pointB = goodPoints[x + 1]
            distx = pointA[0] - pointB[0]
            disty = pointA[1] - pointB[1]
            dist = (distx**2 + disty**2)**.5
            if dist < minDist:
                minDist = dist
                # if (pointA[0] < pointB[0]):
                #     pt = pointA
                # else:
                #     pt = pointB
                pt = [(pointA[0]+pointB[0])/2, (pointA[1]+pointB[1])/2]
        return pt

    def run(self):
        # prepare for contour plot
        xlist = np.linspace(-5, 5, 26)
        ylist = np.linspace(-5, 5, 26)
        x, y = np.meshgrid(xlist, ylist)
        z = self.func(x, y)
        lev = np.linspace(0, 20, 21)
        # iterate location
        v_init = np.array([0, 0])
        num_iter = 30

        loops = 10
        values = np.zeros([num_iter * loops, 2])
        values[0] = v_init

        for w in range(0, loops):
            v = v_init
            alpha = 0.0032
            # alpha = .5
            # actual gradient descent algorithm

            for i in range(0, num_iter):

                pts = []
                multiplier = 1
                alpha -= .00001
                for t in range(0, i * 10):
                    a = v - alpha * multiplier * self.func_grad(v[0], v[1])
                    multiplier += .01
                    pts.append(a)

                try:
                    v = self.closest_pair(pts)
                    print("new v")
                except IndexError:
                    print("using orignal v")
                index = (w * num_iter) + i;
                print("--------!!-----------")
                print(index)
                print(v)
                print("--------!!-----------")
                values[index, :] = v


            v_init = v
            print(values[(w + 1) * (num_iter - 1)])
            #   plotting
            plt.contour(x, y, z, levels=lev)
            plt.plot(values[:, 0], values[:, 1], 'r-')
            plt.plot(values[:, 0], values[:, 1], 'bo')
            grad_norm = LA.norm(self.func_grad(v[0], v[1]))
            title = "alpha %0.2f | final grad %0.3f" % (alpha, grad_norm)
            plt.title(title)
            file = "C:\\Users\\cole\\Desktop\\hw3\\gdn-%2.0f.pdf" % (alpha * 100)
            plt.savefig(file, bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.cla()

if __name__ == "__main__":
    d = descent()
    d.run()