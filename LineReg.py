import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_coef(x, y):

    n = np.size(x)

    m_x, m_y = np.mean(x), np.mean(y)

    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x


    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)

def plot_regression_line(x, y, b):

    plt.scatter(x, y, color = "y",
               marker = "o", s = 30)

    y_pred = b[0] + b[1]*x

    plt.plot(x, y_pred, color = "g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


data = pd.read_csv('Food-Truck-LineReg.csv')

x = np.array(data)[:,0]
y = np.array(data)[:,1]

b = estimate_coef(x, y)
print("Estimated coefficients:")
print("b0 = ", b[0], "b1 = ", b[1])


plot_regression_line(x, y, b)
