import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
mtcars = pd.read_csv("mtcars.csv")
X = pd.DataFrame(mtcars["wt"])
Y = pd.DataFrame(mtcars["mpg"])

reg = LinearRegression().fit(X, Y)

k = reg.coef_
b = reg.intercept_

print("k =",k[0][0] ,"b =", b[0])

plt.figure()
plt.scatter(X, Y)
plt.plot(X, reg.predict(X), color="r")
plt.xlabel("Car weight (wt)")
plt.ylabel("Car mpg (mpg)")
plt.title("Predicted vehicle gas mileage (mpg) in relation to weight (wt)")
plt.show()