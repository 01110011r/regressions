import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Datasetni yaratish
np.random.seed(42)
# Xususiyatlar
gas_consumation = np.random.uniform(100, 1000, 20)
oil_price = np.random.uniform(50, 100, 20)
temperature = np.random.uniform(-10,35, 20)
#bog'liq o'zgaruvchilar
gas_price = 20 + 0.5 * gas_consumation + 2 * oil_price - 0.3 * temperature
gas_price += np.random.normal(0, 10, 20)
#datasetni DataFrame shaklida saqlash
data = pd.DataFrame({
    'Gas Consumption (m^3)': gas_consumation,
    'Oil price ($)': oil_price,
    'Temperature (C)': temperature,
    'Gas price ($)': gas_price
})

print(data.head())

# 2. Bir o'zgaruvchili regresiya modeli quramiz
X = data[['Gas Consumption (m^3)']].values #mustaqil o'zgaruvchi
Y = data[['Gas price ($)']].values #bogliq o'zgaruchi
degrees = [1, 3, 6] #polinom darajalari
#grafik tayyoralsh
plt.scatter(X, Y, color='gray', label='Real data') # asl malumotlar
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1) #x oralig'i
#polinom darajalari uchun regresiya modellari
for degree in degrees:
    #polinom xususiyatlarini yaratish
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    #modelni o'qitish
    model = LinearRegression()
    model.fit(X_poly, Y)
    #bashorat qilish
    y_pred = model.predict(poly.fit_transform(x_range))
    #grafik chizish
    plt.plot(x_range, y_pred, label=f'Degree {degree}')
#grafikni sozlash
plt.xlabel('Gas Consumption (m^3)')
plt.ylabel('Gas price ($)')
plt.title('Polynomial regression model (1, 3, 6 degress)')
plt.legend()
plt.show()

# 3. ko'p o'zgaruvchili regresiya modeli quramiz
X_multi = data[['Gas Consumption (m^3)', 'Oil price ($)', 'Temperature (C)']].values #mustaqil o'zgaruvchilar
Y_multi = data[['Gas price ($)']].values #bog'liq ozgaruvchi
degrees = [2, 5] #polinom darajalari
results = {} # natijalar
#polinom darajalariga regresiya modellari
for degree in degrees:
    poly = PolynomialFeatures(degree) #polinom xususiyatlarini yaratish
    X_poly_multi = poly.fit_transform(X_multi)
    #o'qitish
    model = LinearRegression()
    model.fit(X_poly_multi, Y_multi)
    #basorat qilish
    y_pred_multi = model.predict(X_poly_multi)
    #metrikalarni hisoblash
    mse = mean_squared_error(Y_multi, y_pred_multi)
    r2 = r2_score(Y_multi, y_pred_multi)
    results[degree] = {'MSE': mse, 'R2': r2}
    print(f'Degree: {degree}, MSE: {mse}, R2: {r2}')
# matijalarni ko'rish
for degree, metrics in results.items():
    print(f'Plinomial degree: {degree}, MSE = {metrics["MSE"]:.2f}, R2 = {metrics["R2"]:.2f}')

# 4. modellarini testlash
print("Testing.........\nBir o'zgaruvchili regresiya natijalari:")
for degree in [1, 3, 6]:
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    #modelni o'qitish
    model = LinearRegression()
    model.fit(X_poly, Y)
    #bashorat qilish
    y_pred = model.predict(X_poly)
    #metrikalar
    mse = mean_squared_error(Y, y_pred)
    r2 = r2_score(Y, y_pred)
    print(f"polinomial degree: {degree}, MSE = {mse:.2f}, R2 = {r2:.2f}")

print("\nKo'p o'zgaruvchili regresiya natijalari:")
for degree in [2, 5]:
    poly = PolynomialFeatures(degree)
    X_poly_multi = poly.fit_transform(X_multi)
    #o'qitish
    model = LinearRegression()
    model.fit(X_poly_multi, Y_multi)
    #predict
    y_pred_multi = model.predict(X_poly_multi)
    #metrikalar
    mse_multi = mean_squared_error(Y_multi, y_pred_multi)
    r2_multi = r2_score(Y_multi, y_pred_multi)
    print(f"polinomial daraja {degree}: MSE = {mse_multi:.2f}, R2 = {r2_multi:.2f}")