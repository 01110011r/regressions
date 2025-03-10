import numpy as np
import pandas as pd

# Dataset yaratish
np.random.seed(42)
data = {
    "Maydon": np.random.randint(50, 200, 100),  # Kvadrat metrda uy maydoni
    "Xonalar_soni": np.random.randint(1, 5, 100),  # Xonalar soni
    "Narx": np.random.randint(50, 500, 100)  # Ming $ da narx
}

df = pd.DataFrame(data)
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Xususiyatlar va targetni ajratish
X = df[["Maydon", "Xonalar_soni"]]
y = df["Narx"]

# Ma'lumotlarni trening va test to'plamlariga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xususiyatlarni standartlashtirish
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Model yaratish va o'qitish
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Bashorat qilish
y_pred = model.predict(X_test_scaled)

# Metrikalarni hisoblash
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

import matplotlib.pyplot as plt

# Haqiqiy va bashorat qilingan qiymatlarni solishtirish
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal chiziq
plt.title("Haqiqiy va Bashorat Qilingan Qiymatlar")
plt.xlabel("Haqiqiy Narx (ming $)")
plt.ylabel("Bashorat Qilingan Narx (ming $)")
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ma'lumotlar (x va y qiymatlari)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

# Regressiya modelini yaratish
model = LinearRegression()

# Modelni o'rgatish
model.fit(x, y)

# Yechimlarni bashorat qilish
y_pred = model.predict(x)

# R^2 qiymatini hisoblash
r2 = r2_score(y, y_pred)

# Grafik chizish
plt.scatter(x, y, color='blue', label='Asl ma\'lumotlar')
plt.plot(x, y_pred, color='red', label='Chiziqli regressiya modeli')
plt.xlabel('x (Mustaqil o\'zgaruvchi)')
plt.ylabel('y (Qaram o\'zgaruvchi)')
plt.title('Chiziqli Regressiya')
plt.legend()
plt.grid(True)
plt.show()

# Natijalarni chiqarish
print(f"Modelning R^2 qiymati: {r2}")
print(f"Modelning kesishgan nuqtasi (intercept): {model.intercept_}")
print(f"To\'g\'ri chiziqning egri darajasi (slope): {model.coef_[0]}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y, y_pred)}")

