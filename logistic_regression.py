import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Datasetni yaratish
#Tasodifiylikni ta'minlash uchun seed sozlamasi
np.random.seed(42)
# Xususiyatlar
age = np.random.randint(25, 65, 40) #Yosh (25-65)
experience = np.random.randint(1, 40, 40)   #Tajriba (1-40 yil)
publications = np.random.randint(0, 50, 40) #Ilmiy maqolalar soni (0-50)
salary = np.random.randint(500, 1000, 40) #Oylik maosh (500-1000 $)

#Sinflar yaratish
classes = np.random.choice(['Professor', 'Dotsent', "O'qituvchi"], 40)

#datasetni DataFrame shaklida shakillantirish
data = pd.DataFrame({
    'Age': age,
    'Experience': experience,
    'Publications': publications,
    'Salary': salary,
    'Class': classes
})
print(data.head())

# 2. Datasetni grafikda ko'rsatish
classes = data['Class'].unique()
plt.figure(figsize=(8, 6))
for cls in classes:
    subset = data[data['Class'] == cls] #sinfga mos ma'lumotlarni ajratib olamiz
    plt.scatter(subset['Age'], subset['Salary'], label=cls)

    plt.title("O'qituvchilarning yoshi va oylik maoshi")
    plt.xlabel("Age (Yosh)")
    plt.ylabel("Salary ($)")
    plt.legend(title="Class (Toifa)")
    plt.grid(True)
    plt.show()

#3. Tasniflash (classification) modeli qurish
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#mustaqil o'zgaruvchilar (xususiyatlar)
X = data[['Age', 'Experience', 'Publications', 'Salary']].values
y= data['Class'].values #bog'liq o'zgaruvchi
#datasetni train va test qismga ajratib olamiz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#random forest modelni qurish
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
#predikt qilish
y_pred = model.predict(X_test)
#aniqlilikni baholash
accuracy = accuracy_score(y_test, y_pred)
print(f"Model aniqligi(accuarcy): {accuracy:.2f}")

#4. sklearn bilan logistik regressiya
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

#5. Logistik regressiya modelini baholash (o'rgatuvchi to'plam uchun)
y_pred_train_log = log_model.predict(X_test)#predikt qilish
accuracy_train_log = accuracy_score(y_test, y_pred_train_log)#train to'plam uchun aniqlilikni baholash
print(f"Logistik regressiya modeli aniqligi(train): {accuracy_train_log:.2f}")

#6. Logistik regressiya modelini baholash (test to'plam uchun)
y_pred_test_log = log_model.predict(X_test)#predikt qilish
accuracy_test_log = accuracy_score(y_test, y_pred_test_log)#test to'plam uchun aniqlilikni baholash

print(f"Logistik regressiya modeli aniqligi(test): {accuracy_test_log:.2f}")

#7. sinconfusion matrix (xatolik matrisi) test to'plami uchun
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

cm=confusion_matrix(y_test, y_pred_test_log, labels=['Professor', 'Dotsent', 'O‘qituvchi'])#xatolik matrisini hisoblash
print('Confusion matrix:')
print(cm)
#matrisni vizualizatsiya qilish
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Professor', 'Dotsent', 'O‘qituvchi'], yticklabels=['Professor', 'Dotsent', 'O‘qituvchi'])
plt.title('Confusion matrix for test data (Logistic regression)')
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()