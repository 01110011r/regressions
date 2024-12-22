import numpy as np
import pandas as pd
np.random.seed(42)
#xususiyatlar
age = np.random.randint(25, 65, 60)
experience = np.random.randint(1, 40, 60)
publications = np.random.randint(0, 50, 60)
salary = np.random.randint(1000,5000,60)
#classlar
classes = np.random.choice(['Professor', 'Dotsent', "O'qituvchi"], 60)
#datasetni DataFrame shaklida saqlash
data = pd.DataFrame({
    'Age': age,
    'Experience': experience,
    'Publications': publications,
    'Salary': salary,
    'Class': classes
})
print(data)

# 2. Datasetni grafikda ko'rsatish
import matplotlib.pyplot as plt
classes = data['Class'].unique() #sinflar
plt.figure(figsize=(8, 6))
for cls in classes:
    subset = data[data['Class']==cls] #sinfga mos ma'lumotlar
    plt.scatter(subset['Age'], subset['Salary'], label=cls)
plt.title("O'qituvchilarning yoshi va oylik maoshi sinf bo'yicha")
plt.xlabel("Age (Yosh)")
plt.ylabel("Salary ($)")
plt.legend(title="Class (Toifa)")
plt.grid(True)
plt.show()

# 3. Datasetni train va test qismga ajratamiz
from sklearn.model_selection import train_test_split
X = data[['Age', 'Experience', 'Publications', 'Salary']].values # xususiyatlar
y = data['Class'].values #bog'liq o'zgaruvchi (sinflar)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"O'qituvchi to'plam hajmi: {len(X_train)}")
print(f"Sinov to'plam hajmi: {len(X_test)}")

# 4. sinflashtiruvchi modellarni qurish
# 4.1. KNN modeli
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=3) #3 ta yaqin qo'shilganlar
knn.fit(X_train, y_train)

y_train_pred_knn = knn.predict(X_train)
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn) # o'rgatuvchi to'plamiga aniqlilik
print(f"KNN Model Aniqligi (Train): {train_accuracy_knn:.2f}")
y_test_pred_knn = knn.predict(X_test)
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn) #sinov to'plamiga aniqlilik
print(f"KNN Model Aniqligi (Test): {test_accuracy_knn:.2f}")

# 4.2. SVM modeli
from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

y_train_pred_svm = svm.predict(X_train)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm) # o'rgatuvchi to'plamiga aniqlilik
print(f"SVM Model Aniqligi (Train): {train_accuracy_svm:.2f}")
y_test_pred_svm = svm.predict(X_test)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm) #sinov to'plamiga aniqlilik
print(f"SVM Model Aniqligi (Test): {test_accuracy_svm:.2f}")

# 4.3. Decision Tree modeli
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_train_pred_dt = dt.predict(X_train)
train_accuracy_dt = accuracy_score(y_train, y_train_pred_dt) # o'rgatuvchi to'plamiga aniqlilik
print(f"Decision Tree Model Aniqligi (Train): {train_accuracy_dt:.2f}")
y_test_pred_dt = dt.predict(X_test)
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt) #sinov to'plamiga aniqlilik
print(f"Decision Tree Model Aniqligi (Test): {test_accuracy_dt:.2f}")

# 4.4. Random Forest modeli
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf) # o'rgatuvchi to'plamiga aniqlilik
print(f"Random Forest Model Aniqligi (Train): {train_accuracy_rf:.2f}")
y_test_pred_rf = rf.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf) #sinov to'plamiga aniqlilik
print(f"Random Forest Model Aniqligi (Test): {test_accuracy_rf:.2f}")

# 5. confusion matrix (xatolik matrisi) test to'plami uchun
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
class_labels = ['Professor', 'Dotsent', "O'qituvchi"]
# KNN uchun
cm_knn = confusion_matrix(y_test, y_test_pred_knn, labels=class_labels)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_labels)
disp_knn.plot(cmap='Blues')
plt.title("KNN Tartibsizlik Matrisasi")
plt.show()
# SVM uchun
cm_svm = confusion_matrix(y_test, y_test_pred_svm, labels=class_labels)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=class_labels)
disp_svm.plot(cmap='Blues')
plt.title("SVM Tartibsizlik Matrisasi")
plt.show()
# Decision Tree uchun
cm_dt = confusion_matrix(y_test, y_test_pred_dt, labels=class_labels)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=class_labels)
disp_dt.plot(cmap='Blues')
plt.title("Decision Tree Tartibsizlik Matrisasi")
plt.show()
# Random Forest uchun
cm_rf = confusion_matrix(y_test, y_test_pred_rf, labels=class_labels)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_labels)
disp_rf.plot(cmap='Blues')
plt.title("Random Forest Tartibsizlik Matrisasi")
plt.show()


# Natijalarni jamlash
results = {
    "Model": ["KNN", "SVM", "Decision Tree", "Random Forest"],
    "Train Accuracy": [train_accuracy_knn, train_accuracy_svm, train_accuracy_dt, train_accuracy_rf],
    "Test Accuracy": [test_accuracy_knn, test_accuracy_svm, test_accuracy_dt, test_accuracy_rf]
}

# Jadval shaklida natijalarni ko'rsatish
results_df = pd.DataFrame(results)
print(results_df)

# Grafikni sozlash
plt.figure(figsize=(10, 6))
x = results_df["Model"]
plt.bar(x, results_df["Train Accuracy"], color='blue', alpha=0.6, label="Train Accuracy")
plt.bar(x, results_df["Test Accuracy"], color='orange', alpha=0.6, label="Test Accuracy")

# Grafikga sozlamalar qo'shish
plt.title("Modellarning o'rgatuvchi va test to'plamdagi aniqligi")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()