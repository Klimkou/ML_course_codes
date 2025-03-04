
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Генерация двух классов точек на плоскости
X, y = make_blobs(n_samples=500, centers=2, random_state=42,cluster_std = 3)

# Визуализация
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Класс 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Класс 1')
plt.title('Two blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Обучение линейного классификатора с функцией ошибки MSE
clf_mse = SGDClassifier(loss='squared_error', max_iter=1000, random_state=2)
clf_mse.fit(X, y)

# Предсказания и вычисление ошибки MSE
predictions_mse = clf_mse.predict(X)
mse = mean_squared_error(y, predictions_mse)
print(f"MSE: {mse}")

# Обучение линейного классификатора с функцией ошибки LogLoss
clf_logistic = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
clf_logistic.fit(X, y)

# Предсказания вероятностей и вычисление логистической ошибки
probabilities_logistic = clf_logistic.predict_proba(X)
logistic_loss = log_loss(y, probabilities_logistic)
print(f"LogLoss: {logistic_loss}")

# Визуализация линий разделения классов
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(clf_mse, X, y, "Decision Boundary (MSE)")
plot_decision_boundary(clf_logistic, X, y, "Decision Boundary (Logistic Loss)")

# Confusion matrix
def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()

plot_confusion_matrix(clf_logistic, X, y, "Confusion Matrix (Logistic Loss)")
