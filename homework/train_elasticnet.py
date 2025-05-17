#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# importacion de librerias
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from homework.src.internals.calculate_metrics import calculate_metrics
from homework.src.internals.save_model_if_better import save_model_if_better

# descarga de datos
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

df.to_csv("data/winequality-red.csv")

# preparacion de datos
y = df["quality"]
x = df.copy()
x.pop("quality")

# dividir los datos en entrenamiento y testing
(x_train, x_test, y_train, y_test) = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=123456,
)

# entrenar el modelo
estimator = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=12345)
estimator.fit(x_train, y_train)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)

print()
print("Metricas de entrenamiento:")
print(f"  MSE: {mse}")
print(f"  MAE: {mae}")
print(f"  R2: {r2}")

# Metricas de error durante testing
print()
print("Metricas de testing:")
mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)

print(f"  MSE: {mse}")
print(f"  MAE: {mae}")
print(f"  R2: {r2}")

save_model_if_better(estimator, x_test, y_test)
