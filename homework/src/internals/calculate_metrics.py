# Metricas de error durante entrenamiento
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(x, y, model):
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse,mae,r2