"""
Activity 1.11: Linear Regression - Medical Cost Personal dataset
Objetivo: Predecir "charges" con loss <= 19,000,000
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar datos
def main():
    df = pd.read_csv("Insurance.csv")

    X = df.drop("charges", axis=1)
    y = df["charges"]
    print("Dataset shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nX shape:", X.shape)
    print("y shape:", y.shape)


# =============================================================================
#  2: Preprocesamiento
# - Convertir sex, smoker, region a números con LabelEncoder
# - Feature Engineering para mejorar predicciones, escalamos todos los valores menos el precio
# =============================================================================
    categorical_columns = ["sex", "smoker", "region"]
    
    le = LabelEncoder()
    
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col])
    
    X['age_squared'] = X['age'] ** 2
    X['bmi_smoker'] = X['bmi'] * X['smoker'] 
    X['age_smoker'] = X['age'] * X['smoker']
    X['bmi_squared'] = X['bmi'] ** 2
    X['age_bmi'] = X['age'] * X['bmi']
    
    print("\nAfter encoding and feature engineering:")
    print(X.head())
    print(f"\nNew X shape: {X.shape}")



# =============================================================================
#  3: Entrenamiento
# - scaler
# - train_test_split
# - Crear LinearRegression y entrenar
# - Predecir y calcular MSE
# =============================================================================
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        train_size=0.8,
        random_state=25
    )

    print(f"\nTrain size: {round(len(X_train) / len(X) * 100)}%")
    print(f"Test size: {round(len(X_test) / len(X) * 100)}%")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"\nMean Squared Error: {mse:.2f}")

# =============================================================================
#  4: Resultados
# - Imprimir loss
# - Mostrar predicciones vs reales
# - Guardar CSV
# =============================================================================
    print("RESULTADOS FINALES")
    
    # Verificar si cumple con el objetivo
    expected_loss = 19_000_000
    print(f"\nLoss (MSE): {mse:,.2f}")
    print(f"Expected loss: <= {expected_loss:,}")
    
    # Mostrar predicciones vs valores reales
    print("\n--- Predicciones vs Valores Reales (primeros 5) ---")
    results_df = pd.DataFrame({
        'Real': y_test.values[:5],
        'Predicción': y_pred[:5],
        'Diferencia': np.abs(y_test.values[:5] - y_pred[:5])
    })
    print(results_df.to_string(index=False))
    
    # Guardar resultados en CSV
    full_results = pd.DataFrame({
        'Real': y_test.values,
        'Predicción': y_pred,
        'Diferencia': np.abs(y_test.values - y_pred)
    })
    full_results.to_csv("resultados_predicciones.csv", index=False)


if __name__ == "__main__":
    main()
