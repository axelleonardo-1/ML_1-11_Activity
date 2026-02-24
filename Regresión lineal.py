"""
Activity 1.11: Linear Regression - Medical Cost Personal dataset
Objetivo: Predecir "charges" con loss <= 19,000,000
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

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
# =============================================================================
    categorical_columns = ["sex", "smoker", "region"]
    
    le = LabelEncoder()
    
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col])
    
    print("\nAfter encoding:")
    print(X.head())



# =============================================================================
#  3: Entrenamiento
# - train_test_split
# - Crear LinearRegression y entrenar
# - Predecir y calcular MSE
# =============================================================================


# =============================================================================
#  4: Resultados
# - Imprimir loss
# - Mostrar predicciones vs reales
# - Guardar CSV
# =============================================================================


if __name__ == "__main__":
    main()
