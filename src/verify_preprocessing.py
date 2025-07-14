import pandas as pd
import os
import numpy as np # Importar numpy para select_dtypes

# Define la ruta al archivo data_processed.csv
processed_data_file = '../data/processed/data_processed.csv'

# Lista de columnas que quieres describir.
# ¡IMPORTANTE!: Estos nombres de columnas deben ser los NOMBRES LIMPIOS
# que resultan de tu función `limpiar_nombres_columnas` en `data_preprocessing.py`.
columns_to_describe_cleaned = [
    'operating_expense_rate',
    'research_and_development_expense_rate',
    'interest_bearing_debt_interest_rate',
    'revenue_per_share_yuan',
    'total_asset_growth_rate',
    'net_value_growth_rate',
    'current_ratio',
    'quick_ratio',
    'total_debt_total_net_worth',
    'accounts_receivable_turnover',
    'average_collection_days',
    'inventory_turnover_rate_times',
    'fixed_assets_turnover_frequency',
    'revenue_per_person',
    'allocation_rate_per_person',
    'quick_assets_current_liability',
    'cash_current_liability',
    'inventory_current_liability',
    'long_term_liability_to_current_assets',
    'current_asset_turnover_rate',
    'quick_asset_turnover_rate',
    'cash_turnover_rate',
    'fixed_assets_to_assets',
    'total_assets_to_gnp_price'
]

try:
    # Cargar el DataFrame preprocesado
    df_processed = pd.read_csv(processed_data_file)
    print(f"DataFrame cargado desde: {processed_data_file}. Shape: {df_processed.shape}")

    # Primero, verifica los nombres de las columnas en el df_processed para estar seguro
    print("\nNombres de columnas en el DataFrame procesado (primeras 10):")
    print(df_processed.columns.tolist()[:10]) # Muestra las 10 primeras para verificar

    actual_columns_to_describe = [col for col in columns_to_describe_cleaned if col in df_processed.columns]

    if not actual_columns_to_describe:
        print("\nAdvertencia: Ninguna de las columnas especificadas fue encontrada en el DataFrame.")
        print("Asegúrate de que los nombres de las columnas en la lista 'columns_to_describe_cleaned' coincidan")
        print("con los nombres de las columnas limpias en tu 'data_processed.csv'.")

    else:
        print(f"\nEstadísticas descriptivas para las columnas especificadas ({len(actual_columns_to_describe)} encontradas):")
        # Realizar el describe para las columnas seleccionadas
        # Utiliza .T para transponer y ver las estadísticas por columna más fácilmente
        print(df_processed[actual_columns_to_describe].describe().T[['min', 'max', 'mean', 'std']])

        # Opcional: describe de todas las columnas numéricas para una vista general
        print("\nEstadísticas descriptivas para todas las columnas numéricas (vista general):")
        print(df_processed.select_dtypes(include=np.number).describe().T[['min', 'max', 'mean', 'std']])


except FileNotFoundError:
    print(f"Error: Archivo no encontrado en '{processed_data_file}'.")
    print("Asegúrate de que 'data_preprocessing.py' se haya ejecutado correctamente y haya creado el archivo.")
except Exception as e:
    print(f"Ocurrió un error al cargar o describir el DataFrame: {e}")