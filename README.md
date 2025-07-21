# 💸 Predicción Temprana de Quiebra Empresarial con Inteligencia Artificial 🚀

## 📌 Introducción

Las quiebras empresariales pueden causar pérdidas millonarias, interrupciones operativas y un impacto severo en la estabilidad económica. Este proyecto de **Machine Learning** tiene como objetivo anticipar la quiebra de empresas mediante un sistema de alerta temprana que transforma incertidumbre en decisiones estratégicas.

---

## 💡 El Problema

Las quiebras generan:
- 💸 **Pérdidas Financieras**: Para bancos, inversores y proveedores.
- 🔗 **Interrupciones Operativas**: En cadenas de suministro y relaciones comerciales.
- ❗ **Falta de Anticipación**: Las herramientas tradicionales no son suficientes.

👉 Este proyecto busca cerrar esa brecha, ofreciendo un **“radar financiero”** basado en IA que detecta señales de riesgo con suficiente antelación.

---

## 🎯 La Solución: Radar Financiero Impulsado por IA

Hemos desarrollado un sistema integral que permite:                                                                                                                                                                 
✔️ **Predecir la Probabilidad de Quiebra** (clasificación binaria: quiebra / no quiebra).  
✔️ **Asignar un Perfil de Riesgo** mediante análisis de clústeres.  
✔️ **Visualizar los resultados** en una aplicación interactiva para usuarios no técnicos.                                                                                                                           
✔️ **Análisis Finnanciero realizado por IA**                                                                                                                                                                        
✔️ **Chatbot de un agente IA ser experto en finanzas**                                                                                                                                                              ✔️ **Base de datos NoSQL (Firebase de Google) para análisis de datos y reentrenamiento**                                                                                                                            

👥 Beneficiarios:
- 🏦 Bancos y Entidades Financieras.
- 💰 Inversores y Fondos.
- 🏢 Empresas (autodiagnóstico).
- 🤝 Departamentos de Compras/Ventas.

---

## 🛠️ Pipeline End-to-End

### 1️⃣ Adquisición y Preprocesamiento de Datos
- Fuente: Kaggle - Company Bankruptcy Prediction (6,819 registros, 96 variables).
- Normalización de nombres, conversión de tipos, imputación de valores faltantes.
- Reducción de características a 78 tras eliminar correlación alta y ruido.

### 2️⃣ Modelado Supervisado
- Modelos evaluados:
  - RandomForest (Class Weighted) → **Modelo Final Seleccionado**.
  - AdaBoost
  - XGBoost
  - GradientBoosting
  - LightGBM
- Estrategias contra desbalance:
  - `class_weight='balanced'`
  - **SMOTE** (oversampling a 25%).
- Métrica prioritaria: **Recall (detección de quiebras)**.

### 3️⃣ Análisis No Supervisado (Clustering)
- **PCA** (95% de varianza, 52 componentes).
- **KMeans** (3 clústeres):
  - 🔵 Clúster 0: Bajo Riesgo (0.14% quiebra).
  - 🟡 Clúster 2: Riesgo Moderado (4.5% quiebra).
  - 🔴 Clúster 1: Riesgo Extremo (75% quiebra, aunque con muy pocos casos).

### 4️⃣ Despliegue (Streamlit App, imagen alocada en render)
- Subida de archivos CSV.
- Predicción binaria + probabilidad de quiebra.
- Asignación de clúster de riesgo.
- Visualización de perfil financiero.

---

## 📈 Resultados Clave

| Métrica                      | Valor     |
|------------------------------|-----------|
| **Modelo Final**              | RandomForest (Class Weighted) |
| **Recall (detección quiebras)** | 66% (detecta 66 de cada 100 quiebras reales) |
| **Precisión (fiabilidad alertas)** | 25% (1 de cada 4 alertas es correcta) |
| **ROC AUC**                   | 0.898     |

👉 Se prioriza **detectar la mayoría de las quiebras**, asumiendo un mayor número de falsas alarmas.

### Principales Variables Predictivas:
1. Persistent EPS in the last four seasons.
2. Retained Earnings to Total Assets.
3. Total Income / Total Expense.
4. Debt Ratio (%).
5. Borrowing Dependency.

---

## 📂 Estructura del Proyecto
proyecto_Prediccion_Bancarrota/            
├── app_streamlit/           # Aplicación Streamlit para predicción                              
├── data/                    # Datos de entrada                    
│   ├── processed/           # Datos procesados              
│   ├── raw/                 # Datos en crudo                
│   ├── test/                # Conjunto de prueba                  
│   └── train/               # Conjunto de entrenamiento                  
├── docs/                    # Documentación y presentaciones        
├── models/                  # Modelos entrenados y objetos serializados        
├── notebooks/               # Notebooks de análisis y desarrollo          
├── reports/                 # Reportes y resultados gráficos                                    
├── src/                     # Código fuente (preprocesamiento, entrenamiento, clustering)  
└── README.md                # Archivo README del proyecto              



## 🚀 Cómo Empezar
### 1️⃣ Clonar el repositorio:


git clone https://github.com/tu_usuario/nombre_proyecto_final_ML.git
cd nombre_proyecto_final_ML

### 2️⃣ Ejecutar el pipeline de análisis y modelado:


python src/data_processing.py
python src/model_training.py
python src/cluster_analysis.py
python src/model_evaluation.py

### 3️⃣ Lanzar la aplicación:


ejecutar el streamlit app de manera local o pedir el enlace web a la aplicación desplegada en línea.                                                                                                                
🔍 Conclusiones y Recomendaciones
✅ El modelo RandomForest (Class Weighted) ofrece la mejor capacidad de detección temprana de quiebras (66% de recall).
✅ Se acepta un mayor número de falsas alarmas para evitar no detectar quiebras reales.
✅ El análisis de clústeres añade un contexto extra identificando perfiles de riesgo latente.

## Recomendaciones:
Implementar un sistema automatizado de alertas para clientes o proveedores en riesgo.

Realizar reentrenamiento periódico con nuevos datos.

Establecer revisión manual de casos con alerta para validar y mejorar la precisión.

Monitorear de forma especial el Clúster 1 (Riesgo Extremo).


