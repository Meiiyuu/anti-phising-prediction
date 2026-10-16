# Anti-Phishing: detección de URLs de phishing con Machine Learning

Proyecto que identifica si una página web es **phishing** o **legítima** a partir de características extraídas de su URL y de su código fuente HTML. Se entrenan y comparan cinco clasificadores binarios: red neuronal (PyTorch), Random Forest, SVM, KNN y Naive Bayes.


## Contexto

El phishing consiste en engañar al usuario mediante una copia de una página web legítima o a través de comunicaciones por email o mensajería en las que se suplanta la identidad de la empresa original, con el fin de que el usuario introduzca su información personal. Este proyecto implementa una estrategia anti-phishing basada en aprendizaje automático.

## Dataset

[Phishing Dataset for Machine Learning](https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning?resource=download) (Kaggle), archivo `Phishing_Legitimate_full.csv`.

- 10.000 páginas web: 5.000 phishing y 5.000 legítimas (clases balanceadas).
- 48 características extraídas de la URL y del código HTML.
- Variable objetivo `CLASS_LABEL`: `1` = phishing, `0` = legítima.

## Metodología

1. **Preprocesamiento:** se eliminan columnas redundantes y aquellas que no brindan información valiosa. 
2. **Selección de características:** RFECV (Recursive Feature Elimination with Cross-Validation) con un árbol de decisión como estimador, evaluado con `RepeatedStratifiedKFold` (10 divisiones, 3 repeticiones). Se conservan 23 características.
3. **Modelado:** entrenamiento y evaluación de los cinco clasificadores.
4. **Evaluación:** accuracy y F1 sobre el conjunto de prueba de cada modelo.

### Modelos

| Modelo | Detalles |
|---|---|
| Red neuronal (PyTorch) | Arquitectura 23 → 15 → 3 → 1 con activación sigmoide, optimizador Adam (lr = 0.001), `BCELoss`, batch de 3 y 20 épocas. Se conserva el modelo con menor error de validación. |
| Random Forest | `RandomForestClassifier` de scikit-learn. |
| SVM | Kernel RBF. `C` y `gamma` se buscan con Grid Search (5 folds, métrica `roc_auc`) sobre datos estandarizados. |
| KNN | k = 3 sobre datos estandarizados. También se analiza el accuracy para k de 1 a 99. |
| Naive Bayes | `GaussianNB` con hiperparámetros por defecto. |

## Resultados

| Modelo | Partición (train / test) | Accuracy | F1 |
|---|---|---|---|
| **Random Forest** | 70% / 30% | **0.978** | **0.979** |
| SVM | 70% / 30% | 0.958 | 0.959 |
| KNN (k = 3) | 70% / 30% | 0.946 | 0.947 |
| Red neuronal | 81% / 10% (+ 9% validación) | 0.943 | 0.942 |
| Naive Bayes | 67% / 33% | 0.858 | 0.842 |

Random Forest obtuvo el mejor rendimiento. Naive Bayes tuvo el menor, probablemente por la correlación entre varias características, que incumple su supuesto de independencia.

## Estructura del repositorio

```
.
├── Anti_Phishing.ipynb            # Notebook con el análisis completo
├── Phishing_Legitimate_full.csv   # Dataset 
├── requirements.txt               # Dependencias
└── README.md
```

## Instalación y uso

Requiere Python 3.9 o superior (probado con Python 3.12).

```bash
# 1. Clonar el repositorio
git clone <URL-del-repositorio>
cd <nombre-del-repositorio>

# 2. (Opcional) Crear un entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Abrir el notebook
jupyter notebook Anti_Phishing.ipynb
```

## Tecnologías

Python · pandas · NumPy · scikit-learn · PyTorch · Matplotlib · Seaborn · Jupyter
