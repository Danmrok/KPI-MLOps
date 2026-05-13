# MLOps Lab 1: Air Quality CO Classification

**Автор:** Striltsov Denys
**Група:** TR-52mp
**Дата:** 2026  
**Варіант:** 18

---

## Опис проекту

Класифікація рівня монооксиду вуглецю (CO) у повітрі на основі Air Quality UCI Dataset.
Датасет містить щогодинні вимірювання хімічних сенсорів та метеопоказники, зібрані
в Італії протягом 2004–2005 рр.

**Задача:** Мультикласова класифікація рівня CO:
- `0` — Low CO (< 2 мг/м³)
- `1` — Medium CO (2–7 мг/м³)
- `2` — High CO (> 7 мг/м³)

---

## Структура проекту

```
kpi_mlops_lab1/
├── data/
│   ├── external/               # Зовнішні джерела даних
│   ├── processed/              # Оброблені дані з таргетом (під DVC)
│   │   ├── air_quality_processed.csv.dvc
│   │   └── .gitkeep
│   └── raw/                    # Оригінальні незмінні дані (під DVC)
│       ├── air_quality.csv.dvc # DVC метафайл
│       └── .gitkeep
│   
├── models/                     # Збережені моделі (під DVC)
│   ├── pipeline.pkl.dvc        # DVC метафайл найкращої моделі
│   └── .gitkeep
├── notebooks/                  # Jupyter notebooks для EDA
├── src/
│   ├── data/                   # Завантаження та обробка даних
│   ├── features/               # Feature engineering
│   └── models/
│       ├── pipeline.py         # Визначення ML Pipeline
│       └── train_pipeline.py   # Тренування з MLflow
├── tests/                      # Юніт-тести
├── create_dataset.py           # Завантаження та підготовка датасету
├── .gitignore
├── .dvcignore
├── requirements.txt
└── README.md
```

---

## Швидкий старт

### Встановлення

```bash
# Клонування репозиторію
git clone <repo-url>
cd kpi_mlops_lab1

# Встановлення залежностей
pip install -r requirements.txt
```

### Отримання даних через DVC

```bash
# Налаштування remote (локальний)
dvc remote add -d myremote /tmp/dvc-storage-lab1

# Отримати дані та модель
dvc pull
```

### Завантаження та підготовка датасету

```bash
python create_dataset.py
```

### Тренування всіх моделей

```bash
python src/models/train_pipeline.py
```

### Запуск MLflow UI

```bash
mlflow ui --port 5000
# Відкрити: http://localhost:5000
```

---

## Датасет

| Параметр | Значення |
|---|---|
| Джерело | UCI Machine Learning Repository — Air Quality Dataset |
| Розмір | 9 357 записів × 14 ознак |
| Частота | Щогодинні вимірювання |
| Період | 10 березня 2004 — 4 квітня 2005 |
| Тип задачі | Мультикласова класифікація (3 класи) |

### Ознаки

| Стовпець | Опис |
|---|---|
| `PT08.S1(CO)` | Відгук сенсора на CO (SnO2) |
| `C6H6(GT)` | Концентрація бензолу (мкг/м³) |
| `PT08.S2(NMHC)` | Відгук сенсора на NMHC (titania) |
| `PT08.S3(NOx)` | Відгук сенсора на NOx (вольфрамат) |
| `PT08.S4(NO2)` | Відгук сенсора на NO2 (WO3) |
| `PT08.S5(O3)` | Відгук сенсора на O3 (indium oxide) |
| `T` | Температура (°C) |
| `RH` | Відносна вологість (%) |
| `AH` | Абсолютна вологість |

### Розподіл класів

| Клас | Опис | Кількість | % |
|---|---|---|---|
| 0 | Low CO | 5 860 | 62.6% |
| 1 | Medium CO | 3 466 | 37.0% |
| 2 | High CO | 31 | 0.3% |

---

## Результати експериментів

### Таблиця всіх runs

| # | Модель | max_depth | n_est | lr | CV Acc ± std | Test Acc | Test F1 | Run ID |
|---|---|---|---|---|---|---|---|---|
| 1 | **DecisionTree** | 5 | — | — | 0.8445 ± 0.010 | 0.8317 | 0.8317 | `ea166e88` |
| 2 | DecisionTree | 10 | — | — | 0.8208 ± 0.012 | 0.8093 | 0.8094 | `d0d9c2a2` |
| 3 | DecisionTree | 15 | — | — | 0.8110 ± 0.012 | 0.7831 | 0.7843 | `3daa1f6b` |
| 4 | **RandomForest** ⭐ | 10 | 100 | — | 0.8486 ± 0.014 | **0.8376** | **0.8383** | `d7cedd13` |
| 5 | RandomForest | 15 | 200 | — | 0.8470 ± 0.015 | 0.8328 | 0.8330 | `48536474` |
| 6 | **GradientBoosting** | 3 | 100 | 0.10 | 0.8496 ± 0.013 | 0.8365 | 0.8367 | `74a2fc0e` |
| 7 | GradientBoosting | 5 | 150 | 0.05 | 0.8462 ± 0.014 | 0.8349 | 0.8354 | `820edfa4` |

### Найкраща модель

- **Модель:** Random Forest Classifier  
- **Параметри:** `n_estimators=100`, `max_depth=10`, `random_state=42`  
- **CV Accuracy:** 0.8486 ± 0.0139  
- **Test Accuracy:** **0.8376**  
- **Test F1 (weighted):** **0.8383**  
- **Run ID:** `d7cedd13ae0c4c96b5d96b792bed5460`  
- **Збережено:** `models/pipeline.pkl`

### Classification Report (найкраща модель)

```
              precision    recall  f1-score   support

      Low CO       0.88      0.86      0.87      1172
   Medium CO       0.77      0.81      0.79       694
     High CO       0.75      0.50      0.60         6

    accuracy                           0.84      1872
   macro avg       0.80      0.72      0.75      1872
weighted avg       0.84      0.84      0.84      1872
```

### Висновки

1. **Decision Tree** із малою глибиною (5) показує найкращий результат серед DT,
   глибші дерева перенавчаються.
2. **Random Forest** (100 дерев, depth=10) — найкраща загальна модель: +0.6% vs DT.
3. **Gradient Boosting** показує конкурентні результати, але потребує більше часу на тренування.
4. Всі моделі добре справляються з класами 0 та 1, клас 2 (High CO) складний через
   значний дисбаланс (~0.3% даних).

---

## Pipeline

```
[SimpleImputer (median)] → [StandardScaler] → [Classifier]
```

Переваги Pipeline:
- **Запобігає data leakage**: scaler fit тільки на train даних
- **Відтворюваність**: всі кроки серіалізовані разом
- **Простота deployment**: один `.pkl` файл

---

## Використання моделі

```python
from src.utils import load_model_from_file, predict_with_labels
import numpy as np

# Завантаження
model = load_model_from_file('models/pipeline.pkl')

# Передбачення (9 ознак: PT08.S1, C6H6, PT08.S2, PT08.S3, PT08.S4, PT08.S5, T, RH, AH)
sample = np.array([[1000, 5.2, 950, 1400, 1200, 900, 18, 55, 1.1]])
labels = predict_with_labels(model, sample)
print(labels)  # ['Low CO'] / ['Medium CO'] / ['High CO']
```

---

## DVC Workflow

```bash
# Перевірка статусу
dvc status

# Оновлення даних
python create_dataset.py
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Update dataset"
dvc push

# Відновлення даних (на новій машині)
git clone <repo>
dvc pull

# Повернення до попередньої версії даних
git checkout <commit-hash> data/raw/air_quality.csv.dvc
dvc checkout
```

---

## MLflow Workflow

```bash
# Запуск UI
mlflow ui --port 5000

# Перегляд експериментів через CLI
mlflow experiments list

# Список runs
mlflow runs list --experiment-name air_quality_co_classification
```

---

## Troubleshooting

### SSL Certificate Error (macOS)

```
ssl.SSLCertVerificationError: certificate verify failed
```

**Рішення 1** (рекомендовано):
```bash
/Applications/Python\ 3.X/Install\ Certificates.command
```

**Рішення 2** (в коді — для розробки):
```python
import requests, urllib3
urllib3.disable_warnings()
response = requests.get(url, verify=False)
```

### DVC push fails

```bash
# Перевірити конфіг remote
dvc remote list
# Перевірити чи існує директорія
mkdir -p /tmp/dvc-storage-lab1
```

### MLflow — experiment not found

```bash
# Переконатись що mlruns/ в поточній директорії
cd kpi_mlops_lab1
mlflow ui
```

### Imbalanced classes (High CO = 0.3%)

Клас 2 рідкий — можна покращити через:
```python
from sklearn.utils.class_weight import compute_class_weight
# або використати SMOTE для oversampling
```