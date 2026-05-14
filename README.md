# Лабораторна робота №3 — Моніторинг ML API та детекція проблем

**Автор:** Striltsov Denys 
**Група:** TR-52mp
**Варіант:** 18

Розширення REST API класифікації Iris (ЛР2) шаром Prometheus-метрик,
KS-детекцією drift та структурованим JSON-логуванням.

---

## Стек технологій

| Компонент | Технологія |
|---|---|
| API фреймворк | FastAPI 0.115 |
| ASGI сервер | Uvicorn |
| ML бібліотека | scikit-learn 1.5 |
| Серіалізація | joblib |
| Валідація | Pydantic v2 |
| Метрики | prometheus-client 0.21 |
| Drift detection | scipy (KS-тест) |
| Логування | python-json-logger |
| Моніторинг | Prometheus 2.55 |
| Звіти drift | Evidently 0.4 (bonus) |
| Тестування | pytest + httpx |
| CI/CD | GitHub Actions |
| Хостинг | Render |

---

## Структура репозиторію

```
ml-api-lab3/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI з моніторингом
│   ├── schemas.py          # Pydantic-схеми
│   ├── metrics.py          # NEW: Prometheus-метрики
│   ├── drift.py            # NEW: KS-детектор drift
│   └── logging_config.py   # NEW: structured logging
├── ml/
│   ├── __init__.py
│   └── train.py            # Оновлено: зберігає reference stats
├── monitoring/
│   ├── prometheus.yml
│   └── docker-compose.monitoring.yml
├── scripts/
│   └── evidently_report.py  # New: Evidently HTML-звіт
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_metrics.py      # NEW
│   └── test_drift.py        # NEW
├── model.joblib
├── reference_stats.joblib   # NEW
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Реалізовані метрики Prometheus

| Метрика | Тип | Опис |
|---|---|---|
| `ml_predictions_total` | Counter | Кількість прогнозів з мітками `class_name`, `status` |
| `ml_prediction_latency_seconds` | Histogram | Час обробки /predict (buckets 5ms–5s) |
| `ml_prediction_confidence` | Histogram | Розподіл predict_proba (0.1–1.0) |
| `ml_errors_total` | Counter | Помилкові запити з міткою `error_type` |
| `ml_model_loaded` | Gauge | 1 = модель завантажена, 0 = ні |
| `ml_drift_checks_total` | Counter | Кількість перевірок на drift |
| `ml_drift_detected_total` | Counter | Виявлені drift-події з міткою `feature` |

---

## Ендпоінти

| Метод | URL | Опис |
|---|---|---|
| GET | `/` | Статус сервісу |
| GET | `/health` | Liveness probe (модель + drift detector) |
| POST | `/predict` | Класифікація квітки Iris |
| GET | `/metrics` | Prometheus exposition format |
| POST | `/check-drift` | Перевірка drift у батчі даних |
| GET | `/docs` | Swagger UI |

---

## Drift Detection

Реалізовано клас `DriftDetector` (KS-тест Колмогорова-Смирнова):

- зберігає тренувальну вибірку як `reference`
- для кожної ознаки виконує `scipy.stats.ks_2samp(ref, current)`
- повертає `drift_detected`, `p_value`, статистику D по кожній ознаці
- `drift_detected = True` якщо хоча б для однієї ознаки `p_value < alpha`

### Приклад запиту /check-drift

```bash
curl -X POST http://localhost:8000/check-drift \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      [9.0,8.0,8.0,5.0],[9.5,7.5,8.5,5.5],[8.5,8.5,7.5,4.5],
      [9.2,8.2,8.2,5.2],[9.8,7.8,8.8,5.8],[8.8,8.8,7.8,4.8],
      [9.4,8.4,8.4,5.4],[9.6,7.6,8.6,5.6],[8.6,8.6,7.6,4.6],
      [9.1,8.1,8.1,5.1]
    ],
    "alpha": 0.05
  }'
```

```json
{
  "drift_detected": true,
  "n_drifted_features": 4,
  "drifted_features": ["sepal_length","sepal_width","petal_length","petal_width"],
  "per_feature": {
    "sepal_length": {"statistic": 0.99, "p_value": 0.0, "drift_detected": true},
    ...
  },
  "n_samples": 10,
  "alpha": 0.05
}
```

---

## Логування

Структуроване JSON-логування через `python-json-logger`:

```json
{"timestamp":"2026-04-28T10:33:15Z","level":"INFO","logger":"ml-api",
 "message":"prediction","event":"prediction","class_id":0,
 "class_name":"setosa","probability":0.9823,
 "features":{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}}
```

Логуються події: `startup_complete`, `prediction`, `drift_check`, `inference_error`.

---

## Як запустити локально

```bash
# 1. Встановити залежності
pip install -r requirements.txt

# 2. Натренувати модель та зберегти reference stats
python -m ml.train
# Model trained. Test accuracy: 0.9667
# Saved model to:     .../model.joblib
# Saved reference to: .../reference_stats.joblib

# 3. Запустити API
uvicorn app.main:app --reload

# 4. Відкрити Swagger UI
# http://localhost:8000/docs

# 5. Перевірити метрики
# http://localhost:8000/metrics
```

---

## Запуск моніторингу (Prometheus)

```bash
# З каталогу monitoring/
cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build

# Prometheus UI: http://localhost:9090
# Цілі: http://localhost:9090/targets  (має бути UP)
```


## Генерація навантаження

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
    > /dev/null
done
```

---

## Запуск тестів

```bash
pytest -v
```

Очікуваний результат:
```
tests/test_api.py::test_root_endpoint              PASSED
tests/test_api.py::test_health_endpoint            PASSED
tests/test_api.py::test_predict_setosa             PASSED
tests/test_api.py::test_predict_virginica          PASSED
tests/test_api.py::test_predict_invalid_input      PASSED
tests/test_api.py::test_check_drift_no_drift       PASSED
tests/test_api.py::test_check_drift_with_drift     PASSED
tests/test_model.py::test_train_creates_model_file         PASSED
tests/test_model.py::test_model_predicts_three_classes     PASSED
tests/test_model.py::test_reference_stats_structure        PASSED
tests/test_metrics.py::test_metrics_endpoint_available     PASSED
tests/test_metrics.py::test_predict_increments_counter     PASSED
tests/test_metrics.py::test_drift_check_increments_counter PASSED
tests/test_drift.py::test_no_drift_on_same_distribution    PASSED
tests/test_drift.py::test_drift_on_shifted_distribution    PASSED
tests/test_drift.py::test_detect_returns_correct_structure PASSED
tests/test_drift.py::test_invalid_reference_raises         PASSED
tests/test_drift.py::test_invalid_current_raises           PASSED
tests/test_drift.py::test_custom_alpha                     PASSED
19 passed
```

---

## Evidently HTML-звіт (bonus)

```bash
python scripts/evidently_report.py
# Report saved to .../drift_report.html
```

Відкрийте `drift_report.html` у браузері для повного інтерактивного звіту порівняння розподілів.

---

## Висновки

У ЛР3 ML API розширено повноцінним шаром спостережуваності:

1. **Prometheus-метрики** дозволяють у реальному часі бачити latency, throughput, розподіл класів та впевненість моделі.
2. **KS-детектор drift** автоматично виявляє зміщення вхідних даних відносно тренувальної вибірки — критично важливо, бо деградація моделі не проявляється як HTTP-помилка.
3. **Структуроване JSON-логування** перетворює логи на машинно-читабельний джерело даних для аналізу та аудиту.
4. **Evidently** (bonus) забезпечує детальний візуальний звіт для розслідування інцидентів.
