# Лабораторна робота №2

**Автор:** Striltsov Denys
**Група:** TR-52mp
**Варіант:** 18

REST API для класифікації квіток Iris на основі FastAPI + scikit-learn,
з автоматичним CI через GitHub Actions та розгортанням на Render.

---

## Стек технологій

| Компонент | Технологія |
|---|---|
| API фреймворк | FastAPI 0.115 |
| ASGI сервер | Uvicorn |
| ML бібліотека | scikit-learn 1.5 |
| Серіалізація моделі | joblib |
| Валідація даних | Pydantic v2 |
| Тестування | pytest + httpx |
| Контейнеризація | Docker |
| CI/CD | GitHub Actions |
| Хостинг | Render |

---

## Структура репозиторію

```
ml-api-lab2/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions workflow
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI застосунок
│   └── schemas.py          # Pydantic-схеми вхід/вихід
├── ml/
│   ├── __init__.py
│   └── train.py            # Скрипт тренування моделі
├── tests/
│   ├── __init__.py
│   ├── test_api.py         # Інтеграційні тести API
│   └── test_model.py       # Unit-тести моделі
├── model.joblib            # Артефакт навченої моделі
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Як запустити локально

```bash
# 1. Клонувати репозиторій
git clone <repo-url>
cd ml-api-lab2

# 2. Встановити залежності
pip install -r requirements.txt

# 3. Натренувати модель
python -m ml.train
# Model trained. Test accuracy: 0.9667

# 4. Запустити API
uvicorn app.main:app --reload

# 5. Відкрити документацію
# http://localhost:8000/docs
```

---

## Запуск через Docker

```bash
# Зібрати образ (модель тренується всередині при збірці)
docker build -t ml-api:lab2 .

# Запустити контейнер
docker run --rm -p 8000:8000 ml-api:lab2

# Перевірити
curl http://localhost:8000/health
```

---

## Як запустити тести

```bash
pytest -v
```

Очікуваний результат:
```
tests/test_api.py::test_root_endpoint        PASSED
tests/test_api.py::test_health_endpoint      PASSED
tests/test_api.py::test_predict_setosa       PASSED
tests/test_api.py::test_predict_virginica    PASSED
tests/test_api.py::test_predict_invalid_input PASSED
tests/test_model.py::test_train_creates_model_file    PASSED
tests/test_model.py::test_model_predicts_three_classes PASSED
7 passed
```

---

## Як працює API

### Ендпоінти

| Метод | URL | Опис |
|---|---|---|
| GET | `/` | Статус сервісу |
| GET | `/health` | Перевірка здоров'я (liveness probe) |
| POST | `/predict` | Класифікація квітки Iris |
| GET | `/docs` | Swagger UI документація |

### Приклад запиту до /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Відповідь

```json
{
  "class_id": 0,
  "class_name": "setosa",
  "probability": 0.9823
}
```

### Класи

| class_id | class_name | Опис |
|---|---|---|
| 0 | setosa | Iris setosa |
| 1 | versicolor | Iris versicolor |
| 2 | virginica | Iris virginica |

---

## Посилання на деплой

🌐 **Render:** `https://ml-api-<your-name>.onrender.com`

```bash
# Перевірити що сервіс живий
curl https://ml-api-<your-name>.onrender.com/health
# {"status":"healthy","model_loaded":true}
```

---

## Розгортання на Render

1. Зареєструватись на [render.com](https://render.com) через GitHub
2. New → Web Service → вибрати репозиторій
3. Environment: **Docker**
4. Instance Type: **Free**
5. Натиснути **Create Web Service**
6. Після білду (~5 хв) отримати публічний URL

---

## Troubleshooting

**`FileNotFoundError: model.joblib`**  
→ Забули запустити `python -m ml.train` перед стартом API

**`ModuleNotFoundError: No module named 'app'`**  
→ Запускати uvicorn з кореня проекту, не з середини папки `app/`

**Тести падають з `model_loaded: False`**  
→ Використовувати `TestClient` як контекстний менеджер (`with TestClient(app) as c`)
