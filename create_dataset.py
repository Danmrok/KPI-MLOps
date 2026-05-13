import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io


def download_air_quality() -> pd.DataFrame:

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    print("Завантажуємо Air Quality UCI датасет...")
    try:
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep=';', decimal=',')
        print(f"✅ Завантажено з UCI: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️  Не вдалось завантажити з UCI ({e}), генеруємо синтетичний датасет...")
        return generate_synthetic_air_quality()


def generate_synthetic_air_quality() -> pd.DataFrame:

    np.random.seed(42)
    n = 9357  # розмір оригінального датасету

    # Базові концентрації газів (мкг/м³)
    co_true = np.random.gamma(shape=2.5, scale=0.8, size=n)           # CO (GT)
    nmhc_true = np.random.gamma(shape=1.5, scale=50, size=n)          # NMHC (GT)
    nox_true = np.random.gamma(shape=2.0, scale=60, size=n)           # NOx (GT)
    no2_true = np.random.gamma(shape=3.0, scale=30, size=n)           # NO2 (GT)

    # Показники сенсорів (з шумом + дрейфом)
    noise = lambda scale: np.random.normal(0, scale, n)

    df = pd.DataFrame({
        'Date_Time': pd.date_range('2004-03-10', periods=n, freq='h'),
        'CO(GT)': co_true,
        'PT08.S1(CO)': 1000 + co_true * 150 + noise(80),
        'NMHC(GT)': nmhc_true,
        'C6H6(GT)': nmhc_true * 0.08 + noise(2),
        'PT08.S2(NMHC)': 900 + nmhc_true * 3 + noise(100),
        'NOx(GT)': nox_true,
        'PT08.S3(NOx)': 1600 - nox_true * 1.2 + noise(100),
        'NO2(GT)': no2_true,
        'PT08.S4(NO2)': 1400 + no2_true * 4 + noise(120),
        'PT08.S5(O3)': 800 + nox_true * 2 + noise(150),
        'T': np.random.normal(18, 8, n),       # Температура (°C)
        'RH': np.random.normal(50, 15, n),     # Відносна вологість (%)
        'AH': np.random.normal(1.0, 0.3, n),   # Абсолютна вологість
    })

    # Вводимо пропуски (-200 в оригіналі), ~10% даних
    for col in ['CO(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']:
        mask = np.random.random(n) < 0.10
        df.loc[mask, col] = np.nan

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    if 'Date' in df.columns and 'Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H.%M.%S',
            errors='coerce'
        )
        df = df.drop(['Date', 'Time'], axis=1)

    # Видаляємо порожні стовпці Unnamed
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    df = df.drop(unnamed, axis=1, errors='ignore')

    # Замінюємо -200 (сенсорні збої) на NaN
    df = df.replace(-200, np.nan)
    df = df.replace(-200.0, np.nan)

    # Виключаємо нечислові стовпці при заповненні NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Видаляємо рядки, де немає CO(GT) (потрібний для таргету)
    if 'CO(GT)' in df.columns:
        df = df.dropna(subset=['CO(GT)'])

    # ── Цільова змінна ──────────────────────────────────────────────
    # CO рівні (WHO: <2 мг/м³ — норма, 2-10 — підвищений, >10 — небезпечний)
    # В датасеті одиниці мг/м³
    if 'CO(GT)' in df.columns:
        df['CO_level'] = pd.cut(
            df['CO(GT)'],
            bins=[-np.inf, 2.0, 7.0, np.inf],
            labels=[0, 1, 2]   # 0=низький, 1=середній, 2=високий
        ).astype(int)

    df = df.reset_index(drop=True)
    return df


def create_and_save_dataset() -> pd.DataFrame:
    """Головна функція: завантаження, обробка та збереження датасету."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 1. Завантаження
    df_raw = download_air_quality()

    # 2. Зберігаємо сирі дані
    raw_path = 'data/raw/air_quality.csv'
    df_raw.to_csv(raw_path, index=False)
    print(f"📁 Сирі дані збережено: {raw_path}  shape={df_raw.shape}")

    # 3. Обробка
    df = preprocess(df_raw.copy())

    # 4. Зберігаємо оброблені
    proc_path = 'data/processed/air_quality_processed.csv'
    df.to_csv(proc_path, index=False)

    print(f"\n✅ Оброблений датасет збережено: {proc_path}")
    print(f"   Форма: {df.shape}")
    print(f"   Стовпці: {df.columns.tolist()}")
    if 'CO_level' in df.columns:
        print(f"   Розподіл класів CO_level:\n{df['CO_level'].value_counts().sort_index()}")

    return df


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    create_and_save_dataset()