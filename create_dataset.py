import pandas as pd
import os
import requests
import zipfile
import io

def create_and_save_dataset():
    """Завантаження та збереження Air Quality Dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    
    print("Завантажуємо Air Quality UCI датасет...")
    
    # Завантажуємо ZIP через requests (обхід SSL-проблеми macOS)
    response = requests.get(url, verify=False)  # або verify='/path/to/certifi/cacert.pem'
    response.raise_for_status()
    
    # Розпаковуємо ZIP в пам'яті
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Знаходимо CSV файл всередині архіву
        csv_name = [name for name in z.namelist() if name.endswith('.csv')][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep=';', decimal=',')
    
    # Об'єднуємо Date і Time в один datetime стовпець
    df['Date_Time'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], 
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )
    
    # Видаляємо старі колонки
    df = df.drop(['Date', 'Time'], axis=1, errors='ignore')
    df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, errors='ignore')
    
    # Замінюємо -200 на NaN
    df = df.replace(-200, float('nan'))
    
    # Зберігаємо
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/air_quality.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Датасет успішно збережено: {output_path}")
    print(f"   Форма: {df.shape}")
    print(f"   Колонки: {df.columns.tolist()}")
    return df

if __name__ == "__main__":
    create_and_save_dataset()