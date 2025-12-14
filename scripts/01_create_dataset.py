# scripts/01_create_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import json
from clearml import Dataset, Task
import asyncio
import httpx
from typing import Dict, List, Optional
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDatasetCreator:
    """Создатель датасета для нескольких городов"""
    
    def __init__(self):
        # Инициализация задачи ClearML
        self.task = Task.init(
            project_name='WeatherForecast',
            task_name='Multi-City Dataset Creation',
            task_type=Task.TaskTypes.data_processing
        )
        
        # Загрузка конфигурации
        with open('config/default.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # API конфигурация
        self.WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
        
        # Города для датасета
        self.CITIES = {
            "austin": {"lat": 30.2672, "lon": -97.7431, "timezone": "America/Chicago"},
            "london": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
            "tokyo": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
            "sydney": {"lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"}
        }
        
        # Пути для сохранения
        self.dataset_dir = 'data/multi_city'
        self.models_dir = 'models/multi_city'
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    async def fetch_city_data(self, city_name: str, city_info: Dict) -> Optional[pd.DataFrame]:
        """Получение данных для одного города"""
        params = {
            "latitude": city_info["lat"],
            "longitude": city_info["lon"],
            "start_date": "2021-01-01",
            "end_date": "2023-12-31",
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                     "precipitation_sum,windspeed_10m_max,weathercode",
            "timezone": city_info["timezone"],
            "format": "json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.info(f"🌤️  Fetching data for {city_name}")
                response = await client.get(self.WEATHER_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "daily" not in data:
                    return None
                
                # Преобразуем в DataFrame
                df = pd.DataFrame(data["daily"])
                df = df.rename(columns={"time": "Date"})
                df["Date"] = pd.to_datetime(df["Date"])
                df["Location"] = city_name
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Error fetching {city_name}: {e}")
            return None
    
    def create_features(self, df: pd.DataFrame, city_name: str) -> pd.DataFrame:
        """Создание признаков для модели"""
        df = df.copy()
        
        # Основные метеопараметры
        if 'temperature_2m_mean' in df.columns:
            df['Temperature_C'] = df['temperature_2m_mean']
        
        # Временные признаки (как в API)
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['dayofyear'] = df['Date'].dt.dayofyear
        df['quarter'] = df['Date'].dt.quarter
        df['hour'] = 12
        
        # Циклические признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        
        # Сезонные признаки
        df['season'] = ((df['month'] % 12 + 3) // 3).astype(int)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Лето/зима в зависимости от полушария
        lat = self.CITIES[city_name]["lat"]
        if lat < 0:  # Южное полушарие
            df['is_summer'] = df['month'].isin([12, 1, 2]).astype(int)
            df['is_winter'] = df['month'].isin([6, 7, 8]).astype(int)
        else:  # Северное полушарие
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # Лаговые признаки
        if 'Temperature_C' in df.columns:
            for lag in [1, 2, 3, 7, 14]:
                df[f'Temperature_C_lag_{lag}d'] = df['Temperature_C'].shift(lag)
        
        # Целевые переменные
        if 'Temperature_C' in df.columns:
            for horizon in [1, 3, 7]:
                df[f'target_temp_{horizon}d'] = df['Temperature_C'].shift(-horizon)
                df[f'target_change_{horizon}d'] = df[f'target_temp_{horizon}d'] - df['Temperature_C']
        
        return df
    
    async def create_dataset(self) -> pd.DataFrame:
        """Создание мультигородского датасета"""
        logger.info("=" * 70)
        logger.info("🌍 СОЗДАНИЕ МУЛЬТИГОРОДСКОГО ДАТАСЕТА")
        logger.info("=" * 70)
        
        all_cities_data = []
        
        # Получаем данные для каждого города
        for city_name, city_info in self.CITIES.items():
            logger.info(f"\n📋 Обработка города: {city_name.upper()}")
            
            city_df = await self.fetch_city_data(city_name, city_info)
            if city_df is not None and len(city_df) > 0:
                city_df = self.create_features(city_df, city_name)
                all_cities_data.append(city_df)
                logger.info(f"✅ {city_name}: {len(city_df)} записей")
            else:
                logger.error(f"❌ {city_name}: пропущен")
        
        if not all_cities_data:
            raise ValueError("Не удалось получить данные ни для одного города")
        
        # Объединяем все города
        combined_df = pd.concat(all_cities_data, ignore_index=True)
        
        # Очистка
        initial_size = len(combined_df)
        key_cols = ['Temperature_C', 'Date']
        for horizon in ['1d', '3d', '7d']:
            if f'target_temp_{horizon}d' in combined_df.columns:
                key_cols.append(f'target_temp_{horizon}d')
        
        combined_df = combined_df.dropna(subset=key_cols)
        logger.info(f"🧹 Очистка: {initial_size - len(combined_df)} строк удалено")
        
        # Статистика
        logger.info(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"  • Записей: {len(combined_df):,}")
        logger.info(f"  • Городов: {len(self.CITIES)}")
        logger.info(f"  • Период: {combined_df['Date'].min().date()} - {combined_df['Date'].max().date()}")
        
        for city in combined_df['Location'].unique():
            city_data = combined_df[combined_df['Location'] == city]
            logger.info(f"  • {city}: {len(city_data)} дней, "
                       f"{city_data['Temperature_C'].mean():.1f}°C")
        
        return combined_df
    
    def save_dataset(self, df: pd.DataFrame):
        """Сохранение датасета"""
        logger.info("\n💾 СОХРАНЕНИЕ ДАТАСЕТА")
        
        # Сохраняем полный датасет
        dataset_path = f'{self.dataset_dir}/weather_multi_city.parquet'
        df.to_parquet(dataset_path, index=False)
        logger.info(f"✅ Датсет сохранен: {dataset_path}")
        
        # Определяем признаки для модели
        exclude_cols = [
            'Date', 'Location', 
            'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
            'precipitation_sum', 'windspeed_10m_max', 'weathercode',
        ]
        
        exclude_cols += [col for col in df.columns if 'target_' in col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Сохраняем список признаков
        features_path = f'{self.models_dir}/feature_list.json'
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"✅ Список признаков сохранен: {features_path}")
        
        # Сохраняем метаданные городов
        city_metadata = {}
        for city_name, city_info in self.CITIES.items():
            city_data = df[df['Location'] == city_name]
            city_metadata[city_name] = {
                'lat': city_info["lat"],
                'lon': city_info["lon"],
                'timezone': city_info["timezone"],
                'records': int(len(city_data)),
                'temp_mean': float(city_data['Temperature_C'].mean()),
                'temp_std': float(city_data['Temperature_C'].std()),
                'temp_min': float(city_data['Temperature_C'].min()),
                'temp_max': float(city_data['Temperature_C'].max()),
                'start_date': city_data['Date'].min().strftime('%Y-%m-%d'),
                'end_date': city_data['Date'].max().strftime('%Y-%m-%d')
            }
        
        metadata_path = f'{self.dataset_dir}/city_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(city_metadata, f, indent=2)
        logger.info(f"✅ Метаданные сохранены: {metadata_path}")
        
        # Создаем ClearML Dataset
        dataset = Dataset.create(
            dataset_project=self.config['project']['name'],
            dataset_name='Multi_City_Weather_v1',
            dataset_tags=['multi-city', 'api-ready', 'weather-forecast']
        )
        
        dataset.add_files(dataset_path)
        dataset.add_files(features_path)
        dataset.add_files(metadata_path)
        dataset.finalize(auto_upload=True)
        
        logger.info(f"✅ ClearML Dataset создан: {dataset.id}")
        
        # Логируем в ClearML
        self.task.upload_artifact('dataset_info', {
            'n_records': len(df),
            'n_cities': len(self.CITIES),
            'n_features': len(feature_cols),
            'date_range': f"{df['Date'].min().date()} - {df['Date'].max().date()}",
            'cities': list(self.CITIES.keys())
        })
        
        return dataset_path, features_path, metadata_path
    
    def run(self):
        """Основной метод запуска"""
        try:
            # Создаем датасет
            df = asyncio.run(self.create_dataset())
            
            # Сохраняем
            self.save_dataset(df)
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 ДАТАСЕТ УСПЕШНО СОЗДАН!")
            logger.info("=" * 70)
            
            self.task.close()
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            self.task.close()
            raise

if __name__ == "__main__":
    creator = WeatherDatasetCreator()
    creator.run()