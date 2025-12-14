# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import logging
import traceback
import httpx
from typing import Tuple
import asyncio
from scipy import stats

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather Forecast API",
    description="API для прогнозирования температуры в Остине и получения исторических данных",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Конфигурация
WEATHER_API_URL: str = "https://api.open-meteo.com/v1/forecast"
CITY_COORDINATES = {
    "austin": {"lat": 30.2672, "lon": -97.7431, "timezone": "America/Chicago"},
    "moscow": {"lat": 55.7558, "lon": 37.6173, "timezone": "Europe/Moscow"},
    "newyork": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
    "london": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
    "tokyo": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
    "paris": {"lat": 48.8566, "lon": 2.3522, "timezone": "Europe/Paris"},
    "berlin": {"lat": 52.5200, "lon": 13.4050, "timezone": "Europe/Berlin"},
    "beijing": {"lat": 39.9042, "lon": 116.4074, "timezone": "Asia/Shanghai"},
    "sydney": {"lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"},
    "dubai": {"lat": 25.2048, "lon": 55.2708, "timezone": "Asia/Dubai"},
}

# ==================== МОДЕЛЬ И ДАННЫЕ ====================
model = None
feature_names = []
is_dummy_model = False

def create_dummy_model():
    """Создание демо-модели для тестирования без LightGBM"""
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Создаем простую модель
    np.random.seed(42)
    X = np.random.rand(100, 15)
    # Реалистичные температуры для Остина: 10-35°C
    y = 22.5 + 12.5 * np.sin(2 * np.pi * X[:, 1]) + np.random.randn(100) * 3
    
    dummy_model = RandomForestRegressor(n_estimators=20, random_state=42)
    dummy_model.fit(X, y)
    
    # Создаем имена признаков
    dummy_features = [
        'year', 'month', 'day', 'dayofweek', 'dayofyear',
        'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
        'hour', 'is_weekend', 'is_summer', 'is_winter',
        'season', 'quarter'
    ]
    
    # Добавляем feature_names_in_ для совместимости
    dummy_model.feature_names_in_ = dummy_features
    
    return dummy_model, dummy_features

def load_hpo_model():
    """Попытка загрузки HPO модели"""
    global model, feature_names, is_dummy_model
    
    model_paths = [
        "models/api_ready/hpo_fastapi_model.pkl",  # Новая HPO модель
        "models/austin_fixed_model.pkl"  # Резервная модель
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                logger.info(f"🔄 Attempting to load model from {model_path}")
                
                # Загружаем HPO модель
                import pickle
                with open(model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                # HPO модель сохранена как словарь с метаданными
                if isinstance(loaded_data, dict) and 'model' in loaded_data:
                    model = loaded_data['model']
                    feature_names = loaded_data.get('feature_names', [])
                    
                    # Если есть вспомогательные функции в модели, используем их
                    if 'api_helpers' in loaded_data:
                        logger.info("✅ API-совместимая модель загружена с вспомогательными функциями")
                    
                    logger.info(f"✅ Model loaded from HPO format")
                    logger.info(f"📊 Features: {len(feature_names)}")
                    logger.info(f"🏙️  Cities metadata: {len(loaded_data.get('city_metadata', {}))}")
                    
                    # Добавляем feature_names_in_ если его нет
                    if feature_names and not hasattr(model, 'feature_names_in_'):
                        try:
                            model.feature_names_in_ = np.array(feature_names)
                        except:
                            logger.warning("⚠️  Could not add feature_names_in_ attribute")
                    
                    is_dummy_model = False
                    return True
                
                else:
                    logger.warning("⚠️  Invalid format - expecting dictionary with 'model' key")
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️  Failed to load {model_path}: {error_msg}")
                
                # Пробуем через joblib
                try:
                    loaded_model = joblib.load(model_path)
                    model_type = type(loaded_model).__name__
                    
                    # Получаем имена признаков
                    if hasattr(loaded_model, 'feature_names_in_'):
                        feature_names = list(loaded_model.feature_names_in_)
                    else:
                        # Используем стандартные признаки
                        feature_names = [
                            'year', 'month', 'day', 'dayofweek', 'dayofyear',
                            'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
                            'hour', 'is_weekend', 'is_summer', 'is_winter',
                            'season', 'quarter'
                        ]
                    
                    model = loaded_model
                    is_dummy_model = False
                    
                    logger.info(f"✅ Model loaded with joblib: {model_type}")
                    return True
                    
                except Exception as joblib_error:
                    logger.warning(f"⚠️  Failed to load with joblib: {joblib_error}")
    
    return False
# ==================== ЗАГРУЗКА МОДЕЛИ ====================
@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске приложения"""
    global model, feature_names, is_dummy_model
    
    logger.info("🚀 Starting Weather Forecast API...")
    
    # Пробуем загрузить HPO модель
    if load_hpo_model():
        logger.info("✅ HPO model loaded successfully")
    else:
        # Создаем демо-модель
        logger.info("🔄 Creating dummy model...")
        model, feature_names = create_dummy_model()
        is_dummy_model = True
        logger.info("✅ Dummy model created")
    
    logger.info(f"📊 Final model type: {'dummy' if is_dummy_model else 'production'}")
    logger.info(f"🎯 Features count: {len(feature_names)}")
    
    # Выводим информацию о модели при запуске
    if model is not None:
        logger.info(f"📈 Model ready: {type(model).__name__}")
        if hasattr(model, 'n_estimators'):
            logger.info(f"🌳 RandomForest estimators: {model.n_estimators}")

# ==================== МОДЕЛИ ДАННЫХ ====================
class PredictionRequest(BaseModel):
    dates: List[str] = Field(
        ...,
        description="Список дат для прогноза в формате YYYY-MM-DD",
        example=["2024-05-19", "2024-05-20", "2024-05-21"]
    )
    city: str = Field(
        default="austin",
        description="Город для прогноза (austin, london, tokyo, sydney)"
    )
    include_confidence: bool = Field(
        default=True,
        description="Включать доверительные интервалы"
    )

class PredictionItem(BaseModel):
    date: str
    temperature_c: float
    temperature_f: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_type: str

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    metadata: Dict[str, Any]

class HistoricalWeatherItem(BaseModel):
    date: str
    temperature_2m_max: Optional[float] = None
    temperature_2m_min: Optional[float] = None
    temperature_2m_mean: Optional[float] = None
    precipitation_sum: Optional[float] = None
    windspeed_10m_max: Optional[float] = None
    weather_code: Optional[int] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None

class HistoricalWeatherResponse(BaseModel):
    city: str
    coordinates: Dict[str, float]
    timezone: str
    days_requested: int
    historical_data: List[HistoricalWeatherItem]
    metadata: Dict[str, Any]

class WeeklyForecastRequest(BaseModel):
    city: str = Field(
        default="austin",
        description="Город для прогноза"
    )
    historical_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Количество исторических дней для анализа (1-30)"
    )
    forecast_days: int = Field(
        default=7,
        ge=1,
        le=14,
        description="Количество дней для прогноза (1-14)"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Уровень доверия для интервалов (0.5-0.99)"
    )

class WeeklyForecastItem(BaseModel):
    date: str
    temperature_c: float
    temperature_f: float
    confidence_interval: Dict[str, float]
    is_weekend: bool
    season: str

# ==================== ИСПРАВЛЕННЫЕ МОДЕЛИ ДЛЯ WEEKLY FORECAST ====================
class PeriodInfo(BaseModel):
    start: str
    end: str
    days: int  # Это должно быть int, а не str

class WeeklyForecastResponse(BaseModel):
    city: str
    historical_period: PeriodInfo
    forecast_period: PeriodInfo
    historical_stats: Dict[str, Any]
    forecast: List[WeeklyForecastItem]
    metadata: Dict[str, Any]
def get_city_month_avg_temp(city: str, month: int) -> float:
    """Получение средней температуры для города и месяца"""
    # Средние температуры на основе климатических данных
    avg_temps = {
        'austin': {
            1: 10.6, 2: 12.8, 3: 16.7, 4: 20.6, 5: 24.4, 6: 27.8,
            7: 29.4, 8: 29.7, 9: 26.7, 10: 21.7, 11: 16.1, 12: 11.7
        },
        'london': {
            1: 5.2, 2: 5.3, 3: 7.4, 4: 9.9, 5: 13.3, 6: 16.2,
            7: 18.3, 8: 18.0, 9: 15.5, 10: 11.9, 11: 8.0, 12: 5.5
        },
        'tokyo': {
            1: 5.2, 2: 5.7, 3: 8.7, 4: 14.1, 5: 18.7, 6: 21.8,
            7: 25.4, 8: 26.9, 9: 23.3, 10: 17.5, 11: 12.1, 12: 7.6
        },
        'sydney': {
            1: 22.8, 2: 22.8, 3: 21.4, 4: 18.4, 5: 15.4, 6: 13.0,
            7: 12.3, 8: 13.4, 9: 15.6, 10: 18.3, 11: 20.1, 12: 21.8
        }
    }
    
    return avg_temps.get(city.lower(), {}).get(month, 20.0)  # 20°C по умолчанию

async def get_historical_temperatures_for_city(city: str, days: int = 30) -> Dict[str, float]:
    """Получение реальных исторических температур для города"""
    try:
        lat, lon, timezone = get_city_coordinates(city)
        
        # Рассчитываем даты
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean",
            "timezone": timezone or "auto"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "daily" in data and "temperature_2m_mean" in data["daily"]:
                dates = data["daily"]["time"]
                temps = data["daily"]["temperature_2m_mean"]
                
                # Создаем словарь дата -> температура
                temp_dict = {}
                for date_str, temp in zip(dates, temps):
                    if temp is not None:
                        temp_dict[date_str] = float(temp)
                
                logger.info(f"✅ Got {len(temp_dict)} historical temperatures for {city}")
                return temp_dict
            
            logger.warning(f"No temperature data found for {city}")
            return {}
            
    except Exception as e:
        logger.error(f"Error fetching historical temperatures for {city}: {e}")
        return {}
# ==================== ФУНКЦИИ ДЛЯ СОЗДАНИЯ ПРИЗНАКОВ ====================
def create_features_for_date(date_str: str, city: str = "austin", 
                           historical_temps: Dict[str, float] = None):
    """Создание признаков с использованием реальных исторических температур"""
    try:
        dt = pd.to_datetime(date_str)
        
        # Создаем базовые признаки
        features = {}
        
        # 1. Временные признаки (как раньше)
        features.update({
            'year': float(dt.year),
            'month': float(dt.month),
            'day': float(dt.day),
            'dayofweek': float(dt.dayofweek),
            'dayofyear': float(dt.dayofyear),
            'quarter': float((dt.month - 1) // 3 + 1),
            'hour': 12.0,
            'is_weekend': 1.0 if dt.dayofweek >= 5 else 0.0,
            'is_summer': 1.0 if dt.month in [6, 7, 8] else 0.0,
            'is_winter': 1.0 if dt.month in [12, 1, 2] else 0.0,
            'season': float(((dt.month % 12 + 3) // 3)),
        })
        
        # 2. Циклические признаки
        features.update({
            'month_sin': float(np.sin(2 * np.pi * dt.month / 12)),
            'month_cos': float(np.cos(2 * np.pi * dt.month / 12)),
            'dayofyear_sin': float(np.sin(2 * np.pi * dt.dayofyear / 365.25)),
            'dayofyear_cos': float(np.cos(2 * np.pi * dt.dayofyear / 365.25)),
        })
        
        # 3. **КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Реальные исторические температуры**
        if historical_temps:
            # Получаем текущую дату и вычисляем лаговые даты
            current_date = dt.strftime("%Y-%m-%d")
            
            # Заполняем Temperature_C (текущая/вчерашняя температура)
            # Если нет данных на текущую дату, используем ближайшую доступную
            if current_date in historical_temps:
                features['Temperature_C'] = historical_temps[current_date]
            else:
                # Ищем ближайшую доступную дату
                available_dates = list(historical_temps.keys())
                if available_dates:
                    # Берем последнюю доступную дату
                    last_date = max(available_dates)
                    features['Temperature_C'] = historical_temps[last_date]
                else:
                    # Fallback: средняя температура для города и месяца
                    features['Temperature_C'] = get_city_month_avg_temp(city, dt.month)
            
            # Заполняем лаговые признаки реальными данными
            for lag in [1, 2, 3, 7, 14]:
                lag_date = (dt - timedelta(days=lag)).strftime("%Y-%m-%d")
                lag_feature = f'Temperature_C_lag_{lag}d'
                
                if lag_date in historical_temps:
                    features[lag_feature] = historical_temps[lag_date]
                else:
                    # Если нет данных, используем скользящее среднее
                    features[lag_feature] = features['Temperature_C']
            
            # Добавляем скользящие средние на основе реальных данных
            if len(historical_temps) >= 3:
                recent_temps = list(historical_temps.values())[-3:]
                features['temperature_rolling_3d_avg'] = np.mean(recent_temps)
            
            if len(historical_temps) >= 7:
                recent_temps = list(historical_temps.values())[-7:]
                features['temperature_rolling_7d_avg'] = np.mean(recent_temps)
                
        else:
            # Fallback: синтетические данные (старый подход)
            logger.warning(f"No historical temperatures provided for {city}, using synthetic data")
            base_temp = get_city_month_avg_temp(city, dt.month)
            features['Temperature_C'] = base_temp
            
            for lag in [1, 2, 3, 7, 14]:
                features[f'Temperature_C_lag_{lag}d'] = base_temp
        
        # 4. Корректировка для южного полушария
        if city.lower() == 'sydney':
            features['is_summer'] = 1.0 if dt.month in [12, 1, 2] else 0.0
            features['is_winter'] = 1.0 if dt.month in [6, 7, 8] else 0.0
        
        # Создаем DataFrame
        features_df = pd.DataFrame([features])
        
        # 5. Добавляем недостающие признаки для модели
        if feature_names:
            for feature in feature_names:
                if feature not in features_df.columns:
                    # Для температурных признаков используем текущую температуру
                    if 'temp' in feature.lower():
                        features_df[feature] = features.get('Temperature_C', 20.0)
                    else:
                        features_df[feature] = 0.0
            
            # Упорядочиваем как в модели
            if hasattr(model, 'feature_names_in_'):
                features_df = features_df[model.feature_names_in_]
            else:
                features_df = features_df[feature_names]
        
        logger.debug(f"Created features for {date_str} in {city}")
        logger.debug(f"Using historical temps: {historical_temps is not None}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        raise
def get_season_name(month: int) -> str:
    """Получение названия сезона по месяцу"""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

# ==================== OPEN-METEO API ФУНКЦИИ ====================
def get_city_coordinates(city_name: str) -> Tuple[float, float, str]:
    """Получение координат города по его имени"""
    city_lower = city_name.lower()
    
    if city_lower in CITY_COORDINATES:
        city_data = CITY_COORDINATES[city_lower]
        return city_data["lat"], city_data["lon"], city_data["timezone"]
    
    # Если город не найден в списке, возвращаем координаты Остина по умолчанию
    logger.warning(f"City '{city_name}' not found in database, using Austin coordinates")
    default_city = CITY_COORDINATES["austin"]
    return default_city["lat"], default_city["lon"], default_city["timezone"]

async def fetch_historical_weather(
    city: str, 
    days: int = 7,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    timezone: Optional[str] = None
) -> Dict[str, Any]:
    """Получение исторических данных погоды через Open-Meteo API"""
    
    # Получаем координаты города
    if lat is None or lon is None:
        lat, lon, timezone = get_city_coordinates(city)
    
    # Рассчитываем даты
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days-1)
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max,weathercode,sunrise,sunset",
        "timezone": timezone or "auto"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"🌤️  Fetching weather data for {city} ({lat}, {lon})")
            response = await client.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Проверяем наличие данных
            if "daily" not in data:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No weather data found for {city}"
                )
            
            return data
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching weather data: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Weather API error: {str(e)}"
        )
    except httpx.RequestError as e:
        logger.error(f"Request error fetching weather data: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to weather service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

def process_historical_weather_data(raw_data: Dict[str, Any], city: str) -> List[HistoricalWeatherItem]:
    """Обработка сырых данных погоды в структурированный формат"""
    daily_data = raw_data.get("daily", {})
    
    if not daily_data:
        return []
    
    # Получаем списки данных
    dates = daily_data.get("time", [])
    temp_max = daily_data.get("temperature_2m_max", [])
    temp_min = daily_data.get("temperature_2m_min", [])
    temp_mean = daily_data.get("temperature_2m_mean", [])
    precipitation = daily_data.get("precipitation_sum", [])
    windspeed = daily_data.get("windspeed_10m_max", [])
    weather_code = daily_data.get("weathercode", [])
    sunrise = daily_data.get("sunrise", [])
    sunset = daily_data.get("sunset", [])
    
    historical_data = []
    
    for i, date in enumerate(dates):
        item = HistoricalWeatherItem(
            date=date,
            temperature_2m_max=temp_max[i] if i < len(temp_max) else None,
            temperature_2m_min=temp_min[i] if i < len(temp_min) else None,
            temperature_2m_mean=temp_mean[i] if i < len(temp_mean) else None,
            precipitation_sum=precipitation[i] if i < len(precipitation) else None,
            windspeed_10m_max=windspeed[i] if i < len(windspeed) else None,
            weather_code=weather_code[i] if i < len(weather_code) else None,
            sunrise=sunrise[i] if i < len(sunrise) else None,
            sunset=sunset[i] if i < len(sunset) else None
        )
        historical_data.append(item)
    
    return historical_data

# ==================== ФУНКЦИИ ДЛЯ ПРОГНОЗА НА ОСНОВЕ ИСТОРИЧЕСКИХ ДАННЫХ ====================
def calculate_confidence_interval(
    predictions: List[float],
    historical_temps: List[float],
    confidence_level: float = 0.95
) -> List[Dict[str, float]]:
    """Расчет доверительных интервалов на основе исторической волатильности"""
    if not historical_temps:
        return [{"lower": p - 2.5, "upper": p + 2.5, "uncertainty": 2.5} for p in predictions]
    
    # Рассчитываем стандартное отклонение исторических температур
    std_dev = np.std(historical_temps)
    if std_dev == 0:
        std_dev = 2.5  # Значение по умолчанию если нет изменчивости
    
    # Для RandomForest можно использовать стандартную ошибку
    # Для упрощения используем t-распределение
    n = len(historical_temps)
    if n > 1:
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_of_error = t_value * std_dev / np.sqrt(n)
    else:
        margin_of_error = 2.5
    
    intervals = []
    for pred in predictions:
        intervals.append({
            "lower": round(pred - margin_of_error, 2),
            "upper": round(pred + margin_of_error, 2),
            "uncertainty": round(margin_of_error, 2)
        })
    
    return intervals

def analyze_historical_trend(historical_data: List[HistoricalWeatherItem]) -> Dict[str, Any]:
    """Анализ исторических данных для выявления трендов"""
    if not historical_data:
        return {}
    
    temps = [d.temperature_2m_mean for d in historical_data if d.temperature_2m_mean is not None]
    dates = [pd.to_datetime(d.date) for d in historical_data]
    
    if len(temps) < 2:
        return {
            "avg_temperature": round(float(temps[0]), 2) if temps else None,
            "trend": "insufficient_data"
        }
    
    # Рассчитываем статистики
    stats_dict = {
        "avg_temperature": round(float(np.mean(temps)), 2),
        "min_temperature": round(float(np.min(temps)), 2),
        "max_temperature": round(float(np.max(temps)), 2),
        "std_deviation": round(float(np.std(temps)), 2),
        "data_points": len(temps)
    }
    
    # Определяем тренд
    if len(temps) >= 3:
        x = np.arange(len(temps))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, temps)
        
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        stats_dict.update({
            "trend": trend,
            "trend_slope": round(float(slope), 3),
            "correlation": round(float(r_value), 3)
        })
    else:
        stats_dict["trend"] = "insufficient_data"
    
    return stats_dict

def generate_future_dates(start_date: datetime, days: int) -> List[str]:
    """Генерация списка будущих дат"""
    dates = []
    for i in range(days):
        future_date = start_date + timedelta(days=i+1)
        dates.append(future_date.strftime("%Y-%m-%d"))
    return dates

async def predict_weekly_forecast(
    city: str,
    historical_days: int = 7,
    forecast_days: int = 7,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Прогноз на неделю на основе исторических данных"""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs."
        )
    
    # Получаем исторические данные
    raw_data = await fetch_historical_weather(city=city, days=historical_days)
    historical_data = process_historical_weather_data(raw_data, city)
    
    if not historical_data:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data available for {city}"
        )
    
    # Анализируем исторические данные
    historical_stats = analyze_historical_trend(historical_data)
    historical_temps = [d.temperature_2m_mean for d in historical_data if d.temperature_2m_mean is not None]
    
    # Генерируем даты для прогноза
    last_date = pd.to_datetime(historical_data[-1].date)
    forecast_dates = generate_future_dates(last_date, forecast_days)
    
    # Делаем прогнозы
    predictions = []
    for date_str in forecast_dates:
        try:
            # Создаем признаки
            features_df = create_features_for_date(date_str)
            
            # Делаем прогноз
            temp_c = float(model.predict(features_df)[0])
            
            # Ограничиваем реалистичными значениями
            temp_c = max(-20.0, min(40.0, temp_c))
            
            dt = pd.to_datetime(date_str)
            is_weekend = dt.dayofweek >= 5
            season = get_season_name(dt.month)
            
            predictions.append({
                "date": date_str,
                "temperature_c": temp_c,
                "temperature_f": temp_c * 9/5 + 32,
                "is_weekend": is_weekend,
                "season": season
            })
            
        except Exception as e:
            logger.error(f"Error predicting for {date_str}: {e}")
            # Используем среднюю историческую температуру в случае ошибки
            avg_temp = np.mean(historical_temps) if historical_temps else 22.5
            dt = pd.to_datetime(date_str)
            is_weekend = dt.dayofweek >= 5
            season = get_season_name(dt.month)
            
            predictions.append({
                "date": date_str,
                "temperature_c": avg_temp,
                "temperature_f": avg_temp * 9/5 + 32,
                "is_weekend": is_weekend,
                "season": season,
                "error": str(e)[:100]
            })
    
    # Рассчитываем доверительные интервалы
    pred_temps = [p["temperature_c"] for p in predictions]
    intervals = calculate_confidence_interval(pred_temps, historical_temps, confidence_level)
    
    # Формируем итоговый ответ
    forecast_items = []
    for i, pred in enumerate(predictions):
        forecast_items.append(WeeklyForecastItem(
            date=pred["date"],
            temperature_c=round(pred["temperature_c"], 2),
            temperature_f=round(pred["temperature_f"], 2),
            confidence_interval=intervals[i] if i < len(intervals) else {
                "lower": round(pred["temperature_c"] - 2.5, 2),
                "upper": round(pred["temperature_c"] + 2.5, 2),
                "uncertainty": 2.5
            },
            is_weekend=pred["is_weekend"],
            season=pred["season"]
        ))
    
    # Получаем координаты города
    lat, lon, timezone = get_city_coordinates(city)
    
    # Формируем периоды
    historical_start = historical_data[0].date
    historical_end = historical_data[-1].date
    forecast_start = forecast_dates[0]
    forecast_end = forecast_dates[-1]
    
    return {
        "city": city,
        "historical_period": PeriodInfo(
            start=historical_start,
            end=historical_end,
            days=historical_days
        ),
        "forecast_period": PeriodInfo(
            start=forecast_start,
            end=forecast_end,
            days=forecast_days
        ),
        "historical_stats": historical_stats,
        "forecast": forecast_items,
        "metadata": {
            "model_type": "dummy" if is_dummy_model else "production",
            "confidence_level": confidence_level,
            "coordinates": {"latitude": lat, "longitude": lon},
            "timezone": timezone,
            "prediction_time": datetime.now().isoformat(),
            "api_version": "1.0.0"
        }
    }

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    """Корневой endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "message": "Weather Forecast API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "model_loaded": model is not None,
        "model_type": "dummy" if is_dummy_model else "production",
        "model_status": model_status,
        "features_count": len(feature_names),
        "supported_features": feature_names[:10] if feature_names else [],
        "supported_cities": list(CITY_COORDINATES.keys())
    }

@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy" if model else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_type": "dummy" if is_dummy_model else "production",
        "features_count": len(feature_names),
        "service": "weather-forecast-api",
        "uptime": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Прогноз температуры с использованием реальных исторических данных"""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs."
        )
    
    predictions = []
    
    # **НОВОЕ: Получаем реальные исторические температуры**
    logger.info(f"🌡️ Fetching historical temperatures for {request.city}")
    historical_temps = await get_historical_temperatures_for_city(request.city, days=30)
    
    if not historical_temps:
        logger.warning(f"⚠️ No historical data for {request.city}, using synthetic features")
    
    for date in request.dates:
        try:
            # **ИЗМЕНЕНИЕ: Передаем реальные температуры в функцию создания признаков**
            features_df = create_features_for_date(
                date, 
                request.city, 
                historical_temps
            )
            
            # Прогноз
            temp_c = float(model.predict(features_df)[0])
            
            # Ограничение температуры
            temp_min, temp_max = 5.0, 40.0
            temp_c = max(temp_min, min(temp_max, temp_c))
            
            # Формирование ответа
            item = {
                "date": date,
                "temperature_c": round(temp_c, 2),
                "temperature_f": round(temp_c * 9/5 + 32, 2),
                "model_type": "production",
                "city": request.city,
                "used_historical_data": historical_temps is not None and len(historical_temps) > 0
            }
            
            # Доверительные интервалы
            if request.include_confidence:
                # **Улучшенные интервалы на основе волатильности исторических данных**
                if historical_temps:
                    historical_values = list(historical_temps.values())
                    std_dev = np.std(historical_values) if len(historical_values) > 1 else 2.5
                    uncertainty = min(5.0, max(1.0, std_dev * 1.5))
                else:
                    uncertainty = 3.0 if is_dummy_model else 2.5
                
                item["confidence_interval"] = {
                    "lower": round(temp_c - uncertainty, 2),
                    "upper": round(temp_c + uncertainty, 2),
                    "uncertainty": round(uncertainty, 2),
                    "based_on_historical_volatility": historical_temps is not None
                }
            
            predictions.append(item)
            
            logger.info(f"📅 Prediction for {date} in {request.city}: {temp_c:.1f}°C")
            
            # Логирование используемых данных
            if historical_temps and date in historical_temps:
                logger.debug(f"   Historical temp for {date}: {historical_temps[date]:.1f}°C")
            
        except Exception as e:
            logger.error(f"Error predicting for {date}: {e}")
            predictions.append({
                "date": date,
                "city": request.city,
                "temperature_c": 22.5,
                "temperature_f": 72.5,
                "model_type": "error_fallback",
                "error": str(e)[:100]
            })
    
    # Метаданные с информацией об используемых данных
    metadata = {
        "model_type": "production",
        "city": request.city,
        "prediction_time": datetime.now().isoformat(),
        "total_dates": len(request.dates),
        "successful_predictions": len([p for p in predictions if p.get("model_type") != "error_fallback"]),
        "features_used": len(feature_names),
        "used_historical_temperatures": historical_temps is not None and len(historical_temps) > 0,
        "historical_data_points": len(historical_temps) if historical_temps else 0,
        "historical_data_period": f"Last {len(historical_temps)} days" if historical_temps else "None",
        "api_version": "2.0.0"  # Обновляем версию API
    }
    
    return PredictionResponse(
        predictions=predictions,
        metadata=metadata
    )

@app.get("/weather/historical-temperatures/{city}")
async def get_historical_temperatures_endpoint(
    city: str,
    days: int = Query(14, ge=1, le=90, description="Количество дней истории")
):
    """Получение реальных исторических температур"""
    temps = await get_historical_temperatures_for_city(city, days)
    
    if not temps:
        raise HTTPException(
            status_code=404,
            detail=f"No historical temperature data available for {city}"
        )
    
    # Статистика
    temp_values = list(temps.values())
    stats = {
        "avg": round(np.mean(temp_values), 2),
        "min": round(np.min(temp_values), 2),
        "max": round(np.max(temp_values), 2),
        "std": round(np.std(temp_values), 2),
        "count": len(temp_values)
    }
    
    return {
        "city": city,
        "days_requested": days,
        "days_available": len(temps),
        "date_range": {
            "start": min(temps.keys()),
            "end": max(temps.keys())
        },
        "statistics": stats,
        "temperatures": temps,
        "fetched_at": datetime.now().isoformat()
    }
@app.get("/predict/single/{city}/{date}")
async def predict_single(city: str, date: str):
    """Упрощенный endpoint для прогноза на одну дату для конкретного города"""
    try:
        request = PredictionRequest(
            dates=[date],
            city=city,
            include_confidence=True
        )
        
        response = await predict(request)
        return response.predictions[0] if response.predictions else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing date {date} for city {city}: {str(e)}")

@app.post("/predict/weekly", response_model=WeeklyForecastResponse)
async def predict_weekly(request: WeeklyForecastRequest):
    """Прогноз температуры на следующую неделю на основе исторических данных"""
    
    logger.info(f"🌍 Starting weekly forecast for {request.city}")
    logger.info(f"📊 Historical days: {request.historical_days}, Forecast days: {request.forecast_days}")
    
    result = await predict_weekly_forecast(
        city=request.city,
        historical_days=request.historical_days,
        forecast_days=request.forecast_days,
        confidence_level=request.confidence_level
    )
    
    return WeeklyForecastResponse(**result)

@app.get("/predict/weekly/simple/{city}")
async def predict_weekly_simple(
    city: str = "austin",
    historical_days: int = Query(7, ge=1, le=30),
    forecast_days: int = Query(7, ge=1, le=14)
):
    """Упрощенный endpoint для прогноза на неделю"""
    try:
        request = WeeklyForecastRequest(
            city=city,
            historical_days=historical_days,
            forecast_days=forecast_days
        )
        
        response = await predict_weekly(request)
        
        # Формируем упрощенный ответ
        simplified_response = {
            "city": response.city,
            "forecast_period": {
                "start": response.forecast_period.start,
                "end": response.forecast_period.end,
                "days": response.forecast_period.days
            },
            "predictions": [
                {
                    "date": item.date,
                    "temperature_c": item.temperature_c,
                    "temperature_f": item.temperature_f,
                    "confidence_lower": item.confidence_interval["lower"],
                    "confidence_upper": item.confidence_interval["upper"]
                }
                for item in response.forecast
            ],
            "historical_stats": response.historical_stats,
            "metadata": response.metadata
        }
        
        return simplified_response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating weekly forecast: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Информация о загруженной модели"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_type = type(model).__name__
    
    # Загружаем информацию из HPO модели если доступно
    hpo_model_path = "models/api_ready/hpo_fastapi_model.pkl"
    additional_info = {}
    
    if os.path.exists(hpo_model_path):
        try:
            import pickle
            with open(hpo_model_path, 'rb') as f:
                hpo_data = pickle.load(f)
            
            if isinstance(hpo_data, dict):
                additional_info = {
                    "hpo_metrics": hpo_data.get('original_hpo_metrics', {}),
                    "city_metadata": list(hpo_data.get('city_metadata', {}).keys()),
                    "temperature_ranges": hpo_data.get('temperature_ranges', {}),
                    "model_info": hpo_data.get('model_info', {})
                }
        except:
            pass
    
    model_info = {
        "model_type": model_type,
        "is_dummy": is_dummy_model,
        "features_count": len(feature_names),
        "features": feature_names,
        "description": "Dummy model for testing" if is_dummy_model else "Production HPO model",
        "converted_model": not is_dummy_model and "api_ready" in hpo_model_path,
        **additional_info
    }
    
    # Дополнительная информация для LightGBM
    if model_type == "LGBMRegressor":
        model_info.update({
            "n_estimators": getattr(model, 'n_estimators_', 'unknown'),
            "num_leaves": getattr(model, 'num_leaves_', 'unknown'),
            "feature_importances": dict(zip(feature_names, model.feature_importances_.tolist())) 
            if hasattr(model, 'feature_importances_') else {}
        })
    elif model_type == "RandomForestRegressor":
        model_info.update({
            "n_estimators": getattr(model, 'n_estimators', 'unknown'),
            "max_depth": getattr(model, 'max_depth', 'unknown'),
            "model_params": model.get_params() if hasattr(model, 'get_params') else {}
        })
    
    return model_info
# Добавьте этот endpoint для отладки
@app.get("/debug/features/{city}/{date}")
async def debug_features(city: str, date: str):
    """Отладочный endpoint для проверки признаков"""
    try:
        features_df = create_features_for_date(date, city)
        
        # Получим первые 15 признаков
        features_dict = features_df.iloc[0].to_dict()
        
        # Проверим заполнение признаков
        non_zero_features = {k: v for k, v in features_dict.items() if v != 0}
        zero_features = {k: v for k, v in features_dict.items() if v == 0}
        
        return {
            "date": date,
            "city": city,
            "features_shape": features_df.shape,
            "features_columns": list(features_df.columns),
            "non_zero_features": non_zero_features,
            "non_zero_count": len(non_zero_features),
            "zero_features_count": len(zero_features),
            "temperature_features": {k: v for k, v in features_dict.items() if 'temp' in k.lower()}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("/features/example/{date}")
async def get_features_example(date: str):
    """Пример признаков для конкретной даты"""
    try:
        features_df = create_features_for_date(date)
        features_dict = features_df.iloc[0].to_dict()
        
        return {
            "date": date,
            "features": features_dict,
            "features_count": len(features_dict),
            "feature_names": list(features_df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== НОВЫЕ ЭНДПОИНТЫ ДЛЯ ИСТОРИЧЕСКИХ ДАННЫХ ====================
@app.get("/weather/historical/{city}", response_model=HistoricalWeatherResponse)
async def get_historical_weather(
    city: str,
    days: int = Query(7, ge=1, le=30, description="Количество дней для получения данных (1-30)"),
    lat: Optional[float] = Query(None, description="Широта (переопределяет город)"),
    lon: Optional[float] = Query(None, description="Долгота (переопределяет город)")
):
    """Получение исторических данных погоды за последние N дней для города"""
    
    # Получаем координаты города если не указаны явно
    if lat is None or lon is None:
        city_lat, city_lon, timezone = get_city_coordinates(city)
    else:
        city_lat, city_lon = lat, lon
        timezone = "auto"
    
    # Получаем сырые данные
    raw_data = await fetch_historical_weather(
        city=city,
        days=days,
        lat=city_lat,
        lon=city_lon,
        timezone=timezone
    )
    
    # Обрабатываем данные
    historical_data = process_historical_weather_data(raw_data, city)
    
    # Метаданные ответа
    metadata = {
        "data_source": "Open-Meteo API",
        "request_time": datetime.now().isoformat(),
        "days_requested": days,
        "api_version": "1.0.0"
    }
    
    return HistoricalWeatherResponse(
        city=city,
        coordinates={"latitude": city_lat, "longitude": city_lon},
        timezone=timezone,
        days_requested=days,
        historical_data=historical_data,
        metadata=metadata
    )

@app.get("/weather/cities/supported")
async def get_supported_cities():
    """Получение списка поддерживаемых городов"""
    cities_info = []
    for city_name, city_data in CITY_COORDINATES.items():
        cities_info.append({
            "name": city_name.title(),
            "latitude": city_data["lat"],
            "longitude": city_data["lon"],
            "timezone": city_data["timezone"]
        })
    
    return {
        "supported_cities": cities_info,
        "total_cities": len(cities_info)
    }

@app.get("/weather/current/{city}")
async def get_current_weather(city: str):
    """Получение текущей погоды для города"""
    try:
        # Получаем исторические данные за сегодня
        raw_data = await fetch_historical_weather(city=city, days=1)
        historical_data = process_historical_weather_data(raw_data, city)
        
        if not historical_data:
            raise HTTPException(
                status_code=404,
                detail=f"No weather data available for {city}"
            )
        
        # Берем данные за сегодня
        today_data = historical_data[0]
        
        # Получаем координаты города
        lat, lon, timezone = get_city_coordinates(city)
        
        return {
            "city": city,
            "date": today_data.date,
            "temperature": {
                "max_c": today_data.temperature_2m_max,
                "min_c": today_data.temperature_2m_min,
                "mean_c": today_data.temperature_2m_mean,
                "max_f": today_data.temperature_2m_max * 9/5 + 32 if today_data.temperature_2m_max else None,
                "min_f": today_data.temperature_2m_min * 9/5 + 32 if today_data.temperature_2m_min else None
            },
            "precipitation_mm": today_data.precipitation_sum,
            "windspeed_kmh": today_data.windspeed_10m_max,
            "coordinates": {"latitude": lat, "longitude": lon},
            "timezone": timezone,
            "sunrise": today_data.sunrise,
            "sunset": today_data.sunset,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching current weather: {str(e)}"
        )

@app.get("/weather/compare/{city1}/{city2}")
async def compare_cities_weather(
    city1: str,
    city2: str,
    days: int = Query(7, ge=1, le=14, description="Количество дней для сравнения")
):
    """Сравнение погоды в двух городах"""
    try:
        # Получаем данные для обоих городов параллельно
        task1 = fetch_historical_weather(city=city1, days=days)
        task2 = fetch_historical_weather(city=city2, days=days)
        
        raw_data1, raw_data2 = await asyncio.gather(task1, task2)
        
        # Обрабатываем данные
        data1 = process_historical_weather_data(raw_data1, city1)
        data2 = process_historical_weather_data(raw_data2, city2)
        
        # Получаем координаты городов
        lat1, lon1, tz1 = get_city_coordinates(city1)
        lat2, lon2, tz2 = get_city_coordinates(city2)
        
        # Сравниваем средние температуры
        avg_temp1 = np.mean([d.temperature_2m_mean for d in data1 if d.temperature_2m_mean is not None])
        avg_temp2 = np.mean([d.temperature_2m_mean for d in data2 if d.temperature_2m_mean is not None])
        
        return {
            "cities": [
                {"name": city1, "latitude": lat1, "longitude": lon1, "timezone": tz1},
                {"name": city2, "latitude": lat2, "longitude": lon2, "timezone": tz2}
            ],
            "comparison_period": f"Last {days} days",
            "average_temperatures": {
                city1: round(float(avg_temp1), 2) if not np.isnan(avg_temp1) else None,
                city2: round(float(avg_temp2), 2) if not np.isnan(avg_temp2) else None,
                "difference": round(float(avg_temp1 - avg_temp2), 2) if not (np.isnan(avg_temp1) or np.isnan(avg_temp2)) else None
            },
            "city1_data": data1,
            "city2_data": data2,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error comparing cities: {str(e)}"
        )

# ==================== ОБРАБОТЧИКИ ОШИБОК ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик общих исключений"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("API_DEBUG", "false").lower() == "true" else None,
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

# ==================== ЗАПУСК ПРИЛОЖЕНИЯ ====================
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )