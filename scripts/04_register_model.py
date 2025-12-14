# scripts/04_register_model.py
from clearml import Task, OutputModel, Model
import joblib
import json
import yaml
from datetime import datetime
import os
import pandas as pd
import numpy as np

def register_hpo_model():
    """Регистрация HPO-оптимизированной модели в ClearML Model Registry"""
    
    # Инициализация задачи
    task = Task.init(
        project_name='WeatherForecast',
        task_name='Register HPO Model',
        task_type=Task.TaskTypes.custom
    )
    
    # Загрузка конфигурации
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("🏷️  РЕГИСТРАЦИЯ HPO-ОПТИМИЗИРОВАННОЙ МОДЕЛИ")
    print("=" * 70)
    
    # Пути к файлам HPO модели
    hpo_model_path = 'models/hpo/hpo_optimized_model.pkl'
    hpo_results_path = 'models/hpo/hpo_results.json'
    best_params_path = 'models/hpo/best_params.json'
    features_path = 'models/multi_city/feature_list.json'
    
    # Проверка существования файлов
    missing_files = []
    for file_path in [hpo_model_path, hpo_results_path, best_params_path, features_path]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Отсутствуют файлы:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nСначала запустите:")
        print("  1. python scripts/01_create_dataset.py")
        print("  2. python scripts/02_train_model.py")
        print("  3. python scripts/03_hpo_optimization.py")
        task.close()
        exit(1)
    
    # Загрузка HPO модели
    try:
        print("📦 ЗАГРУЗКА HPO МОДЕЛИ...")
        with open(hpo_model_path, 'rb') as f:
            model_data = joblib.load(f)
        
        model = model_data['model']
        features = model_data['features']
        hpo_results = model_data['hpo_results']
        
        print(f"✅ Модель загружена: {type(model).__name__}")
        print(f"📊 Признаков: {len(features)}")
        print(f"🎯 Целевая переменная: {model_data.get('target_column', 'target_temp_1d')}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        task.close()
        exit(1)
    
    # Загрузка результатов HPO
    try:
        with open(hpo_results_path, 'r') as f:
            hpo_results_full = json.load(f)
        
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        
        with open(features_path, 'r') as f:
            feature_list = json.load(f)
        
        print(f"📈 Результаты HPO загружены")
        print(f"⚙️  Лучших параметров: {len(best_params)}")
        
    except Exception as e:
        print(f"⚠️  Ошибка загрузки метаданных: {e}")
        hpo_results_full = {}
        best_params = {}
        feature_list = features
    
    # Реальные метрики из HPO
    metrics = {
        'test_mae': hpo_results.get('test_mae', hpo_results_full.get('model_performance', {}).get('test_mae', 0)),
        'cv_mae': hpo_results.get('best_mae', hpo_results_full.get('hpo_info', {}).get('best_mae', 0)),
        'baseline_mae': hpo_results.get('baseline_mae', hpo_results_full.get('model_performance', {}).get('baseline_mae', 0)),
        'improvement': hpo_results.get('improvement', hpo_results_full.get('model_performance', {}).get('improvement', 0)),
        'n_trials': hpo_results.get('n_trials', hpo_results_full.get('hpo_info', {}).get('n_trials_completed', 0)),
        'model_type': 'LightGBM',
        'locations': 'multi-city (Austin, London, Tokyo, Sydney)',
        'horizon': '1 day forecast'
    }
    
    # Логирование метрик в ClearML
    print("\n📊 ЛОГИРОВАНИЕ МЕТРИК В CLEARML...")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            task.get_logger().report_scalar(
                title='HPO Model Metrics',
                series=metric_name,
                value=metric_value,
                iteration=0
            )
            print(f"  ✅ {metric_name}: {metric_value}")
    
    # Создание OutputModel для регистрации
    print("\n🏷️  СОЗДАНИЕ OUTPUT MODEL...")
    output_model = OutputModel(
        task=task,
        name='WeatherForecast_HPO_MultiCity',
        framework='LightGBM',
        tags=['hpo-optimized', 'multi-city', 'weather', 'regression', 'time-series', 'production']
    )
    
    # Добавление модели (ОСНОВНОЙ ШАГ)
    output_model.update_weights(weights_filename=hpo_model_path)
    print(f"✅ Веса модели добавлены: {hpo_model_path}")
    
    # Установка метаданных
    metadata = {
        'model_type': 'LightGBM_HPO',
        'training_date': datetime.now().isoformat(),
        'features_count': len(features),
        'n_features_in': model.n_features_in_ if hasattr(model, 'n_features_in_') else len(features),
        'n_estimators': model.n_estimators_ if hasattr(model, 'n_estimators_') else 'unknown',
        'locations': ['austin', 'london', 'tokyo', 'sydney'],
        'target_column': model_data.get('target_column', 'target_temp_1d'),
        'horizon': '1d',
        'hpo_trials': metrics['n_trials'],
        'metrics': metrics,
        'best_params': best_params,
        'feature_list_sample': feature_list[:20] if feature_list else []
    }
    
    for key, value in metadata.items():
        output_model.set_metadata(key, value)
        print(f"  ✅ Метаданные: {key}")
    
    # Регистрация версии
    print("\n🚀 ПУБЛИКАЦИЯ МОДЕЛИ В MODEL REGISTRY...")
    output_model.publish()
    
    print("\n" + "=" * 70)
    print("✅ HPO МОДЕЛЬ УСПЕШНО ЗАРЕГИСТРИРОВАНА!")
    print("=" * 70)
    
    print(f"\n📋 ИНФОРМАЦИЯ О МОДЕЛИ:")
    print(f"  Model ID: {output_model.id}")
    print(f"  Название: WeatherForecast_HPO_MultiCity")
    print(f"  Версия: v1.0-hpo")
    print(f"  Теги: {output_model.tags}")
    
    print(f"\n📊 МЕТРИКИ:")
    print(f"  • Test MAE: {metrics['test_mae']:.4f}°C")
    print(f"  • CV MAE:   {metrics['cv_mae']:.4f}°C")
    print(f"  • Улучшение: {metrics['improvement']:.1f}%")
    print(f"  • HPO Trials: {metrics['n_trials']}")
    
    print(f"\n🏙️  ГОРОДА:")
    print(f"  • Austin, TX")
    print(f"  • London, UK")
    print(f"  • Tokyo, Japan")
    print(f"  • Sydney, Australia")
    
    print(f"\n⚙️  ПАРАМЕТРЫ МОДЕЛИ:")
    print(f"  • Признаков: {metadata['features_count']}")
    print(f"  • Деревьев: {metadata['n_estimators']}")
    print(f"  • Целевая: {metadata['target_column']}")
    
    print(f"\n🔗 ССЫЛКИ:")
    print(f"  Веб-интерфейс: http://localhost:8080/models/{output_model.id}")
    print(f"  Проект: http://localhost:8080/projects/{task.project}")
    
    print(f"\n💻 ИСПОЛЬЗОВАНИЕ В КОДЕ:")
    print(f'''from clearml import Model

# Загрузка модели по ID
model = Model(model_id="{output_model.id}")
model_path = model.get_local_copy()

# Использование
import joblib
with open(model_path, 'rb') as f:
    model_data = joblib.load(f)

model = model_data['model']
features = model_data['features']
''')
    
    # Создание тестового предсказания для демонстрации
    print(f"\n🧪 ТЕСТОВОЕ ПРЕДСКАЗАНИЕ:")
    try:
        # Создаем тестовые данные
        np.random.seed(42)
        n_samples = 5
        n_features = len(features)
        
        X_test = np.random.randn(n_samples, n_features)
        
        # Делаем предсказания
        predictions = model.predict(X_test)
        
        for i, pred in enumerate(predictions):
            print(f"  Образец {i+1}: {pred:.1f}°C")
        
        print(f"  Диапазон: {predictions.min():.1f} - {predictions.max():.1f}°C")
        print(f"  Среднее: {predictions.mean():.1f}°C")
        
    except Exception as e:
        print(f"  ⚠️  Тестовое предсказание не удалось: {e}")
    
    # Логирование дополнительной информации
    task.get_logger().report_text(f"Модель зарегистрирована в Model Registry: {output_model.id}")
    task.get_logger().report_text(f"Metrics: {metrics}")
    
    # Создание артефакта с информацией
    task.upload_artifact('model_info', {
        'model_id': output_model.id,
        'model_name': 'WeatherForecast_HPO_MultiCity',
        'metrics': metrics,
        'features_count': len(features),
        'hpo_trials': metrics['n_trials'],
        'registration_date': datetime.now().isoformat()
    })
    
    # Создание таблицы с лучшими параметрами
    if best_params:
        params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
        task.get_logger().report_table(
            title='Best HPO Parameters',
            series='parameters',
            table_plot=params_df
        )
    
    print("\n" + "=" * 70)
    print("🎯 РЕГИСТРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    
    task.close()
    
    return output_model.id

def verify_registration(model_id: str):
    """Проверка регистрации модели"""
    print(f"\n🔍 ПРОВЕРКА РЕГИСТРАЦИИ...")
    
    try:
        model = Model(model_id=model_id)
        print(f"✅ Модель найдена в реестре")
        print(f"   ID: {model.id}")
        print(f"   Название: {model.name}")
        print(f"   Теги: {model.tags}")
        print(f"   Статус: {model.status}")
        
        # Проверка метаданных
        metadata = model.get_metadata()
        if metadata:
            print(f"   Метаданные: {len(metadata)} полей")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки: {e}")
        return False

if __name__ == "__main__":
    # Регистрация модели
    model_id = register_hpo_model()
    
    # Проверка регистрации
    if model_id:
        verify_registration(model_id)