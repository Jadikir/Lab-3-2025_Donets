# scripts/03_hpo_optimization.py
import pandas as pd
import numpy as np
import yaml
import os
import json
import joblib
from datetime import datetime
from clearml import Task
import optuna
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import logging
from typing import Dict, Any

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherHPO:
    """Гиперпараметрическая оптимизация для модели погоды"""
    
    def __init__(self):
        # Инициализация задачи ClearML
        self.task = Task.init(
            project_name='WeatherForecast',
            task_name='Multi-City HPO Optimization',
            task_type=Task.TaskTypes.optimizer
        )
        
        # Загрузка конфигурации
        with open('config/default.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Пути
        self.data_dir = 'data/multi_city'
        self.models_dir = 'models/multi_city'
        self.hpo_dir = 'models/hpo'
        
        os.makedirs(self.hpo_dir, exist_ok=True)
        
        # Загрузка данных
        self.df = pd.read_parquet(f'{self.data_dir}/weather_multi_city.parquet')
        
        with open(f'{self.models_dir}/feature_list.json', 'r') as f:
            self.features = json.load(f)
        
        # Целевая переменная
        self.target_col = 'target_temp_1d'
        
        # Подготовка данных
        logger.info("=" * 70)
        logger.info("⚡ ЗАГРУЗКА ДАННЫХ ДЛЯ HPO")
        logger.info("=" * 70)
        
        X = self.df[self.features]
        y = self.df[self.target_col]
        
        # Очистка
        mask = y.notna() & X.notna().all(axis=1)
        self.X = X[mask]
        self.y = y[mask]
        
        logger.info(f"📊 Данные: {self.X.shape}")
        logger.info(f"🎯 Цель: {self.target_col}")
        
        # TimeSeries Split для HPO
        self.tscv = TimeSeriesSplit(n_splits=3)
        logger.info(f"🔄 TimeSeries CV: {self.tscv.n_splits} фолда")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации"""
        
        # Параметры для оптимизации
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'metric': 'mae'
        }
        
        # Кросс-валидация
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.X)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='mae',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                scores.append(mae)
                
                # Логирование в ClearML
                self.task.get_logger().report_scalar(
                    title='Fold MAE',
                    series=f'fold_{fold}',
                    value=mae,
                    iteration=trial.number
                )
                
            except Exception as e:
                logger.warning(f"Ошибка в фолде {fold}: {e}")
                return 100.0  # Большое значение как штраф
        
        avg_mae = np.mean(scores)
        
        # Логирование
        self.task.get_logger().report_scalar(
            title='HPO Results',
            series='Average MAE',
            value=avg_mae,
            iteration=trial.number
        )
        
        # Периодический вывод
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}: MAE = {avg_mae:.4f}°C")
        
        return avg_mae
    
    def run_hpo(self, n_trials: int = 50, timeout: int = 3600):
        """Запуск HPO"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 ЗАПУСК HPO С OPTUNA")
        logger.info("=" * 70)
        
        logger.info(f"⚙️  Параметры HPO:")
        logger.info(f"  • Количество trials: {n_trials}")
        logger.info(f"  • Таймаут: {timeout} секунд")
        logger.info(f"  • Целевая метрика: MAE (минимизация)")
        
        # Создание study
        study_name = f'weather_hpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        storage_url = f'sqlite:///{self.hpo_dir}/hpo_study.db'
        
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        logger.info(f"\n📊 Study: {study_name}")
        logger.info(f"💾 Storage: {storage_url}")
        logger.info(f"\n⏳ Начало оптимизации...")
        
        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                gc_after_trial=True
            )
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  HPO прервано пользователем")
        except Exception as e:
            logger.error(f"\n❌ Ошибка оптимизации: {e}")
        
        return study
    
    def train_final_model(self, study: optuna.study.Study):
        """Обучение финальной модели на лучших параметрах"""
        if not study.best_trial:
            logger.error("❌ Нет результатов HPO")
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("🤖 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
        logger.info("=" * 70)
        
        # Лучшие параметры
        best_params = study.best_params.copy()
        best_params.update({
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'metric': 'mae'
        })
        
        logger.info(f"\n🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
        for param, value in best_params.items():
            if param not in ['n_jobs', 'verbose', 'random_state', 'metric']:
                logger.info(f"  {param}: {value}")
        
        # Разделение на train/test
        split_idx = int(len(self.X) * 0.8)
        X_train = self.X.iloc[:split_idx]
        X_test = self.X.iloc[split_idx:]
        y_train = self.y.iloc[:split_idx]
        y_test = self.y.iloc[split_idx:]
        
        logger.info(f"\n📊 Разделение данных:")
        logger.info(f"  Train: {len(X_train)} записей")
        logger.info(f"  Test:  {len(X_test)} записей")
        
        # Обучение финальной модели
        final_model = lgb.LGBMRegressor(**best_params)
        
        logger.info("\n⏳ Обучение финальной модели...")
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='mae',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Оценка
        y_pred_test = final_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Baseline
        baseline_mae = mean_absolute_error(
            y_test, 
            X_test['Temperature_C'] if 'Temperature_C' in X_test.columns else np.mean(y_train)
        )
        improvement = ((baseline_mae - test_mae) / baseline_mae * 100)
        
        logger.info(f"\n📈 РЕЗУЛЬТАТЫ ФИНАЛЬНОЙ МОДЕЛИ:")
        logger.info(f"  Test MAE:        {test_mae:.4f}°C")
        logger.info(f"  Baseline MAE:    {baseline_mae:.4f}°C")
        logger.info(f"  Улучшение:       {improvement:.1f}%")
        logger.info(f"  Итераций:        {final_model.n_estimators_}")
        
        return final_model, test_mae, baseline_mae, improvement
    
    def save_results(self, study, final_model, test_mae, baseline_mae, improvement):
        """Сохранение результатов HPO"""
        logger.info("\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ HPO")
        
        # Сохраняем study
        study_path = f'{self.hpo_dir}/hpo_study.pkl'
        joblib.dump(study, study_path)
        logger.info(f"✅ Study сохранено: {study_path}")
        
        # Сохраняем лучшие параметры
        best_params_path = f'{self.hpo_dir}/best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        logger.info(f"✅ Лучшие параметры: {best_params_path}")
        
        # Сохраняем финальную модель
        model_data = {
            'model': final_model,
            'features': self.features,
            'target_column': self.target_col,
            'best_params': study.best_params,
            'hpo_results': {
                'best_trial': study.best_trial.number,
                'best_mae': study.best_trial.value,
                'test_mae': test_mae,
                'baseline_mae': baseline_mae,
                'improvement': improvement,
                'n_trials': len(study.trials)
            },
            'training_date': datetime.now().isoformat()
        }
        
        model_path = f'{self.hpo_dir}/hpo_optimized_model.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"✅ Оптимизированная модель: {model_path}")
        
        # Сохраняем результаты
        results = {
            'hpo_info': {
                'study_name': study.study_name,
                'best_trial': study.best_trial.number,
                'best_mae': float(study.best_trial.value),
                'n_trials_completed': len(study.trials),
                'duration_hours': study.best_trial.duration.total_seconds() / 3600
            },
            'model_performance': {
                'test_mae': float(test_mae),
                'baseline_mae': float(baseline_mae),
                'improvement': float(improvement)
            },
            'best_params': study.best_params,
            'features_used': len(self.features)
        }
        
        results_path = f'{self.hpo_dir}/hpo_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Результаты HPO: {results_path}")
        
        # Логируем в ClearML
        self.task.upload_artifact('best_params', study.best_params)
        self.task.upload_artifact('hpo_results', results)
        
        return model_path, results_path
    
    def create_visualizations(self, study):
        """Создание визуализаций HPO"""
        try:
            import plotly.io as pio
            
            logger.info("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ HPO")
            
            # 1. История оптимизации
            fig_history = optuna.visualization.plot_optimization_history(study)
            self.task.get_logger().report_plotly(
                title='Optimization History',
                series='HPO',
                figure=fig_history
            )
            
            # 2. Важность параметров
            fig_importance = optuna.visualization.plot_param_importances(study)
            self.task.get_logger().report_plotly(
                title='Parameter Importances',
                series='HPO',
                figure=fig_importance
            )
            
            # 3. Slice plot
            fig_slice = optuna.visualization.plot_slice(study)
            self.task.get_logger().report_plotly(
                title='Slice Plot',
                series='HPO',
                figure=fig_slice
            )
            
            logger.info("✅ Визуализации созданы и загружены в ClearML")
            
        except Exception as e:
            logger.warning(f"⚠️  Ошибка создания визуализаций: {e}")
    
    def compare_with_baseline(self):
        """Сравнение с базовой моделью"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 СРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ")
        logger.info("=" * 70)
        
        # Загружаем базовую модель
        baseline_path = f'{self.models_dir}/weather_forecast_model.pkl'
        if os.path.exists(baseline_path):
            try:
                baseline_data = joblib.load(baseline_path)
                baseline_metrics = baseline_data.get('metrics', {})
                
                if 'test' in baseline_metrics:
                    baseline_mae = baseline_metrics['test']['mae']
                    
                    logger.info(f"\n📈 СРАВНЕНИЕ:")
                    logger.info(f"  Базовая модель (без HPO):")
                    logger.info(f"    • Test MAE: {baseline_mae:.4f}°C")
                    logger.info(f"    • Test R²:  {baseline_metrics['test'].get('r2', 'N/A'):.4f}")
                    
                    # Сравнение будет после HPO
                    logger.info(f"\n  HPO модель:")
                    logger.info(f"    • Результаты будут после оптимизации")
                    
            except Exception as e:
                logger.warning(f"⚠️  Не удалось загрузить базовую модель: {e}")
    
    def run(self):
        """Основной метод запуска"""
        try:
            # Сравнение с baseline
            self.compare_with_baseline()
            
            # Запуск HPO
            study = self.run_hpo(
                n_trials=self.config.get('hpo', {}).get('n_trials', 30),
                timeout=self.config.get('hpo', {}).get('timeout', 1800)
            )
            
            if not study.best_trial:
                logger.error("❌ HPO не дало результатов")
                self.task.close()
                return
            
            # Обучение финальной модели
            final_model, test_mae, baseline_mae, improvement = self.train_final_model(study)
            
            if final_model is None:
                self.task.close()
                return
            
            # Сохранение результатов
            model_path, results_path = self.save_results(
                study, final_model, test_mae, baseline_mae, improvement
            )
            
            # Создание визуализаций
            self.create_visualizations(study)
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 HPO ОПТИМИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
            logger.info("=" * 70)
            
            logger.info(f"\n📊 ИТОГИ:")
            logger.info(f"  • Лучший MAE (CV): {study.best_trial.value:.4f}°C")
            logger.info(f"  • Test MAE:        {test_mae:.4f}°C")
            logger.info(f"  • Улучшение:       {improvement:.1f}%")
            logger.info(f"  • Trials:          {len(study.trials)}")
            
            logger.info(f"\n💾 СОХРАНЕННЫЕ ФАЙЛЫ:")
            logger.info(f"  • Модель: {model_path}")
            logger.info(f"  • Параметры: {self.hpo_dir}/best_params.json")
            logger.info(f"  • Результаты: {results_path}")
            
            self.task.close()
            
        except Exception as e:
            logger.error(f"❌ Ошибка HPO: {e}")
            import traceback
            traceback.print_exc()
            self.task.close()
            raise

if __name__ == "__main__":
    hpo = WeatherHPO()
    hpo.run()