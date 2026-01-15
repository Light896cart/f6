"""Назначение:
Дублирует model_lgb.py и model_xgb.py — те же функции get_lgb_model, get_xgb_model.
Проблема:

Использует max_depth=6 (в отличие от max_depth=3 в других файлах).
Нигде не импортируется в других скриптах.
Статус: Устарел"""

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

def get_lgb_model(X_train, X_val, y_train, y_val, **kwargs):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_iterations': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'class_weight': 'balanced',
        'random_state': 42,
        'verbosity': -1
    }
    params.update(kwargs)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    return lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[valid_data],
        num_boost_round=params['num_iterations'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

def get_xgb_model(X_train, X_val, y_train, y_val, **kwargs):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'scale_pos_weight': 5,
        'random_state': 42,
        'verbosity': 0
    }
    params.update(kwargs)
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)