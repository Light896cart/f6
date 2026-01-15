"""
Назначение:
Обучение LightGBM с early stopping и валидацией.
Статус: Актуальный — используется в model/stacking.py и model/oof_stacking.py.
"""

import lightgbm as lgb

def get_lgb_model(X_train, X_val, y_train, y_val, **kwargs):
    """
    Обучает LGBMClassifier с early stopping.
    """
    default_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_iterations': 1000,
        'learning_rate': 0.05,
        'max_depth': 3,
        'num_leaves': 31,
        'class_weight': 'balanced',
        'random_state': 42,
        'verbosity': -1
    }
    params = {**default_params, **kwargs}

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[valid_data],
        valid_names=['val'],
        num_boost_round=params['num_iterations'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)  # отключаем лог
        ]
    )
    return model