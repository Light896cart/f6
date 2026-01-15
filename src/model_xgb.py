"""
Назначение:
Обучение XGBoost с балансировкой классов (scale_pos_weight).
Статус: Актуальный — используется в стекингах.
"""

import xgboost as xgb

def get_xgb_model(X_train, X_val, y_train, y_val, **kwargs):
    """
    Обучает XGBClassifier
    """
    scale_pos_weight = len(y_train[y_train == 'G']) / len(y_train[y_train == 'F'])
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 3,
        'scale_pos_weight': scale_pos_weight,  # ≈ G/F ratio
        'random_state': 42,
        'verbosity': 0
    }
    params = {**default_params, **kwargs}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model