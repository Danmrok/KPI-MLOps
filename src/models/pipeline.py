from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
import numpy as np


FEATURE_COLS = [
    'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
    'T', 'RH', 'AH'
]
TARGET_COL = 'CO_level'


def create_decision_tree_pipeline(
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = 'gini',
    random_state: int = 42
) -> Pipeline:

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        ))
    ])
    return pipeline


def create_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    random_state: int = 42
) -> Pipeline:

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline


def create_gradient_boosting_pipeline(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42
) -> Pipeline:

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    return pipeline


if __name__ == "__main__":
    pipe = create_decision_tree_pipeline()
    print("Decision Tree Pipeline steps:", list(pipe.named_steps.keys()))
    pipe_rf = create_random_forest_pipeline()
    print("Random Forest Pipeline steps:", list(pipe_rf.named_steps.keys()))
    pipe_gb = create_gradient_boosting_pipeline()
    print("Gradient Boosting Pipeline steps:", list(pipe_gb.named_steps.keys()))
