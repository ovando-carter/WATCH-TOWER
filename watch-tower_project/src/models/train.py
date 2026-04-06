from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def prepare_model_inputs(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = [
        "payments_count",
        "lifetime_months",
        "monthly_payment",
        "postcode",
        "class_day",
        "class_time",
        "class_type",
    ]
    available_features = [feature for feature in features if feature in model_df.columns]
    X = pd.get_dummies(model_df[available_features], drop_first=True)
    y = model_df["missed_payment"]
    return X, y



def build_logistic_coefficients(model: LogisticRegression, feature_names: Iterable[str]) -> pd.DataFrame:
    coefficients = pd.DataFrame({
        "feature": list(feature_names),
        "coefficient": model.coef_[0],
    })
    coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
    return coefficients.sort_values("coefficient", ascending=False).reset_index(drop=True)



def build_feature_importance(model: RandomForestClassifier, feature_names: Iterable[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=list(feature_names)).sort_values(ascending=False)



def train_models(model_df: pd.DataFrame, random_state: int, test_size: float) -> dict[str, object]:
    X, y = prepare_model_inputs(model_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    logistic_model = LogisticRegression(max_iter=1000, random_state=random_state)
    logistic_model.fit(X_train, y_train)

    random_forest_model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    random_forest_model.fit(X_train, y_train)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "logistic_report": classification_report(y_test, logistic_model.predict(X_test), output_dict=True),
        "random_forest_report": classification_report(y_test, random_forest_model.predict(X_test), output_dict=True),
        "logistic_coefficients": build_logistic_coefficients(logistic_model, X.columns),
        "rf_feature_importance": build_feature_importance(random_forest_model, X.columns),
    }



def chi_square_test(client_summary: pd.DataFrame) -> dict[str, object]:
    contingency_table = pd.crosstab(client_summary["postcode"], client_summary["missed_payment"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return {
        "contingency_table": contingency_table,
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "expected": expected,
    }
