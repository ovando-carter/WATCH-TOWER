
"""
Refactored Stripe analytics pipeline extracted from the notebook.

Goals:
- centralize imports and constants
- remove repeated logic
- isolate cleaning / feature engineering / modelling
- make the workflow reusable and testable
- keep the code notebook-friendly while being production-oriented
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

DAY_PATTERN = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\b",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:-\s*\d{1,2}(?::\d{2})?\s*)?(?:am|pm))\b",
    re.IGNORECASE,
)
OUTWARD_POSTCODE_PATTERN = re.compile(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)")

DAY_MAP = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}


@dataclass(frozen=True)
class PipelineConfig:
    checkout_path: Path = Path("checkout_sessions-8.csv")
    payment_links_path: Path = Path("payment_links-2.csv")
    payments_path: Path = Path("unified_payments-16.csv")
    valid_outward_postcodes: tuple[str, ...] = (
        "BR1", "BR2", "BR3",
        "CRO", "CR7",
        "SE4", "SE6",
        "SE13", "SE15", "SE19", "SE20", "SE21",
        "SE22", "SE23", "SE24", "SE25", "SE26",
        "SW2", "SW18",
    )
    removable_columns: tuple[str, ...] = (
        "Card CVC Status",
        "Shipping Name",
        "Shipping Address Line1",
        "Shipping Address Line2",
        "Shipping Address City",
        "Shipping Address State",
        "Shipping Address Country",
        "Shipping Address Postal Code",
        "Disputed Amount",
        "Dispute Date (UTC)",
        "Dispute Evidence Due (UTC)",
        "Dispute Reason",
        "Dispute Status",
        "Checkout Session ID",
        "Checkout Custom Field 1 Key",
        "Checkout Custom Field 1 Value",
        "Checkout Custom Field 2 Key",
        "Checkout Custom Field 2 Value",
        "Checkout Custom Field 3 Key",
        "Checkout Custom Field 3 Value",
        "Checkout Promotional Consent",
        "Checkout Terms of Service Consent",
        "Client Reference ID",
        "Payment Link ID",
        "UTM Campaign",
        "UTM Content",
        "UTM Medium",
        "UTM Source",
        "UTM Term",
        "Terminal Location ID",
        "Terminal Reader ID",
        "Application Fee",
        "Application ID",
        "Destination",
        "Transfer",
        "Transfer Group",
        "Billing Address (metadata)",
        "Customer Email (metadata)",
        "Customer Name (metadata)",
        "Customer Phone (metadata)",
        "wix_transaction_id (metadata)",
        "Shipping Address (metadata)",
        "wix_metasite_id (metadata)",
        "Invoice Id (metadata)",
    )
    random_state: int = 42
    test_size: float = 0.25


def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def coerce_datetimes(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def normalize_blank_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    replacements = {"": np.nan, "nan": np.nan, "None": np.nan}
    for column in columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .replace(replacements)
                .replace(r"^\s*$", np.nan, regex=True)
            )
    return df


def drop_empty_and_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            if not ((series.isna()) | (series == 0)).all():
                keep_columns.append(column)
        elif not series.isna().all():
            keep_columns.append(column)
    return df[keep_columns].copy()


def standardize_customer_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Customer Email" in df.columns:
        df["Customer Email"] = (
            df["Customer Email"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )

    if "Customer Name" in df.columns:
        df["Customer Name"] = (
            df["Customer Name"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )

    return df


def generate_client_id(row: pd.Series) -> Optional[str]:
    email = row.get("Customer Email")
    name = row.get("Customer Name")
    if pd.notna(email):
        return str(email)
    if pd.notna(name):
        return str(name)
    return None


def extract_day(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    match = DAY_PATTERN.search(str(text))
    if not match:
        return None
    return DAY_MAP.get(match.group(1).lower())


def extract_time(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    match = TIME_PATTERN.search(str(text))
    if not match:
        return None
    value = match.group(1).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s*(am|pm)$", r" \1", value)


def extract_class_type(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    return "ASC" if re.search(r"\bASC\b", str(text), flags=re.IGNORECASE) else "ACD"


def extract_outward_postcode(series: pd.Series, valid_postcodes: Iterable[str]) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.upper()
        .str.replace(" ", "", regex=False)
    )
    candidate = cleaned.str.extract(OUTWARD_POSTCODE_PATTERN, expand=False)
    return candidate.where(candidate.isin(set(valid_postcodes)))


def propagate_first_non_null_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
) -> pd.DataFrame:
    df = df.copy()
    non_null_lookup = (
        df[[group_col, target_col]]
        .dropna(subset=[group_col, target_col])
        .drop_duplicates(subset=[group_col], keep="first")
        .set_index(group_col)[target_col]
    )

    missing_mask = df[target_col].isna() & df[group_col].notna()
    df.loc[missing_mask, target_col] = df.loc[missing_mask, group_col].map(non_null_lookup)
    return df


class StripeAnalyticsPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def load_primary_tables(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Loading checkout, payment links, and payments files")
        checkout = pd.read_csv(self.config.checkout_path)
        payment_links = pd.read_csv(self.config.payment_links_path)
        payments = pd.read_csv(self.config.payments_path)

        checkout = strip_columns(checkout)
        payment_links = strip_columns(payment_links)
        payments = strip_columns(payments)

        checkout = coerce_datetimes(checkout, ["Created (UTC)"])
        payments = coerce_datetimes(
            payments,
            ["Created date (UTC)", "Refunded date (UTC)"],
        )

        checkout = drop_empty_and_zero_columns(standardize_customer_fields(checkout))
        payment_links = drop_empty_and_zero_columns(payment_links)
        payments = drop_empty_and_zero_columns(standardize_customer_fields(payments))

        checkout["client_id"] = checkout.apply(generate_client_id, axis=1)
        payments["client_id"] = payments.apply(generate_client_id, axis=1)

        return checkout, payment_links, payments

    def build_transaction_level_dataset(self) -> pd.DataFrame:
        checkout, payment_links, payments = self.load_primary_tables()

        payments_merged = payments.merge(
            payment_links[[c for c in ["id", "Name"] if c in payment_links.columns]],
            left_on="Payment Link ID",
            right_on="id",
            how="left",
        )

        checkout_columns = [
            column
            for column in ["Payment Link ID", "client_id", "Created (UTC)"]
            if column in checkout.columns
        ]

        final_df = payments_merged.merge(
            checkout[checkout_columns].drop_duplicates(),
            on=["Payment Link ID", "client_id"],
            how="left",
        )

        if "Card Address Zip" in final_df.columns:
            final_df["postcode"] = extract_outward_postcode(
                final_df["Card Address Zip"],
                self.config.valid_outward_postcodes,
            )

        summary_col = "Line Item Summary" if "Line Item Summary" in final_df.columns else None
        if summary_col:
            final_df["class_day"] = final_df[summary_col].map(extract_day)
            final_df["class_time"] = final_df[summary_col].map(extract_time)
            final_df["class_type"] = final_df[summary_col].map(extract_class_type)

        return final_df

    def load_unified_payments_dataset(self) -> pd.DataFrame:
        logger.info("Loading unified payments dataset for enrichment")
        df = pd.read_csv(self.config.payments_path)
        df = strip_columns(df)
        df = coerce_datetimes(df, ["Created date (UTC)", "Refunded date (UTC)"])
        df = normalize_blank_strings(df, ["Customer ID", "Checkout Line Item Summary"])

        if {"Customer ID", "Checkout Line Item Summary"}.issubset(df.columns):
            df = propagate_first_non_null_by_group(
                df,
                group_col="Customer ID",
                target_col="Checkout Line Item Summary",
            )

        if "Checkout Line Item Summary" in df.columns:
            df["class_day"] = df["Checkout Line Item Summary"].map(extract_day)
            df["class_time"] = df["Checkout Line Item Summary"].map(extract_time)
            df["Class Type"] = df["Checkout Line Item Summary"].map(extract_class_type)

        postcode_source = "Card Address Zip" if "Card Address Zip" in df.columns else None
        if postcode_source:
            df["postcode_outward"] = extract_outward_postcode(
                df[postcode_source],
                self.config.valid_outward_postcodes,
            )

        removable = [c for c in self.config.removable_columns if c in df.columns]
        df = df.drop(columns=removable, errors="ignore")

        return df

    @staticmethod
    def build_client_summary(final_df: pd.DataFrame) -> pd.DataFrame:
        required = {"client_id", "postcode", "Created date (UTC)", "Amount"}
        missing = required - set(final_df.columns)
        if missing:
            raise ValueError(f"Missing required columns for client summary: {sorted(missing)}")

        working = final_df.dropna(subset=["client_id", "postcode"]).copy()

        refunded_source = "Amount Refunded" if "Amount Refunded" in working.columns else None
        if refunded_source is None:
            working["Amount Refunded"] = 0
            refunded_source = "Amount Refunded"

        id_source = "id_x" if "id_x" in working.columns else ("id" if "id" in working.columns else None)
        if id_source is None:
            raise ValueError("No transaction identifier column found ('id_x' or 'id').")

        client_summary = (
            working.groupby(["client_id", "postcode"], dropna=False)
            .agg(
                payments_count=(id_source, "count"),
                first_payment=("Created date (UTC)", "min"),
                last_payment=("Created date (UTC)", "max"),
                refunded_amount=(refunded_source, "sum"),
                total_amount_paid=("Amount", "sum"),
            )
            .reset_index()
        )

        client_summary["lifetime_months"] = (
            (client_summary["last_payment"] - client_summary["first_payment"]).dt.days / 30
        ).round(2)

        client_summary["outcome"] = np.select(
            [
                (client_summary["payments_count"] == 1) & (client_summary["refunded_amount"] > 0),
                (client_summary["payments_count"] > 1) & (client_summary["refunded_amount"] == 0),
            ],
            ["Refunded", "Cancelled"],
            default="Active",
        )

        client_summary["missed_payment"] = (
            (client_summary["refunded_amount"] > 0)
            | (client_summary["payments_count"] == 1)
        ).astype(int)

        return client_summary

    @staticmethod
    def build_model_dataset(
        final_df: pd.DataFrame,
        client_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        client_features = (
            final_df.groupby("client_id", dropna=False)
            .agg(
                monthly_payment=("Amount", "median"),
                class_day=("class_day", "first"),
                class_time=("class_time", "first"),
                class_type=("class_type", "first"),
            )
            .reset_index()
        )

        return client_summary.merge(client_features, on="client_id", how="left")

    @staticmethod
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

    def train_models(self, model_df: pd.DataFrame) -> dict[str, object]:
        X, y = self.prepare_model_inputs(model_df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if y.nunique() > 1 else None,
        )

        logistic_model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        logistic_model.fit(X_train, y_train)

        random_forest_model = RandomForestClassifier(
            n_estimators=300,
            random_state=self.config.random_state,
        )
        random_forest_model.fit(X_train, y_train)

        results = {
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
            "logistic_coefficients": self.build_logistic_coefficients(logistic_model, X.columns),
            "rf_feature_importance": self.build_feature_importance(random_forest_model, X.columns),
        }
        return results

    @staticmethod
    def build_logistic_coefficients(
        model: LogisticRegression,
        feature_names: Iterable[str],
    ) -> pd.DataFrame:
        coefficients = pd.DataFrame(
            {
                "feature": list(feature_names),
                "coefficient": model.coef_[0],
            }
        )
        coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
        return coefficients.sort_values("coefficient", ascending=False).reset_index(drop=True)

    @staticmethod
    def build_feature_importance(
        model: RandomForestClassifier,
        feature_names: Iterable[str],
    ) -> pd.Series:
        return pd.Series(model.feature_importances_, index=list(feature_names)).sort_values(ascending=False)

    @staticmethod
    def chi_square_test(client_summary: pd.DataFrame) -> dict[str, object]:
        contingency_table = pd.crosstab(
            client_summary["postcode"],
            client_summary["missed_payment"],
        )
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return {
            "contingency_table": contingency_table,
            "chi2": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "expected": expected,
        }

    @staticmethod
    def plot_churn_by_postcode(client_summary: pd.DataFrame) -> None:
        ax = (
            client_summary.groupby(["postcode", "outcome"])
            .size()
            .unstack(fill_value=0)
            .plot(kind="bar", stacked=True, figsize=(12, 6))
        )
        ax.set_title("Customer Outcome by Postcode")
        ax.set_xlabel("Postcode")
        ax.set_ylabel("Client Count")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_repeat_customer_geography(client_summary: pd.DataFrame) -> None:
        ax = (
            client_summary.groupby("postcode")["payments_count"]
            .mean()
            .sort_values(ascending=False)
            .plot(kind="bar", figsize=(12, 6))
        )
        ax.set_title("Average Number of Payments by Postcode")
        ax.set_xlabel("Postcode")
        ax.set_ylabel("Average Payments")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_lifetime_vs_spend(client_summary: pd.DataFrame, hue: str = "postcode") -> None:
        q3_df = client_summary[
            (client_summary["payments_count"] > 0)
            & (client_summary["lifetime_months"] > 0)
            & (client_summary["total_amount_paid"] > 0)
        ].copy()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=q3_df,
            x="lifetime_months",
            y="total_amount_paid",
            hue=hue,
            alpha=0.75,
        )
        plt.title("Customer Lifetime vs Total Spend")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap(data: pd.DataFrame, title: str, figsize: tuple[int, int] = (12, 8)) -> None:
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def build_correlation_matrix(
        client_level_df: pd.DataFrame,
        categorical_columns: Iterable[str],
    ) -> pd.DataFrame:
        corr_df = client_level_df.copy()
        categorical_columns = [c for c in categorical_columns if c in corr_df.columns]
        corr_df = pd.get_dummies(corr_df, columns=categorical_columns, drop_first=True)
        return corr_df.corr(numeric_only=True)

    @staticmethod
    def plot_correlation_matrix(correlation_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> None:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap="coolwarm",
            annot=False,
            linewidths=0.5,
            cbar_kws={"label": "Correlation"},
        )
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def run(self) -> dict[str, pd.DataFrame | dict[str, object] | pd.Series]:
        final_df = self.build_transaction_level_dataset()
        client_summary = self.build_client_summary(final_df)
        model_df = self.build_model_dataset(final_df, client_summary)
        model_results = self.train_models(model_df)

        unified_df = self.load_unified_payments_dataset()
        client_level_unified = (
            unified_df.groupby("Customer ID", dropna=False)
            .agg(
                payments_count=("id", "count"),
                first_payment=("Created date (UTC)", "min"),
                last_payment=("Created date (UTC)", "max"),
                total_amount_paid=("Amount", "sum"),
                monthly_payment=("Amount", "median"),
                postcode_outward=("postcode_outward", "first"),
                class_day=("class_day", "first"),
                class_time=("class_time", "first"),
                class_type=("Class Type", "first"),
            )
            .reset_index()
        )
        client_level_unified["lifetime_months"] = (
            (client_level_unified["last_payment"] - client_level_unified["first_payment"]).dt.days / 30
        ).round(2)

        correlation_matrix = self.build_correlation_matrix(
            client_level_unified[
                [
                    "payments_count",
                    "lifetime_months",
                    "total_amount_paid",
                    "monthly_payment",
                    "postcode_outward",
                    "class_day",
                    "class_time",
                    "class_type",
                ]
            ].copy(),
            categorical_columns=["postcode_outward", "class_day", "class_time", "class_type"],
        )

        chi_square_results = self.chi_square_test(client_summary)

        return {
            "final_df": final_df,
            "client_summary": client_summary,
            "model_df": model_df,
            "unified_df": unified_df,
            "client_level_unified": client_level_unified,
            "correlation_matrix": correlation_matrix,
            "chi_square_results": chi_square_results,
            "model_results": model_results,
        }


if __name__ == "__main__":
    pipeline = StripeAnalyticsPipeline()
    outputs = pipeline.run()

    logger.info("Pipeline completed successfully")
    logger.info("Client summary shape: %s", outputs["client_summary"].shape)
    logger.info("Model dataset shape: %s", outputs["model_df"].shape)
    logger.info(
        "Top random forest features:\n%s",
        outputs["model_results"]["rf_feature_importance"].head(10),
    )
