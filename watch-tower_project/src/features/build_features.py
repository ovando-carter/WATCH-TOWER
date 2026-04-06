from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate import require_columns


def build_transaction_level_dataset(
    checkout: pd.DataFrame,
    payment_links: pd.DataFrame,
    payments: pd.DataFrame,
    valid_postcodes: tuple[str, ...],
    postcode_extractor,
    class_enricher,
) -> pd.DataFrame:
    require_columns(payments, ["Payment Link ID", "Customer Email", "Customer Name"], "payments")

    payments_merged = payments.merge(
        payment_links[[c for c in ["id", "Name"] if c in payment_links.columns]],
        left_on="Payment Link ID",
        right_on="id",
        how="left",
    )

    checkout_columns = [
        c for c in ["Payment Link ID", "client_id", "Created (UTC)"] if c in checkout.columns
    ]

    final_df = payments_merged.merge(
        checkout[checkout_columns].drop_duplicates(),
        on=["Payment Link ID", "client_id"],
        how="left",
    )

    if "Card Address Zip" in final_df.columns:
        final_df["postcode"] = postcode_extractor(final_df["Card Address Zip"], valid_postcodes)

    if "Line Item Summary" in final_df.columns:
        final_df = class_enricher(final_df, source_col="Line Item Summary", class_type_col="class_type")

    return final_df



def build_client_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        final_df,
        ["client_id", "postcode", "Created date (UTC)", "Amount"],
        "final_df",
    )

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
        (client_summary["refunded_amount"] > 0) | (client_summary["payments_count"] == 1)
    ).astype(int)
    return client_summary



def build_model_dataset(final_df: pd.DataFrame, client_summary: pd.DataFrame) -> pd.DataFrame:
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



def build_client_level_unified(unified_df: pd.DataFrame) -> pd.DataFrame:
    client_level = (
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
    client_level["lifetime_months"] = (
        (client_level["last_payment"] - client_level["first_payment"]).dt.days / 30
    ).round(2)
    return client_level



def build_correlation_matrix(client_level_df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    corr_df = client_level_df.copy()
    categorical_columns = [c for c in categorical_columns if c in corr_df.columns]
    corr_df = pd.get_dummies(corr_df, columns=categorical_columns, drop_first=True)
    return corr_df.corr(numeric_only=True)
