from __future__ import annotations

import logging

from src.config import ProjectConfig
from src.data.clean import (
    build_client_id,
    coerce_datetimes,
    drop_empty_and_zero_columns,
    enrich_class_columns,
    extract_outward_postcode,
    normalize_blank_strings,
    propagate_first_non_null_by_group,
    standardize_customer_fields,
    strip_columns,
)
from src.data.load import load_raw_tables, load_unified_payments
from src.data.validate import summarize_quality, validate_non_empty
from src.features.build_features import (
    build_client_level_unified,
    build_client_summary,
    build_correlation_matrix,
    build_model_dataset,
    build_transaction_level_dataset,
)
from src.models.train import chi_square_test, train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class StripeAnalyticsPipeline:
    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig()

    def load_and_prepare_primary_tables(self):
        checkout, payment_links, payments = load_raw_tables(self.config)

        checkout = strip_columns(checkout)
        payment_links = strip_columns(payment_links)
        payments = strip_columns(payments)

        checkout = coerce_datetimes(checkout, ["Created (UTC)"])
        payments = coerce_datetimes(payments, ["Created date (UTC)", "Refunded date (UTC)"])

        checkout = drop_empty_and_zero_columns(standardize_customer_fields(checkout))
        payment_links = drop_empty_and_zero_columns(payment_links)
        payments = drop_empty_and_zero_columns(standardize_customer_fields(payments))

        checkout = build_client_id(checkout)
        payments = build_client_id(payments)

        validate_non_empty(checkout, "checkout")
        validate_non_empty(payment_links, "payment_links")
        validate_non_empty(payments, "payments")
        return checkout, payment_links, payments

    def load_and_prepare_unified(self):
        unified_df = load_unified_payments(self.config)
        unified_df = strip_columns(unified_df)
        unified_df = coerce_datetimes(unified_df, ["Created date (UTC)", "Refunded date (UTC)"])
        unified_df = normalize_blank_strings(unified_df, ["Customer ID", "Checkout Line Item Summary"])

        if {"Customer ID", "Checkout Line Item Summary"}.issubset(unified_df.columns):
            unified_df = propagate_first_non_null_by_group(
                unified_df,
                group_col="Customer ID",
                target_col="Checkout Line Item Summary",
            )

        if "Checkout Line Item Summary" in unified_df.columns:
            unified_df = enrich_class_columns(
                unified_df,
                source_col="Checkout Line Item Summary",
                class_type_col="Class Type",
            )

        if "Card Address Zip" in unified_df.columns:
            unified_df["postcode_outward"] = extract_outward_postcode(
                unified_df["Card Address Zip"],
                self.config.valid_outward_postcodes,
            )

        removable = [c for c in self.config.removable_columns if c in unified_df.columns]
        unified_df = unified_df.drop(columns=removable, errors="ignore")
        validate_non_empty(unified_df, "unified_df")
        return unified_df

    def run(self) -> dict[str, object]:
        checkout, payment_links, payments = self.load_and_prepare_primary_tables()
        final_df = build_transaction_level_dataset(
            checkout=checkout,
            payment_links=payment_links,
            payments=payments,
            valid_postcodes=self.config.valid_outward_postcodes,
            postcode_extractor=extract_outward_postcode,
            class_enricher=enrich_class_columns,
        )
        client_summary = build_client_summary(final_df)
        model_df = build_model_dataset(final_df, client_summary)
        model_results = train_models(
            model_df,
            random_state=self.config.random_state,
            test_size=self.config.test_size,
        )

        unified_df = self.load_and_prepare_unified()
        client_level_unified = build_client_level_unified(unified_df)
        correlation_matrix = build_correlation_matrix(
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
        chi_square_results = chi_square_test(client_summary)

        logger.info("Primary dataset quality summary:\n%s", summarize_quality(final_df).head(10))
        logger.info("Pipeline completed successfully")

        return {
            "final_df": final_df,
            "client_summary": client_summary,
            "model_df": model_df,
            "model_results": model_results,
            "unified_df": unified_df,
            "client_level_unified": client_level_unified,
            "correlation_matrix": correlation_matrix,
            "chi_square_results": chi_square_results,
        }


if __name__ == "__main__":
    pipeline = StripeAnalyticsPipeline()
    outputs = pipeline.run()
    logger.info("Client summary shape: %s", outputs["client_summary"].shape)
    logger.info("Model dataset shape: %s", outputs["model_df"].shape)
