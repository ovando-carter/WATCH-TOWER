from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    checkout_path: Path = Path("checkout_sessions-8.csv")
    payment_links_path: Path = Path("payment_links-2.csv")
    payments_path: Path = Path("unified_payments-16.csv")
    random_state: int = 42
    test_size: float = 0.25
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
