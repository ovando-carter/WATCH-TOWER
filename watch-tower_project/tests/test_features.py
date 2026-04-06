import pandas as pd

from src.features.build_features import build_client_summary



def test_build_client_summary_creates_expected_flags():
    df = pd.DataFrame(
        {
            "client_id": ["a@example.com", "a@example.com", "b@example.com"],
            "postcode": ["SE23", "SE23", "SE24"],
            "Created date (UTC)": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "Amount": [100, 100, 100],
            "Amount Refunded": [0, 0, 100],
            "id": [1, 2, 3],
        }
    )

    result = build_client_summary(df)
    a_row = result.loc[result["client_id"] == "a@example.com"].iloc[0]
    b_row = result.loc[result["client_id"] == "b@example.com"].iloc[0]

    assert a_row["payments_count"] == 2
    assert a_row["outcome"] == "Cancelled"
    assert b_row["missed_payment"] == 1
