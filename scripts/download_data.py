"""Download the IBM Telco churn dataset.

The dataset is mirrored in several public locations. Update DATA_URL to
the mirror you trust and pin EXPECTED_SHA256 after the first successful
download so future runs detect tampering or upstream changes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEST = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
EXPECTED_SHA256: str | None = None  # set after first verified download


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST}")
        return 0

    print(f"Downloading {DATA_URL} -> {DEST}")
    response = httpx.get(DATA_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    DEST.write_bytes(response.content)

    digest = hashlib.sha256(response.content).hexdigest()
    print(f"sha256: {digest}")

    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        DEST.unlink()
        print(
            f"Checksum mismatch (expected {EXPECTED_SHA256}); file removed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
