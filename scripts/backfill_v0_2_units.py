#!/usr/bin/env python3
from __future__ import annotations

import argparse

import psycopg

from bid_scoring.backfill_v0_2 import backfill_units_from_chunks
from bid_scoring.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill v0.2 units from existing v0.1 chunks"
    )
    parser.add_argument("--version-id", help="Only backfill a single version_id (UUID)")
    args = parser.parse_args()

    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        stats = backfill_units_from_chunks(conn, version_id=args.version_id)
    print(stats)


if __name__ == "__main__":
    main()
