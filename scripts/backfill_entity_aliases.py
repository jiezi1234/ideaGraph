from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill normalized lowercase Entity.id values into Entity.aliases.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing them.")
    args = parser.parse_args()

    settings = Settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    try:
        updated = _backfill_aliases(driver, database=settings.neo4j_database, dry_run=args.dry_run)
    finally:
        driver.close()
    mode = "would_update" if args.dry_run else "updated"
    print(f"entity_alias_backfill:{mode} count={updated}")


def _backfill_aliases(driver, database: str, dry_run: bool) -> int:
    records, _, _ = driver.execute_query(
        """
        MATCH (e:Entity)
        RETURN e.id AS id, e.aliases AS aliases
        ORDER BY e.id
        """,
        database_=database,
    )
    updates = []
    for record in records:
        entity_id = record["id"]
        aliases = _normalized_alias_list(record["aliases"], entity_id)
        if aliases != list(record["aliases"] or []):
            updates.append({"id": entity_id, "aliases": aliases})
    if dry_run or not updates:
        return len(updates)
    driver.execute_query(
        """
        UNWIND $updates AS update
        MATCH (e:Entity {id: update.id})
        SET e.aliases = update.aliases
        """,
        parameters_={"updates": updates},
        database_=database,
    )
    return len(updates)


def _normalized_alias_list(raw_aliases, entity_id: str) -> list[str]:
    aliases = []
    for alias in raw_aliases or []:
        if alias is not None and alias not in aliases:
            aliases.append(alias)
    lowercase_id = _normalize_alias(entity_id)
    if lowercase_id and lowercase_id not in aliases:
        aliases.append(lowercase_id)
    return aliases


def _normalize_alias(value: str) -> str:
    return " ".join(value.strip().lower().split())


if __name__ == "__main__":
    main()
