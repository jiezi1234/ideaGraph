from pathlib import Path

from neo4j import GraphDatabase

from src.config.settings import Settings
from src.exporters.graphml_exporter import export_graphml_from_driver


def main() -> None:
    settings = Settings()
    output = Path("exports") / f"{settings.neo4j_database}.graphml"
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        export_graphml_from_driver(
            driver=driver,
            database=settings.neo4j_database,
            output_path=output,
        )
        print(f"GraphML exported to {output}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()