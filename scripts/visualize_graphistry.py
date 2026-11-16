from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graphistry
from dotenv import dotenv_values
from neo4j import Driver
from neo4j import GraphDatabase

from src.config.settings import Settings
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphistryCredentials:
    personal_key_id: str
    personal_key_secret: str
    server: str | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        env_file: str | Path = ".env",
    ) -> "GraphistryCredentials":
        env = os.environ if env is None else env
        dotenv_env = {
            key: value
            for key, value in dotenv_values(env_file).items()
            if value is not None
        }
        merged_env = {**dotenv_env, **env}
        personal_key_id = merged_env.get("GRAPHISTRY_PERSONAL_KEY_ID", "").strip()
        personal_key_secret = merged_env.get("GRAPHISTRY_PERSONAL_KEY_SECRET", "").strip()
        missing = [
            name
            for name, value in [
                ("GRAPHISTRY_PERSONAL_KEY_ID", personal_key_id),
                ("GRAPHISTRY_PERSONAL_KEY_SECRET", personal_key_secret),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required Graphistry environment variables: {', '.join(missing)}")
        server = merged_env.get("GRAPHISTRY_SERVER", "").strip() or None
        return cls(
            personal_key_id=personal_key_id,
            personal_key_secret=personal_key_secret,
            server=server,
        )


@dataclass(frozen=True)
class GraphistryVisualizationResult:
    url: str
    node_count: int
    edge_count: int
    database: str
    query: str
    params: dict[str, int]


class DatabaseScopedDriver(Driver):
    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def session(self, *args: Any, **kwargs: Any) -> Any:
        # Graphistry 调用 driver.session() 时不带 database，这里固定到 IdeaGraph 当前库。
        kwargs.setdefault("database", self._database)
        return self._driver.session(*args, **kwargs)

    def close(self) -> None:
        self._driver.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)


def build_graph_query(limit: int | None = None) -> tuple[str, dict[str, int]]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer")
    query = "MATCH (node1)\nOPTIONAL MATCH (node1)-[connection]->(node2)\nRETURN node1, connection, node2"
    params: dict[str, int] = {}
    if limit is not None:
        query = f"{query}\nLIMIT $limit"
        params["limit"] = limit
    return query, params


def run_visualization(
    *,
    settings: Settings,
    credentials: GraphistryCredentials,
    graphistry_client: Any = graphistry,
    driver_factory: Callable[..., Any] = GraphDatabase.driver,
    limit: int | None = None,
    name: str | None = None,
    render: str = "url",
) -> GraphistryVisualizationResult:
    auth_kwargs: dict[str, Any] = {
        "api": 3,
        "personal_key_id": credentials.personal_key_id,
        "personal_key_secret": credentials.personal_key_secret,
    }
    if credentials.server:
        auth_kwargs["server"] = credentials.server
    graphistry_client.register(**auth_kwargs)

    driver = driver_factory(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    scoped_driver = DatabaseScopedDriver(driver, database=settings.neo4j_database)
    try:
        graphistry_client.register(bolt=scoped_driver)
        query, params = build_graph_query(limit=limit)
        plotter = graphistry_client.cypher(query, params)
        url = plotter.plot(
            name=name or f"IdeaGraph {settings.neo4j_database}",
            description=f"Neo4j database: {settings.neo4j_database}",
            render=render,
        )
        return GraphistryVisualizationResult(
            url=str(url),
            node_count=_safe_len(getattr(plotter, "_nodes", None)),
            edge_count=_safe_len(getattr(plotter, "_edges", None)),
            database=settings.neo4j_database,
            query=query,
            params=params,
        )
    finally:
        scoped_driver.close()


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the current IdeaGraph Neo4j graph to Graphistry.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit. Omit this to upload the full graph, including isolated nodes.",
    )
    parser.add_argument("--name", default=None, help="Graphistry visualization name.")
    parser.add_argument("--render", choices=["url", "browser", "auto"], default="url")
    parser.add_argument("--graphistry-server", default=None, help="Override GRAPHISTRY_SERVER for this run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    credentials = GraphistryCredentials.from_env()
    if args.graphistry_server:
        credentials = GraphistryCredentials(
            personal_key_id=credentials.personal_key_id,
            personal_key_secret=credentials.personal_key_secret,
            server=args.graphistry_server,
        )
    result = run_visualization(
        settings=settings,
        credentials=credentials,
        limit=args.limit,
        name=args.name,
        render=args.render,
    )
    logger.info(
        "graphistry_visualization:uploaded database=%s nodes=%s edges=%s",
        result.database,
        result.node_count,
        result.edge_count,
    )
    logger.info("graphistry_visualization:url %s", result.url)


if __name__ == "__main__":
    main()
