from types import SimpleNamespace

from graphistry.bolt_util import to_bolt_driver

from scripts.visualize_graphistry import DatabaseScopedDriver
from scripts.visualize_graphistry import GraphistryCredentials
from scripts.visualize_graphistry import build_graph_query
from scripts.visualize_graphistry import run_visualization


def test_graphistry_credentials_can_be_loaded_from_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GRAPHISTRY_PERSONAL_KEY_ID=dotenv-key-id\n"
        "GRAPHISTRY_PERSONAL_KEY_SECRET=dotenv-key-secret\n",
        encoding="utf-8",
    )

    credentials = GraphistryCredentials.from_env(env={}, env_file=env_file)

    assert credentials.personal_key_id == "dotenv-key-id"
    assert credentials.personal_key_secret == "dotenv-key-secret"


def test_graphistry_credentials_environment_overrides_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GRAPHISTRY_PERSONAL_KEY_ID=dotenv-key-id\n"
        "GRAPHISTRY_PERSONAL_KEY_SECRET=dotenv-key-secret\n",
        encoding="utf-8",
    )

    credentials = GraphistryCredentials.from_env(
        env={
            "GRAPHISTRY_PERSONAL_KEY_ID": "env-key-id",
            "GRAPHISTRY_PERSONAL_KEY_SECRET": "env-key-secret",
        },
        env_file=env_file,
    )

    assert credentials.personal_key_id == "env-key-id"
    assert credentials.personal_key_secret == "env-key-secret"


class FakeNeo4jDriver:
    def __init__(self):
        self.session_calls = []
        self.closed = False

    def session(self, *args, **kwargs):
        self.session_calls.append((args, kwargs))
        return "session"

    def close(self):
        self.closed = True


def test_database_scoped_driver_injects_configured_database():
    driver = FakeNeo4jDriver()
    scoped = DatabaseScopedDriver(driver, database="testidea")

    session = scoped.session(fetch_size=1000)

    assert session == "session"
    assert driver.session_calls == [((), {"fetch_size": 1000, "database": "testidea"})]


def test_database_scoped_driver_preserves_explicit_database_override():
    driver = FakeNeo4jDriver()
    scoped = DatabaseScopedDriver(driver, database="testidea")

    scoped.session(database="system")

    assert driver.session_calls == [((), {"database": "system"})]


def test_database_scoped_driver_is_accepted_by_graphistry_bolt_registration():
    driver = FakeNeo4jDriver()
    scoped = DatabaseScopedDriver(driver, database="testidea")

    assert to_bolt_driver(scoped) is scoped


def test_build_graph_query_uploads_full_graph_by_default():
    query, params = build_graph_query(limit=None)

    assert "MATCH (node1)" in query
    assert "OPTIONAL MATCH (node1)-[connection]->(node2)" in query
    assert "RETURN node1, connection, node2" in query
    assert "LIMIT" not in query
    assert params == {}


def test_build_graph_query_applies_optional_limit():
    query, params = build_graph_query(limit=500)

    assert query.rstrip().endswith("LIMIT $limit")
    assert params == {"limit": 500}


def test_run_visualization_registers_graphistry_and_scoped_bolt_driver():
    settings = SimpleNamespace(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="secret",
        neo4j_database="testidea",
    )
    credentials = GraphistryCredentials(
        personal_key_id="key-id",
        personal_key_secret="key-secret",
        server=None,
    )
    fake_graphistry = FakeGraphistry()
    fake_driver = FakeNeo4jDriver()

    def driver_factory(uri, auth):
        assert uri == "bolt://localhost:7687"
        assert auth == ("neo4j", "secret")
        return fake_driver

    result = run_visualization(
        settings=settings,
        credentials=credentials,
        graphistry_client=fake_graphistry,
        driver_factory=driver_factory,
        limit=12,
        name="IdeaGraph testidea",
        render="url",
    )

    assert fake_graphistry.register_calls[0] == {
        "api": 3,
        "personal_key_id": "key-id",
        "personal_key_secret": "key-secret",
    }
    scoped_driver = fake_graphistry.register_calls[1]["bolt"]
    scoped_driver.session()
    assert fake_driver.session_calls == [((), {"database": "testidea"})]
    assert fake_graphistry.cypher_calls == [
        (
            "MATCH (node1)\nOPTIONAL MATCH (node1)-[connection]->(node2)\nRETURN node1, connection, node2\nLIMIT $limit",
            {"limit": 12},
        )
    ]
    assert fake_graphistry.plotter.plot_calls == [
        {
            "name": "IdeaGraph testidea",
            "description": "Neo4j database: testidea",
            "render": "url",
        }
    ]
    assert result.url == "https://hub.graphistry.com/graph/abc"
    assert result.node_count == 2
    assert result.edge_count == 1
    assert result.database == "testidea"
    assert fake_driver.closed is True


class FakeGraphistry:
    def __init__(self):
        self.register_calls = []
        self.cypher_calls = []
        self.plotter = FakePlotter()

    def register(self, **kwargs):
        self.register_calls.append(kwargs)

    def cypher(self, query, params):
        self.cypher_calls.append((query, params))
        return self.plotter


class FakePlotter:
    def __init__(self):
        self._nodes = [object(), object()]
        self._edges = [object()]
        self.plot_calls = []

    def plot(self, *, name, description, render):
        self.plot_calls.append(
            {
                "name": name,
                "description": description,
                "render": render,
            }
        )
        return "https://hub.graphistry.com/graph/abc"
