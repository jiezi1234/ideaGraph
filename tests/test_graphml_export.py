from src.exporters.graphml_exporter import GraphmlExportConfig, export_graphml_from_driver


class FakeRecord:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class FakeDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters_=None, database_=None):
        self.calls.append((query, parameters_, database_))
        if query.startswith("MATCH (n)"):
            return (
                [
                    FakeRecord(
                        {
                            "node_id": "n1",
                            "labels": ["Paper"],
                            "properties": {
                                "id": "p1",
                                "title": "Seed Paper",
                                "year": 2024,
                            },
                        }
                    ),
                    FakeRecord(
                        {
                            "node_id": "n2",
                            "labels": ["Paper"],
                            "properties": {
                                "id": "p2",
                                "title": "Cited Paper",
                                "year": 2023,
                            },
                        }
                    ),
                ],
                None,
                None,
            )
        return (
            [
                FakeRecord(
                    {
                        "edge_id": "r1",
                        "source": "n1",
                        "target": "n2",
                        "type": "CITES",
                        "properties": {"weight": 1},
                    }
                )
            ],
            None,
            None,
        )


def test_graphml_exporter_creates_graphml_with_nodes_and_edges(tmp_path):
    output = tmp_path / "testidea.graphml"
    driver = FakeDriver()

    summary = export_graphml_from_driver(
        driver=driver,
        database="testidea",
        output_path=output,
        config=GraphmlExportConfig(),
    )

    assert summary.node_count == 2
    assert summary.edge_count == 1
    assert output.exists()

    content = output.read_text(encoding="utf-8")
    assert "<graphml" in content
    assert '<node id="n1">' in content
    assert '<edge id="r1" source="n1" target="n2"' in content
    assert "Paper" in content
    assert "CITES" in content


def test_graphml_exporter_uses_default_query(tmp_path):
    output = tmp_path / "default.graphml"
    driver = FakeDriver()

    summary = export_graphml_from_driver(
        driver=driver,
        database="testidea",
        output_path=output,
    )

    assert summary.output_path == output
    assert summary.query.startswith("MATCH (n)")
    assert summary.node_count == 2
