from src.schemas.document import PaperDocument, RetrievedPaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation, PaperCitationRelation
from src.storage.graph.neo4j_repository import GraphToolNodeProjector, Neo4jGraphRepository


class FakeDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters_=None, database_=None):
        self.calls.append((query, parameters_, database_))


def test_repository_targets_configured_database():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")
    repo.upsert_papers(
        [
            PaperDocument(
                document_id="Seed Paper",
                source_type="openalex",
                source_ref="https://openalex.org/W1",
                abstract="A",
                ingestion_status="ready",
            )
        ]
    )

    assert driver.calls[0][2] == "testidea"


def test_repository_writes_paper_and_entity_edges():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")
    repo.upsert_citations([PaperCitationRelation(source_paper_id="p1", target_paper_id="p2")])
    repo.upsert_entities(
        [
            ExtractedEntity(
                entity_id="graph reasoning",
                entity_type="Concept",
                source_refs=["p1"],
            )
        ]
    )
    repo.upsert_paper_entity_relations("p1", ["graph reasoning"])
    repo.upsert_entity_relations(
        [
            ExtractedRelation(
                source_entity_id="graph reasoning",
                target_entity_id="idea generation",
                relation_type="Used_For",
                source_refs=["p1"],
            )
        ]
    )

    rendered = "\n".join(call[0] for call in driver.calls)
    assert "CITES" in rendered
    assert "SET e:`Concept`" in rendered
    assert "CONTAINS" in rendered
    assert "USED_FOR" in rendered


def test_repository_writes_paper_id_without_title_property():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.upsert_papers(
        [
            PaperDocument(
                document_id="Seed Paper",
                source_type="openalex",
                source_ref="https://openalex.org/W1",
                abstract="A",
                ingestion_status="ready",
            )
        ]
    )

    query, params, _ = driver.calls[0]
    assert "MERGE (p:Paper {id: paper.document_id})" in query
    assert "p.title" not in query
    assert params["papers"][0]["document_id"] == "Seed Paper"
    assert "title" not in params["papers"][0]


def test_repository_replaces_existing_paper_for_rebuild_and_returns_preserved_citations():
    class FakeReplaceDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    {
                        "citations": [
                            {"source_paper_id": "citing-paper", "target_paper_id": "Paper A"},
                            {"source_paper_id": "Paper A", "target_paper_id": "cited-paper"},
                        ]
                    }
                ],
                None,
                None,
            )

    driver = FakeReplaceDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    citations = repo.replace_existing_paper_for_rebuild("Paper A")

    query, params, database = driver.calls[0]
    assert params == {"paper_id": "Paper A"}
    assert database == "testidea"
    assert "MATCH (p:Paper {id: $paper_id})" in query
    assert "OPTIONAL MATCH (p)-[:CONTAINS]->(contained:Entity)" in query
    assert "OPTIONAL MATCH (p)-[:CITES]->(outgoing:Paper)" in query
    assert "OPTIONAL MATCH (incoming:Paper)-[:CITES]->(p)" in query
    assert "DETACH DELETE p" in query
    assert "count(remaining_relation) AS remaining_relation_count" in query
    assert "remaining_relation_count = 0" in query
    assert "DETACH DELETE contained" in query
    assert citations == [
        PaperCitationRelation(source_paper_id="citing-paper", target_paper_id="Paper A"),
        PaperCitationRelation(source_paper_id="Paper A", target_paper_id="cited-paper"),
    ]


def test_repository_sets_entity_type_label_only_on_create():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.upsert_entities(
        [
            ExtractedEntity(
                entity_id="graph reasoning",
                entity_type="Concept",
                aliases=["graph-based reasoning"],
                description="Reasoning over graph structure.",
            )
        ]
    )

    query, params, _ = driver.calls[0]
    assert "ON CREATE SET e:`Concept`" in query
    assert "e.entity_type = $entity_type" in query
    assert "e.aliases" in query
    assert "e.description" in query
    assert "\n            SET e:`Concept`" not in query
    assert params["entity_type"] == "Concept"
    assert params["description"] == "Reasoning over graph structure."


def test_repository_adds_lowercase_entity_id_to_aliases():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.upsert_entities(
        [
            ExtractedEntity(
                entity_id="GraphRAG",
                entity_type="Method",
                aliases=["Graph Retrieval-Augmented Generation"],
                description="Graph retrieval method.",
            )
        ]
    )

    _, params, _ = driver.calls[0]
    assert params["aliases"] == ["Graph Retrieval-Augmented Generation", "graphrag"]


def test_repository_can_replace_entity_description_for_resolved_entities():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.upsert_entities(
        [
            ExtractedEntity(
                entity_id="graph neural network",
                entity_type="Method",
                description="Merged graph neural network definition.",
                description_update_mode="replace",
            )
        ]
    )

    query, params, _ = driver.calls[0]
    assert "WHEN $description_update_mode = 'replace' THEN $description" in query
    assert params["description_update_mode"] == "replace"


def test_repository_merges_entity_relation_refs_and_descriptions():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.upsert_entity_relations(
        [
            ExtractedRelation(
                source_entity_id="graph neural network",
                target_entity_id="idea generation",
                relation_type="Used-For",
                description="GNNs support idea generation.",
                source_refs=["paper-1"],
            )
        ]
    )

    query, params, _ = driver.calls[0]
    assert "coalesce(r.source_refs, []) + $source_refs" in query
    assert "WHEN $description = '' THEN coalesce(r.description, '')" in query
    assert params["source_refs"] == ["paper-1"]


def test_repository_gets_existing_entity_relation_by_canonical_key():
    class FakeRelationDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    {
                        "relation": {
                            "description": "Existing relation evidence.",
                            "source_refs": ["paper-1"],
                        }
                    }
                ],
                None,
                None,
            )

    driver = FakeRelationDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    relation = repo.get_entity_relation(
        source_entity_id="graph neural network",
        relation_type="Used-For",
        target_entity_id="idea generation",
    )

    query, params, database = driver.calls[0]
    assert "-[r:USED_FOR]->" in query
    assert params == {
        "source_entity_id": "graph neural network",
        "target_entity_id": "idea generation",
    }
    assert database == "testidea"
    assert relation == {
        "description": "Existing relation evidence.",
        "source_refs": ["paper-1"],
    }


def test_repository_creates_database_when_missing():
    class FakeAdminDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            if query == "SHOW DATABASES":
                return ([{"name": "neo4j"}], None, None)
            return ([], None, None)

    driver = FakeAdminDriver()

    Neo4jGraphRepository.ensure_database_exists(driver, "testidea")

    assert driver.calls[0][0] == "SHOW DATABASES"
    assert driver.calls[0][2] == "system"
    assert "CREATE DATABASE `testidea` IF NOT EXISTS" in driver.calls[1][0]
    assert driver.calls[1][2] == "system"


def test_repository_ensures_graph_id_uniqueness_constraints():
    class FakeConstraintDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return ([], None, None)

    driver = FakeConstraintDriver()

    Neo4jGraphRepository.ensure_graph_constraints(driver, "testidea")

    rendered = "\n".join(call[0] for call in driver.calls)
    assert "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS" in rendered
    assert "FOR (p:Paper) REQUIRE p.id IS UNIQUE" in rendered
    assert "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS" in rendered
    assert "FOR (e:Entity) REQUIRE e.id IS UNIQUE" in rendered
    assert {call[2] for call in driver.calls} == {"testidea"}


def test_repository_lists_only_papers_with_non_empty_abstracts():
    class FakeReadDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    {
                        "paper": {
                            "id": "Retriever paper",
                            "source_type": "openalex",
                            "source_ref": "https://openalex.org/W1",
                            "abstract": "Useful abstract",
                            "year": 2024,
                            "cited_by_count": 42,
                            "ingestion_status": None,
                        }
                    }
                ],
                None,
                None,
            )

    driver = FakeReadDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    papers = repo.list_papers_with_abstracts()

    assert papers == [
        RetrievedPaperDocument(
            document_id="Retriever paper",
            title="Retriever paper",
            abstract="Useful abstract",
            year=2024,
            cited_by_count=42,
        )
    ]
    assert "MATCH (p:Paper)" in driver.calls[0][0]
    assert "WHERE p.abstract IS NOT NULL" in driver.calls[0][0]
    assert "trim(p.abstract) <> ''" in driver.calls[0][0]
    assert driver.calls[0][2] == "testidea"


def test_repository_lists_papers_for_vector_index_with_full_text_path():
    class FakeReadDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    {
                        "paper": {
                            "id": "Full Text Paper",
                            "abstract": "Useful abstract",
                            "year": 2025,
                            "cited_by_count": 9,
                            "full_text_markdown_path": "data/testidea/paper.md",
                            "text_source": "full_text_md",
                        }
                    }
                ],
                None,
                None,
            )

    driver = FakeReadDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    papers = repo.list_papers_for_vector_index()

    assert papers == [
        RetrievedPaperDocument(
            document_id="Full Text Paper",
            title="Full Text Paper",
            abstract="Useful abstract",
            year=2025,
            cited_by_count=9,
            full_text_markdown_path="data/testidea/paper.md",
            text_source="full_text_md",
        )
    ]
    assert "WHERE p.abstract IS NOT NULL" in driver.calls[0][0]
    assert "trim(p.abstract) <> ''" in driver.calls[0][0]
    assert "full_text_markdown_path: p.full_text_markdown_path" in driver.calls[0][0]
    assert driver.calls[0][2] == "testidea"


def test_repository_gets_paper_by_exact_id():
    class FakeReadDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    {
                        "paper": {
                            "id": "Graph-guided discovery",
                            "abstract": "Useful abstract",
                            "year": 2024,
                            "cited_by_count": 42,
                        }
                    }
                ],
                None,
                None,
            )

    driver = FakeReadDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    paper = repo.get_paper_by_id("Graph-guided discovery")

    assert paper == RetrievedPaperDocument(
        document_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Useful abstract",
        year=2024,
        cited_by_count=42,
    )
    query, params, database = driver.calls[0]
    assert "MATCH (p:Paper {id: $paper_id})" in query
    assert params == {"paper_id": "Graph-guided discovery"}
    assert database == "testidea"


def test_repository_get_paper_by_id_returns_none_when_missing():
    class FakeReadDriver:
        def execute_query(self, query, parameters_=None, database_=None):
            del query, parameters_, database_
            return ([], None, None)

    repo = Neo4jGraphRepository(driver=FakeReadDriver(), database="testidea")

    assert repo.get_paper_by_id("Missing Paper") is None


def test_repository_read_path_preserves_missing_fields_as_none():
    class FakeReadDriver:
        def execute_query(self, query, parameters_=None, database_=None):
            return (
                [
                    {
                        "paper": {
                            "id": "Sparse paper",
                            "source_type": None,
                            "source_ref": None,
                            "abstract": "Still useful",
                            "year": None,
                            "cited_by_count": None,
                            "ingestion_status": None,
                        }
                    }
                ],
                None,
                None,
            )

    repo = Neo4jGraphRepository(driver=FakeReadDriver(), database="testidea")

    papers = repo.list_papers_with_abstracts()

    assert papers == [
        RetrievedPaperDocument(
            document_id="Sparse paper",
            title="Sparse paper",
            abstract="Still useful",
            year=None,
            cited_by_count=None,
        )
    ]


def test_repository_lists_one_hop_neighborhood_with_split_entities_and_relationships():
    class FakeNeo4jRecord:
        def __init__(self, payload):
            self._payload = payload

        def __getitem__(self, key):
            if key != "neighborhood":
                raise KeyError(key)
            return self._payload

        def __iter__(self):
            return iter([("neighborhood", self._payload)])

    class FakeNeighborhoodDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    FakeNeo4jRecord(
                    {
                        "entities": [
                            {
                                "labels": ["Paper"],
                                "properties": {
                                    "id": "Graph-guided discovery",
                                    "abstract": "A paper abstract.",
                                    "source_ref": "hidden",
                                },
                            },
                            {
                                "labels": ["Entity", "Concept"],
                                "properties": {
                                    "id": "knowledge graph",
                                    "entity_type": "Concept",
                                    "description": "Structured knowledge.",
                                    "aliases": ["KG"],
                                },
                            },
                        ],
                        "relationships": [
                            {
                                "source_id": "Graph-guided discovery",
                                "target_id": "knowledge graph",
                                "relationship_type": "CONTAINS",
                            }
                        ],
                        "total_entities": 2,
                        "total_relationships": 1,
                        "page": 1,
                        "page_size": 20,
                    }
                    )
                ],
                None,
                None,
            )

    driver = FakeNeighborhoodDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    result = repo.get_node_neighborhood(
        node_id="Graph-guided discovery",
        node_types=["Paper", "Concept"],
        relationship_types=["CONTAINS"],
        page=1,
    )

    assert result == {
        "entities": [
            {
                "id": "Graph-guided discovery",
                "abstract": "A paper abstract.",
            },
            {
                "id": "knowledge graph",
                "entity_type": "Concept",
                "description": "Structured knowledge.",
            },
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            }
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 2,
        "total_relationships": 1,
    }
    query, params, database = driver.calls[0]
    assert "MATCH (n {id: $node_id})" in query
    assert "type(r) IN $relationship_types" in query
    assert params == {
        "node_id": "Graph-guided discovery",
        "node_types": ["Paper", "Concept"],
        "relationship_types": ["CONTAINS"],
        "skip": 0,
        "page": 1,
        "page_size": 20,
    }
    assert database == "testidea"


def test_repository_neighborhood_filters_paper_nodes_without_abstracts():
    class EmptyNeighborhoodDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return ([], None, None)

    driver = EmptyNeighborhoodDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.get_node_neighborhood("Graph-guided discovery", node_types=["Paper"])

    query = driver.calls[0][0]
    assert "NOT 'Paper' IN labels(n)" in query
    assert "trim(n.abstract) <> ''" in query
    assert "NOT 'Paper' IN labels(m)" in query
    assert "trim(m.abstract) <> ''" in query


def test_repository_resolves_neighborhood_by_case_insensitive_alias_after_exact_miss():
    class FakeNeo4jRecord:
        def __init__(self, payload):
            self._payload = payload

        def __getitem__(self, key):
            if key != "neighborhood":
                raise KeyError(key)
            return self._payload

    class FakeAliasDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            node_id = parameters_["node_id"]
            if node_id == "GraphRAG" and "toLower(alias) = toLower($node_id)" not in query:
                return ([], None, None)
            if node_id == "GraphRAG" and "toLower(alias) = toLower($node_id)" in query:
                return (
                    [
                        FakeNeo4jRecord(
                            {
                                "entities": [
                                    {
                                        "labels": ["Entity", "Method"],
                                        "properties": {
                                            "id": "graphrag",
                                            "entity_type": "Method",
                                            "description": "Graph retrieval method.",
                                            "aliases": ["GraphRAG"],
                                        },
                                    }
                                ],
                                "relationships": [],
                                "total_entities": 1,
                                "total_relationships": 0,
                                "page": 1,
                                "page_size": 20,
                            }
                        )
                    ],
                    None,
                    None,
                )
            raise AssertionError(f"unexpected node_id: {node_id}")

    driver = FakeAliasDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    result = repo.get_node_neighborhood("GraphRAG")

    assert result["entities"] == [
        {
            "id": "graphrag",
            "entity_type": "Method",
            "description": "Graph retrieval method.",
        }
    ]
    assert [call[1]["node_id"] for call in driver.calls] == ["GraphRAG", "GraphRAG"]
    assert "toLower(alias) = toLower($node_id)" in driver.calls[1][0]


def test_graph_tool_node_projector_limits_returned_properties():
    projector = GraphToolNodeProjector()

    paper = projector.project(
        {
            "labels": ["Paper"],
            "properties": {
                "id": "Paper A",
                "abstract": "Useful abstract.",
                "source_ref": "hidden",
                "year": 2024,
            },
        }
    )
    entity = projector.project(
        {
            "labels": ["Entity", "Method"],
            "properties": {
                "id": "graphrag",
                "entity_type": "Method",
                "description": "Graph retrieval method.",
                "aliases": ["GraphRAG"],
                "source_refs": ["Paper A"],
            },
        }
    )

    assert paper == {"id": "Paper A", "abstract": "Useful abstract."}
    assert entity == {
        "id": "graphrag",
        "entity_type": "Method",
        "description": "Graph retrieval method.",
    }


def test_repository_lists_non_paper_entities_for_vector_retrieval():
    class FakeEntityDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return (
                [
                    {
                        "entity": {
                            "id": "graph rag",
                            "entity_type": "Method",
                            "description": "Graph retrieval method.",
                            "aliases": ["graph retrieval augmented generation"],
                        }
                    },
                    {
                        "entity": {
                            "id": "knowledge graph",
                            "entity_type": "Concept",
                            "description": None,
                            "aliases": None,
                        }
                    },
                ],
                None,
                None,
            )

    driver = FakeEntityDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    entities = repo.list_entities()

    assert entities == [
        {
            "id": "graph rag",
            "entity_type": "Method",
            "description": "Graph retrieval method.",
            "aliases": ["graph retrieval augmented generation"],
        },
        {
            "id": "knowledge graph",
            "entity_type": "Concept",
            "description": "",
            "aliases": [],
        },
    ]
    query, params, database = driver.calls[0]
    assert "MATCH (e:Entity)" in query
    assert "'Paper' IN labels(e)" in query
    assert params is None
    assert database == "testidea"


def test_node_type_filter_does_not_filter_queried_center_node():
    class EmptyNeighborhoodDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, parameters_=None, database_=None):
            self.calls.append((query, parameters_, database_))
            return ([], None, None)

    driver = EmptyNeighborhoodDriver()
    repo = Neo4jGraphRepository(driver=driver, database="testidea")

    repo.get_node_neighborhood("neuroevolution", node_types=["Paper"])

    query = driver.calls[0][0]
    assert query.count("size($node_types) = 0") == 1
    assert "WITH n\n        WHERE size($node_types) = 0" not in query
