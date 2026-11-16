from src.tools.graph.context_provider import VectorFallbackGraphContextProvider


def test_vector_fallback_provider_requeries_best_entity_and_marks_lookup_note():
    class FakeGraphContextProvider:
        def __init__(self):
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append((node_id, node_types, relationship_types, page))
            if node_id == "进化神经网络":
                return {
                    "entities": [],
                    "relationships": [],
                    "page": page,
                    "page_size": 20,
                    "total_entities": 0,
                    "total_relationships": 0,
                }
            return {
                "entities": [
                    {
                        "id": "neuroevolution of augmenting topologies (neat)",
                        "entity_type": "Method",
                        "description": "A neuroevolution method.",
                    }
                ],
                "relationships": [],
                "page": page,
                "page_size": 20,
                "total_entities": 1,
                "total_relationships": 0,
            }

    class FakeEntityRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k):
            self.calls.append((query, top_k))
            return [
                {
                    "id": "neuroevolution of augmenting topologies (neat)",
                    "entity_type": "Method",
                    "score": 0.718032,
                }
            ]

    graph_provider = FakeGraphContextProvider()
    entity_retriever = FakeEntityRetriever()
    provider = VectorFallbackGraphContextProvider(
        graph_context_provider=graph_provider,
        entity_retriever=entity_retriever,
    )

    result = provider.get_node_neighborhood(
        node_id="进化神经网络",
        node_types=["Paper"],
        relationship_types=["CONTAINS"],
        page=2,
    )

    assert graph_provider.calls == [
        ("进化神经网络", ["Paper"], ["CONTAINS"], 2),
        ("neuroevolution of augmenting topologies (neat)", ["Paper"], ["CONTAINS"], 2),
    ]
    assert entity_retriever.calls == [("进化神经网络", 1)]
    assert result["entities"][0]["id"] == "neuroevolution of augmenting topologies (neat)"
    assert result["lookup_notes"] == [
        {
            "status": "vector_fallback",
            "requested_node_id": "进化神经网络",
            "resolved_node_id": "neuroevolution of augmenting topologies (neat)",
            "resolved_entity_type": "Method",
            "score": 0.718032,
            "message": "未精确匹配到请求节点，已采用向量相似度最高的非论文实体节点进行一跳查询。",
        }
    ]


def test_vector_fallback_provider_keeps_exact_match_result_without_vector_lookup():
    class FakeGraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            return {
                "entities": [{"id": node_id, "entity_type": "Method"}],
                "relationships": [],
                "page": page,
                "page_size": 20,
                "total_entities": 1,
                "total_relationships": 0,
            }

    class FakeEntityRetriever:
        def retrieve(self, query, top_k):
            raise AssertionError("vector fallback should not run for exact matches")

    provider = VectorFallbackGraphContextProvider(
        graph_context_provider=FakeGraphContextProvider(),
        entity_retriever=FakeEntityRetriever(),
    )

    result = provider.get_node_neighborhood("known method")

    assert result["entities"] == [{"id": "known method", "entity_type": "Method"}]
    assert "lookup_notes" not in result
