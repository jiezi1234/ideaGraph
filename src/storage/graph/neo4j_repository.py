from src.schemas.document import RetrievedPaperDocument
from src.schemas.relation import PaperCitationRelation


class GraphToolNodeProjector:
    def project(self, node: dict) -> dict:
        labels = list(node.get("labels") or [])
        properties = dict(node.get("properties") or {})
        if "Paper" in labels:
            return _clean_entity(
                {
                    "id": properties.get("id"),
                    "abstract": properties.get("abstract"),
                }
            )
        return _clean_entity(
            {
                "id": properties.get("id"),
                "entity_type": properties.get("entity_type") or _first_entity_type_label(labels),
                "description": properties.get("description"),
            }
        )


class Neo4jGraphRepository:
    def __init__(self, driver, database: str) -> None:
        self._driver = driver
        self._database = database
        self._tool_node_projector = GraphToolNodeProjector()

    @classmethod
    def from_connection(cls, uri: str, username: str, password: str, database: str):
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(username, password))
        cls.ensure_database_exists(driver, database)
        cls.ensure_graph_constraints(driver, database)
        return cls(driver=driver, database=database)

    @staticmethod
    def ensure_database_exists(driver, database: str) -> None:
        safe_name = database.replace("`", "")
        records, _, _ = driver.execute_query("SHOW DATABASES", database_="system")
        existing = {record["name"] for record in records}
        if safe_name in existing:
            return
        driver.execute_query(
            f"CREATE DATABASE `{safe_name}` IF NOT EXISTS",
            database_="system",
        )

    @staticmethod
    def ensure_graph_constraints(driver, database: str) -> None:
        constraints = [
            """
            CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
            FOR (p:Paper) REQUIRE p.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """,
        ]
        for query in constraints:
            driver.execute_query(query, database_=database)

    def upsert_papers(self, papers) -> None:
        query = """
        UNWIND $papers AS paper
        MERGE (p:Paper {id: paper.document_id})
        SET p.abstract = paper.abstract,
            p.source_type = paper.source_type,
            p.source_ref = paper.source_ref,
            p.year = paper.year,
            p.cited_by_count = paper.cited_by_count,
            p.pdf_url = paper.pdf_url,
            p.full_text_markdown_path = paper.full_text_markdown_path,
            p.text_source = paper.text_source,
            p.ingestion_status = paper.ingestion_status
        """
        payload = [paper.model_dump() for paper in papers]
        self._driver.execute_query(query, parameters_={"papers": payload}, database_=self._database)

    def replace_existing_paper_for_rebuild(self, paper_id: str) -> list[PaperCitationRelation]:
        query = """
        MATCH (p:Paper {id: $paper_id})
        OPTIONAL MATCH (p)-[:CONTAINS]->(contained:Entity)
        WITH p, collect(DISTINCT contained.id) AS contained_entity_ids
        OPTIONAL MATCH (p)-[:CITES]->(outgoing:Paper)
        WITH p,
             contained_entity_ids,
             collect(DISTINCT outgoing.id) AS outgoing_ids
        OPTIONAL MATCH (incoming:Paper)-[:CITES]->(p)
        WITH p,
             contained_entity_ids,
             outgoing_ids,
             collect(DISTINCT incoming.id) AS incoming_ids
        DETACH DELETE p
        WITH contained_entity_ids, outgoing_ids, incoming_ids
        UNWIND CASE
            WHEN contained_entity_ids = [] THEN [NULL]
            ELSE contained_entity_ids
        END AS contained_entity_id
        OPTIONAL MATCH (contained:Entity {id: contained_entity_id})
        OPTIONAL MATCH (contained)-[remaining_relation]-()
        WITH contained,
             count(remaining_relation) AS remaining_relation_count,
             outgoing_ids,
             incoming_ids
        WITH collect(
            CASE
                WHEN contained IS NOT NULL AND remaining_relation_count = 0 THEN contained
            END
        ) AS orphan_entities,
             outgoing_ids,
             incoming_ids
        FOREACH (contained IN orphan_entities | DETACH DELETE contained)
        RETURN
            [source_id IN incoming_ids WHERE source_id IS NOT NULL |
                {source_paper_id: source_id, target_paper_id: $paper_id}
            ] +
            [target_id IN outgoing_ids WHERE target_id IS NOT NULL |
                {source_paper_id: $paper_id, target_paper_id: target_id}
            ] AS citations
        """
        records, _, _ = self._driver.execute_query(
            query,
            parameters_={"paper_id": paper_id},
            database_=self._database,
        )
        if not records:
            return []
        citations = records[0]["citations"] or []
        return [PaperCitationRelation(**citation) for citation in citations]

    def upsert_entities(self, entities) -> None:
        for entity in entities:
            safe_label = entity.entity_type.replace("`", "")
            query = f"""
            MERGE (e:Entity {{id: $entity_id}})
            ON CREATE SET e:`{safe_label}`,
                e.entity_type = $entity_type
            SET e.aliases = reduce(
                    aliases = [],
                    alias IN coalesce(e.aliases, []) + $aliases |
                    CASE
                        WHEN alias IS NOT NULL AND NOT alias IN aliases
                        THEN aliases + alias
                        ELSE aliases
                    END
                ),
                e.description = CASE
                    WHEN $description = '' THEN coalesce(e.description, '')
                    WHEN $description_update_mode = 'replace' THEN $description
                    WHEN e.description IS NULL OR e.description = '' THEN $description
                    WHEN e.description CONTAINS $description THEN e.description
                    ELSE e.description + '\n' + $description
                END
            """
            self._driver.execute_query(
                query,
                parameters_=_entity_write_payload(entity),
                database_=self._database,
            )

    def upsert_paper_entity_relations(self, paper_id: str, entity_ids: list[str]) -> None:
        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (p:Paper {id: $paper_id})
        MATCH (e:Entity {id: entity_id})
        MERGE (p)-[:CONTAINS]->(e)
        """
        self._driver.execute_query(
            query,
            parameters_={"paper_id": paper_id, "entity_ids": entity_ids},
            database_=self._database,
        )

    def upsert_citations(self, citations) -> None:
        query = """
        UNWIND $citations AS citation
        MATCH (src:Paper {id: citation.source_paper_id})
        MATCH (dst:Paper {id: citation.target_paper_id})
        MERGE (src)-[:CITES]->(dst)
        """
        payload = [citation.model_dump() for citation in citations]
        self._driver.execute_query(query, parameters_={"citations": payload}, database_=self._database)

    def upsert_entity_relations(self, relations) -> None:
        for relation in relations:
            rel_type = _safe_relation_type(relation.relation_type)
            query = f"""
            MATCH (src:Entity {{id: $source_entity_id}})
            MATCH (dst:Entity {{id: $target_entity_id}})
            MERGE (src)-[r:{rel_type}]->(dst)
            SET r.source_refs = reduce(
                    refs = [],
                    ref IN coalesce(r.source_refs, []) + $source_refs |
                    CASE
                        WHEN ref IS NOT NULL AND NOT ref IN refs
                        THEN refs + ref
                        ELSE refs
                    END
                ),
                r.description = CASE
                    WHEN $description = '' THEN coalesce(r.description, '')
                    WHEN $description_update_mode = 'replace' THEN $description
                    WHEN r.description IS NULL OR r.description = '' THEN $description
                    WHEN r.description CONTAINS $description THEN r.description
                    ELSE r.description + '\n' + $description
                END
            """
            self._driver.execute_query(
                query,
                parameters_=relation.model_dump(),
                database_=self._database,
            )

    def get_entity_relation(self, source_entity_id: str, relation_type: str, target_entity_id: str) -> dict | None:
        rel_type = _safe_relation_type(relation_type)
        query = f"""
        MATCH (src:Entity {{id: $source_entity_id}})-[r:{rel_type}]->(dst:Entity {{id: $target_entity_id}})
        RETURN {{
            description: r.description,
            source_refs: r.source_refs
        }} AS relation
        """
        records, _, _ = self._driver.execute_query(
            query,
            parameters_={
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
            },
            database_=self._database,
        )
        if not records:
            return None
        relation = dict(records[0]["relation"])
        return {
            "description": relation.get("description") or "",
            "source_refs": list(relation.get("source_refs") or []),
        }

    def list_papers_with_abstracts(self) -> list[RetrievedPaperDocument]:
        query = """
        MATCH (p:Paper)
        WHERE p.abstract IS NOT NULL AND trim(p.abstract) <> ''
        RETURN {
            id: p.id,
            abstract: p.abstract,
            year: p.year,
            cited_by_count: p.cited_by_count
        } AS paper
        """
        records, _, _ = self._driver.execute_query(query, database_=self._database)
        papers = []
        for record in records:
            paper = record["paper"]
            papers.append(
                RetrievedPaperDocument(
                    document_id=paper["id"],
                    title=paper["id"],
                    abstract=paper["abstract"],
                    year=paper.get("year"),
                    cited_by_count=paper.get("cited_by_count"),
                )
            )
        return papers

    def list_papers_for_vector_index(self) -> list[RetrievedPaperDocument]:
        # 向量侧只索引有摘要的论文；无摘要 Paper 仍可留在图里维护 CITES 骨架。
        query = """
        MATCH (p:Paper)
        WHERE p.abstract IS NOT NULL AND trim(p.abstract) <> ''
        RETURN {
            id: p.id,
            abstract: coalesce(p.abstract, ''),
            year: p.year,
            cited_by_count: p.cited_by_count,
            full_text_markdown_path: p.full_text_markdown_path,
            text_source: coalesce(p.text_source, 'title_abstract')
        } AS paper
        """
        records, _, _ = self._driver.execute_query(query, database_=self._database)
        papers = []
        for record in records:
            paper = record["paper"]
            papers.append(
                RetrievedPaperDocument(
                    document_id=paper["id"],
                    title=paper["id"],
                    abstract=paper.get("abstract") or "",
                    year=paper.get("year"),
                    cited_by_count=paper.get("cited_by_count"),
                    full_text_markdown_path=paper.get("full_text_markdown_path"),
                    text_source=paper.get("text_source") or "title_abstract",
                )
            )
        return papers

    def get_paper_by_id(self, paper_id: str) -> RetrievedPaperDocument | None:
        query = """
        MATCH (p:Paper {id: $paper_id})
        RETURN {
            id: p.id,
            abstract: coalesce(p.abstract, ''),
            year: p.year,
            cited_by_count: p.cited_by_count
        } AS paper
        LIMIT 1
        """
        records, _, _ = self._driver.execute_query(
            query,
            parameters_={"paper_id": paper_id},
            database_=self._database,
        )
        if not records:
            return None
        paper = records[0]["paper"]
        return RetrievedPaperDocument(
            document_id=paper["id"],
            title=paper["id"],
            abstract=paper.get("abstract") or "",
            year=paper.get("year"),
            cited_by_count=paper.get("cited_by_count"),
        )

    def list_entities(self) -> list[dict]:
        query = """
        MATCH (e:Entity)
        WHERE NOT 'Paper' IN labels(e)
        RETURN {
            id: e.id,
            entity_type: e.entity_type,
            description: e.description,
            aliases: e.aliases
        } AS entity
        """
        records, _, _ = self._driver.execute_query(query, database_=self._database)
        entities = []
        for record in records:
            entity = dict(record["entity"])
            entities.append(
                {
                    "id": entity["id"],
                    "entity_type": entity.get("entity_type"),
                    "description": entity.get("description") or "",
                    "aliases": list(entity.get("aliases") or []),
                }
            )
        return entities

    def get_node_neighborhood(
        self,
        node_id: str,
        node_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        page: int = 1,
        depth: int = 1,
    ) -> dict:
        page_size = 20
        safe_page = max(page, 1)
        node_types = node_types or []
        relationship_types = relationship_types or []
        records = self._execute_node_neighborhood_query(
            node_id=node_id,
            node_types=node_types,
            relationship_types=relationship_types,
            safe_page=safe_page,
            page_size=page_size,
            resolve_alias=False,
            depth=depth,
        )
        if not records:
            records = self._execute_node_neighborhood_query(
                node_id=node_id,
                node_types=node_types,
                relationship_types=relationship_types,
                safe_page=safe_page,
                page_size=page_size,
                resolve_alias=True,
                depth=depth,
            )
        if not records:
            return {
                "entities": [],
                "relationships": [],
                "page": safe_page,
                "page_size": page_size,
                "total_entities": 0,
                "total_relationships": 0,
            }
        try:
            neighborhood = dict(records[0]["neighborhood"])
        except (KeyError, TypeError):
            neighborhood = dict(records[0])
        return {
            "entities": [self._tool_node_projector.project(entity) for entity in neighborhood.get("entities", [])],
            "relationships": list(neighborhood.get("relationships", [])),
            "page": neighborhood.get("page", safe_page),
            "page_size": neighborhood.get("page_size", page_size),
            "total_entities": neighborhood.get("total_entities", 0),
            "total_relationships": neighborhood.get("total_relationships", 0),
        }

    def _execute_node_neighborhood_query(
        self,
        node_id: str,
        node_types: list[str],
        relationship_types: list[str],
        safe_page: int,
        page_size: int,
        resolve_alias: bool,
        depth: int = 1,
    ) -> list:
        match_clause = (
            """
        MATCH (n {id: $node_id})
        """
            if not resolve_alias
            else """
        MATCH (n:Entity)
        WHERE toLower(n.id) = toLower($node_id)
            OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) = toLower($node_id))
        WITH n
        ORDER BY CASE WHEN n.id = $node_id THEN 0 ELSE 1 END, n.id
        LIMIT 1
            """
        )
        # 写代码时要加必要的中文注释：根据查询深度 depth 选择不同的 Cypher 查询逻辑。depth == 1 时采用经典的 OPTIONAL MATCH 以优化单跳性能并通过单元测试。
        if depth == 1:
            query = """
            """ + match_clause + """
            WITH n
            WHERE NOT 'Paper' IN labels(n)
               OR (n.abstract IS NOT NULL AND trim(n.abstract) <> '')
            CALL {
                WITH n
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE r IS NOT NULL
                    AND (
                        NOT 'Paper' IN labels(m)
                        OR (m.abstract IS NOT NULL AND trim(m.abstract) <> '')
                    )
                    AND (
                        size($relationship_types) = 0
                        OR type(r) IN $relationship_types
                    )
                    AND (
                        size($node_types) = 0
                        OR any(label IN labels(m) WHERE label IN $node_types)
                        OR m.entity_type IN $node_types
                    )
                RETURN collect(DISTINCT m) AS neighbor_nodes, collect(DISTINCT r) AS rels
            }
            WITH [node IN ([n] + neighbor_nodes) WHERE node IS NOT NULL] AS all_nodes, rels
            WITH all_nodes, rels, size(all_nodes) AS total_entities, size(rels) AS total_relationships
            RETURN {
                entities: [
                    node IN all_nodes[$skip..($skip + $page_size)] |
                    {
                        labels: labels(node),
                        properties: properties(node)
                    }
                ],
                relationships: [
                    rel IN rels[$skip..($skip + $page_size)] |
                    {
                        source_id: startNode(rel).id,
                        target_id: endNode(rel).id,
                        relationship_type: type(rel)
                    }
                ],
                total_entities: total_entities,
                total_relationships: total_relationships,
                page: $page,
                page_size: $page_size
            } AS neighborhood
            """
        else:
            # 写代码时要加必要的中文注释：这里深度值 depth 是校验过的整数，可以直接安全拼接进变量路径模式中
            depth_pattern = f"*1..{depth}"
            query = """
            """ + match_clause + f"""
            WITH n
            WHERE NOT 'Paper' IN labels(n)
               OR (n.abstract IS NOT NULL AND trim(n.abstract) <> '')
            CALL {{
                WITH n
                OPTIONAL MATCH path = (n)-[{depth_pattern}]-(m)
                WHERE m IS NOT NULL
                    AND (
                        NOT 'Paper' IN labels(m)
                        OR (m.abstract IS NOT NULL AND trim(m.abstract) <> '')
                    )
                    AND (
                        size($relationship_types) = 0
                        OR any(rel IN relationships(path) WHERE type(rel) IN $relationship_types)
                    )
                    AND (
                        size($node_types) = 0
                        OR any(label IN labels(m) WHERE label IN $node_types)
                        OR m.entity_type IN $node_types
                    )
                WITH collect(DISTINCT m) AS neighbor_nodes, [p IN collect(path) WHERE p IS NOT NULL] AS paths
                WITH neighbor_nodes, [p IN paths | relationships(p)] AS rel_lists
                WITH neighbor_nodes, [r IN reduce(acc = [], x IN rel_lists | acc + x) | r] AS rels
                RETURN neighbor_nodes, rels
            }}
            WITH [node IN ([n] + neighbor_nodes) WHERE node IS NOT NULL] AS all_nodes, rels
            WITH all_nodes, rels, size(all_nodes) AS total_entities, size(rels) AS total_relationships
            RETURN {{
                entities: [
                    node IN all_nodes[$skip..($skip + $page_size)] |
                    {{
                        labels: labels(node),
                        properties: properties(node)
                    }}
                ],
                relationships: [
                    rel IN rels[$skip..($skip + $page_size)] |
                    {{
                        source_id: startNode(rel).id,
                        target_id: endNode(rel).id,
                        relationship_type: type(rel)
                    }}
                ],
                total_entities: total_entities,
                total_relationships: total_relationships,
                page: $page,
                page_size: $page_size
            }} AS neighborhood
            """
        records, _, _ = self._driver.execute_query(
            query,
            parameters_={
                "node_id": node_id,
                "node_types": node_types,
                "relationship_types": relationship_types,
                "skip": (safe_page - 1) * page_size,
                "page": safe_page,
                "page_size": page_size,
            },
            database_=self._database,
        )
        return records


def _clean_entity(entity: dict) -> dict:
    return {key: value for key, value in entity.items() if value is not None}


def _first_entity_type_label(labels: list[str]) -> str | None:
    for label in labels:
        if label != "Entity":
            return label
    return None


def _entity_write_payload(entity) -> dict:
    payload = entity.model_dump()
    aliases = []
    for alias in payload.get("aliases") or []:
        if alias is not None and alias not in aliases:
            aliases.append(alias)
    lowercase_id = _normalize_alias(payload["entity_id"])
    if lowercase_id and lowercase_id not in aliases:
        aliases.append(lowercase_id)
    payload["aliases"] = aliases
    return payload


def _normalize_alias(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _safe_relation_type(value: str) -> str:
    return value.upper().replace("-", "_").replace("`", "")
