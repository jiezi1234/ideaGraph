from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DEFAULT_NODE_QUERY = (
    "MATCH (n) "
    "RETURN coalesce(n.id, toString(id(n))) AS node_id, "
    "labels(n) AS labels, properties(n) AS properties "
    "ORDER BY node_id"
)
DEFAULT_EDGE_QUERY = (
    "MATCH (a)-[r]->(b) "
    "RETURN toString(id(r)) AS edge_id, "
    "coalesce(a.id, toString(id(a))) AS source_id, "
    "coalesce(b.id, toString(id(b))) AS target_id, "
    "type(r) AS edge_type, properties(r) AS properties "
    "ORDER BY edge_id"
)


@dataclass(frozen=True)
class GraphmlExportConfig:
    graph_id: str = "G"
    edgedefault: str = "directed"
    include_labels: bool = True
    include_properties: bool = True
    include_label_keys: bool = False
    title_property_candidates: tuple[str, ...] = ("title", "name", "display_name")
    color_by_label: bool = True
    keep_numeric_properties: bool = False


@dataclass(frozen=True)
class GraphmlExportSummary:
    output_path: Path
    node_count: int
    edge_count: int
    query: str


def export_graphml_from_driver(
    driver: Any,
    database: str,
    output_path: str | Path,
    config: GraphmlExportConfig | None = None,
    node_query: str = DEFAULT_NODE_QUERY,
    edge_query: str = DEFAULT_EDGE_QUERY,
) -> GraphmlExportSummary:
    config = config or GraphmlExportConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_records = _execute_query(driver, node_query, database)
    edge_records = _execute_query(driver, edge_query, database)

    nodes = [record.data() for record in node_records]
    edges = [record.data() for record in edge_records]

    root = ET.Element(
        "graphml",
        attrib={"xmlns": "http://graphml.graphdrawing.org/xmlns"},
    )

    node_keys, edge_keys = _collect_keys(nodes, edges)
    _append_key_elements(root, node_keys, "node")
    _append_key_elements(root, edge_keys, "edge")

    graph = ET.SubElement(root, "graph", attrib={"id": config.graph_id, "edgedefault": config.edgedefault})

    for node in nodes:
        node_element = ET.SubElement(graph, "node", attrib={"id": str(node["node_id"])})
        labels = node.get("labels") or []
        _append_data(node_element, "labels", _serialize_labels(labels))
        properties = node.get("properties") or {}
        for key, value in properties.items():
            _append_data(node_element, key, _serialize_value(value))

    for edge in edges:
        source_id = edge.get("source_id", edge.get("source"))
        target_id = edge.get("target_id", edge.get("target"))
        edge_element = ET.SubElement(
            graph,
            "edge",
            attrib={
                "id": str(edge["edge_id"]),
                "source": str(source_id),
                "target": str(target_id),
            },
        )
        _append_data(edge_element, "type", _serialize_value(edge.get("edge_type", edge.get("type"))))
        properties = edge.get("properties") or {}
        for key, value in properties.items():
            _append_data(edge_element, key, _serialize_value(value))

    _indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return GraphmlExportSummary(
        output_path=output_path,
        node_count=len(nodes),
        edge_count=len(edges),
        query=node_query,
    )


def _execute_query(driver: Any, query: str, database: str):
    records, _, _ = driver.execute_query(query, database_=database)
    return records


def _collect_keys(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]):
    node_keys: dict[str, str] = {"labels": "string"}
    edge_keys: dict[str, str] = {"type": "string"}
    for node in nodes:
        properties = node.get("properties") or {}
        for key, value in properties.items():
            node_keys[key] = _graphml_type_for_value(value)
    for edge in edges:
        properties = edge.get("properties") or {}
        for key, value in properties.items():
            edge_keys[key] = _graphml_type_for_value(value)
    return node_keys, edge_keys


def _append_key_elements(root: ET.Element, keys: dict[str, str], scope: str) -> None:
    for key_name, attr_type in keys.items():
        ET.SubElement(
            root,
            "key",
            attrib={
                "id": key_name,
                "for": scope,
                "attr.name": key_name,
                "attr.type": attr_type,
            },
        )


def _append_data(parent: ET.Element, key: str, value: Any) -> None:
    if value is None:
        return
    data = ET.SubElement(parent, "data", attrib={"key": key})
    data.text = str(value)


def _serialize_labels(labels: list[str]) -> str:
    return ",".join(labels)


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _graphml_type_for_value(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "long"
    if isinstance(value, float):
        return "double"
    return "string"


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent
