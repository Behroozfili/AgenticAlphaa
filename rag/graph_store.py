"""Temporal, provenance-preserving knowledge-graph storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import anthropic
import sentry_sdk

from core.forecast_contract import parse_utc_datetime
from core.observability import elapsed_latency_ms, sentry_enabled

logger = logging.getLogger(__name__)
_STAGE_IDENTITY = {
    "stage": "04A", "stage_revision": 1, "attempt": 2, "agent": "data_rag",
}
_REQUIRED_TEMPORAL_IDENTITIES = (
    "document_id", "chunk_id", "evidence_id", "canonical_source_id",
    "content_hash", "document_version", "chunk_hash", "ticker", "dataset_version",
)
_REQUIRED_TEMPORAL_TIMES = (
    "published_at", "retrieved_at", "effective_from", "materialized_at",
)
_ALL_TEMPORAL_TIMES = (
    "published_at", "retrieved_at", "filed_at", "effective_from",
    "effective_to", "materialized_at",
)


class GraphUpsertError(RuntimeError):
    """Sanitized retry-safe graph write failure."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class Entity:
    name: str
    type: str
    ticker: str = ""
    description: str = ""

    @property
    def entity_id(self) -> str:
        return _sha256(f"{self.type.strip()}|{self.name.strip().casefold()}")


@dataclass
class Relation:
    source: str
    target: str
    rel_type: str
    weight: float = 1.0
    evidence: str = ""


@dataclass
class GraphDocument:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    source_url: str = ""
    ticker: str = ""
    document_id: str = ""
    chunk_id: str = ""
    evidence_id: str = ""
    canonical_source_id: str = ""
    content_hash: str = ""
    document_version: str = ""
    chunk_hash: str = ""
    chunk_index: int = 0
    dataset_version: str = "legacy-v1"
    published_at: str | None = None
    retrieved_at: str | None = None
    filed_at: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    materialized_at: str | None = None
    is_quarantined: bool = False
    write_mode: Literal["legacy", "temporal_v2"] = "temporal_v2"
    provenance: Literal["legacy_explicit", "stage03_temporal", "unknown"] = "unknown"

    @property
    def temporal_identity_complete(self) -> bool:
        return all(bool(getattr(self, field_name)) for field_name in _REQUIRED_TEMPORAL_IDENTITIES)

    @property
    def temporal_time_complete(self) -> bool:
        return all(bool(getattr(self, field_name)) for field_name in _REQUIRED_TEMPORAL_TIMES)

    @property
    def explicitly_legacy(self) -> bool:
        return self.write_mode == "legacy" and self.provenance == "legacy_explicit"


def _normalized_time(value: object) -> str:
    return parse_utc_datetime(value, field_name="graph_temporal_value").isoformat()


def _prepare_temporal_document(
    document: GraphDocument,
) -> tuple[GraphDocument | None, str | None]:
    """Normalize temporal values once or return a stable fail-closed code."""
    if document.write_mode != "temporal_v2" or document.provenance == "legacy_explicit":
        return None, "GRAPH_WRITE_MODE_INVALID"
    if document.is_quarantined:
        return None, "GRAPH_V2_QUARANTINED"
    if not document.temporal_identity_complete:
        return None, "GRAPH_V2_IDENTITY_INCOMPLETE"
    if not document.materialized_at:
        return None, "GRAPH_V2_MATERIALIZED_AT_MISSING"
    normalized: dict[str, str | None] = {}
    for field_name in _ALL_TEMPORAL_TIMES:
        raw = getattr(document, field_name)
        if raw is None:
            if field_name in _REQUIRED_TEMPORAL_TIMES:
                return None, "GRAPH_V2_TEMPORAL_REQUIRED"
            normalized[field_name] = None
            continue
        try:
            normalized[field_name] = _normalized_time(raw)
        except (TypeError, ValueError):
            return None, "GRAPH_V2_TEMPORAL_INVALID"
    if normalized["effective_to"] is not None:
        effective_from = parse_utc_datetime(normalized["effective_from"], field_name="effective_from")
        effective_to = parse_utc_datetime(normalized["effective_to"], field_name="effective_to")
        if effective_to <= effective_from:
            return None, "GRAPH_V2_EFFECTIVE_INTERVAL_INVALID"
    return replace(document, **normalized), None


_EXTRACTION_PROMPT = """You are a financial knowledge graph extractor.
Extract named entities and relationships from TEXT. Return JSON only.
TEXT:
{text}
TICKER CONTEXT: {ticker}
Relationship types: COMPETES_WITH, SUPPLIES_TO, AFFECTED_BY, LED_BY,
PART_OF, RELATED_TO, ACQUIRED_BY. Keep evidence under 15 words.
{{"entities": [], "relations": []}}"""


class AlphaGraphStore:
    """Writes legacy graph data or additive immutable temporal observations."""

    EXTRACT_MODEL = "claude-haiku-4-5"
    VALID_ENTITY_TYPES = {
        "Company", "Person", "GeopoliticalEvent", "MacroEvent", "Product", "Sector",
    }
    VALID_REL_TYPES = {
        "COMPETES_WITH", "SUPPLIES_TO", "AFFECTED_BY", "LED_BY", "PART_OF",
        "RELATED_TO", "ACQUIRED_BY",
    }

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("AlphaGraphStore requires ANTHROPIC_API_KEY")
        self._api_key = api_key
        self.max_tokens = max_tokens
        self._neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI")
        self._neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self._neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        self._claude = None
        self._driver = None
        self._extraction_failures = 0
        self.last_lifecycle: dict[str, object] = {
            **_STAGE_IDENTITY,
            "operation": "graph_idle",
            "status": "idle",
            "degraded": False,
            "resume": "not_applicable",
            "recovery": "caller_retry_deterministic_merge",
        }

    def connect(self) -> None:
        self._claude = anthropic.AsyncAnthropic(api_key=self._api_key)
        if self._neo4j_uri and self._neo4j_password:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password)
                )
                self._driver.verify_connectivity()
                self._ensure_constraints()
            except Exception:
                logger.warning("GRAPH_CONNECTION_FAILED")
        else:
            logger.warning("NEO4J_URI / NEO4J_PASSWORD not set; graph writes disabled")

    async def extract_batch(self, raw_docs, max_chars_per_doc: int = 1500) -> list[GraphDocument]:
        """Extract graph facts and attach the exact Stage-03 Vector identities."""
        from rag.processor import AlphaProcessor

        started_at = time.perf_counter()
        self._extraction_failures = 0
        results: list[GraphDocument] = []
        processor = AlphaProcessor()
        for doc in raw_docs:
            text = f"{doc.title}\n\n{doc.content}".strip()
            if len(text) < 80:
                continue
            graph_doc = await self._extract_one(text[:max_chars_per_doc], doc.ticker, doc.url)
            chunks = processor.process([doc])
            if not chunks:
                graph_doc.is_quarantined = True
                results.append(graph_doc)
                continue
            metadata = chunks[0].metadata
            graph_doc = replace(
                graph_doc,
                document_id=str(metadata.get("document_id") or ""),
                chunk_id=str(metadata.get("chunk_id") or ""),
                evidence_id=str(metadata.get("evidence_id") or ""),
                canonical_source_id=str(metadata.get("canonical_source_id") or ""),
                content_hash=str(metadata.get("content_hash") or ""),
                document_version=str(metadata.get("document_version") or ""),
                chunk_hash=str(metadata.get("chunk_hash") or ""),
                chunk_index=int(metadata.get("chunk_index") or 0),
                dataset_version=str(metadata.get("dataset_version") or "legacy-v1"),
                published_at=metadata.get("published_at"),
                retrieved_at=metadata.get("retrieved_at"),
                filed_at=metadata.get("filed_at"),
                effective_from=metadata.get("effective_from"),
                effective_to=metadata.get("effective_to"),
                materialized_at=(
                    doc.metadata.get("materialized_at")
                    or metadata.get("materialized_at")
                    or metadata.get("ingested_at")
                ),
                write_mode="temporal_v2",
                provenance="stage03_temporal",
            )
            graph_doc.is_quarantined = graph_doc.is_quarantined or not (
                graph_doc.temporal_identity_complete and graph_doc.temporal_time_complete
            )
            results.append(graph_doc)
        logger.info("Extracted %d graph documents from %d raw docs", len(results), len(raw_docs))
        entity_count = sum(len(item.entities) for item in results)
        relation_count = sum(len(item.relations) for item in results)
        self.last_lifecycle = {
            **_STAGE_IDENTITY,
            "operation": "graph_extract",
            "status": "degraded" if self._extraction_failures else "complete",
            "latency_ms": elapsed_latency_ms(started_at),
            "document_count": len(results),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "skipped_count": max(0, len(raw_docs) - len(results)),
            "failure_count": self._extraction_failures,
            "degraded": bool(self._extraction_failures),
            "error_code": "GRAPH_EXTRACTION_FAILED" if self._extraction_failures else None,
            "resume": "not_applicable",
            "recovery": "caller_retry_deterministic_merge",
        }
        return results

    def upsert_batch(self, graph_docs: list[GraphDocument]) -> dict[str, int]:
        started_at = time.perf_counter()
        if self._driver is None:
            logger.warning("Neo4j not available; skipping graph upsert")
            self.last_lifecycle = {
                **_STAGE_IDENTITY,
                "operation": "graph_upsert",
                "status": "degraded",
                "latency_ms": elapsed_latency_ms(started_at),
                "document_count": len(graph_docs),
                "entity_count": 0,
                "relation_count": 0,
                "skipped_count": len(graph_docs),
                "degraded": True,
                "error_code": "GRAPH_UNAVAILABLE",
                "resume": "not_applicable",
                "recovery": "caller_retry_deterministic_merge",
            }
            return {"nodes_merged": 0, "rels_merged": 0, "skipped": len(graph_docs)}
        nodes_merged = rels_merged = skipped = rejected = 0
        rejection_codes: set[str] = set()
        try:
            with self._driver.session() as session:
                for graph_doc in graph_docs:
                    if graph_doc.explicitly_legacy:
                        for entity in graph_doc.entities:
                            session.execute_write(self._merge_entity, entity)
                            nodes_merged += 1
                        for relation in graph_doc.relations:
                            session.execute_write(self._merge_relation, relation, graph_doc.source_url)
                            rels_merged += 1
                    else:
                        prepared, rejection_code = _prepare_temporal_document(graph_doc)
                        if prepared is None:
                            skipped += 1
                            rejected += 1
                            rejection_codes.add(rejection_code or "GRAPH_V2_WRITE_REJECTED")
                            continue
                        graph_doc = prepared
                        for entity in graph_doc.entities:
                            session.execute_write(self._merge_temporal_entity, entity)
                            nodes_merged += 1
                        for relation in graph_doc.relations:
                            session.execute_write(self._merge_temporal_observation, relation, graph_doc)
                            rels_merged += 1
                    if not graph_doc.relations and not graph_doc.entities:
                        skipped += 1
        except Exception:
            logger.error("GRAPH_UPSERT_FAILED")
            self.last_lifecycle = {
                **_STAGE_IDENTITY,
                "operation": "graph_upsert",
                "status": "failed",
                "latency_ms": elapsed_latency_ms(started_at),
                "document_count": len(graph_docs),
                "entity_count": nodes_merged,
                "relation_count": rels_merged,
                "skipped_count": skipped,
                "rejected_count": rejected,
                "rejection_codes": sorted(rejection_codes),
                "degraded": True,
                "error_code": "GRAPH_UPSERT_FAILED",
                "resume": "not_applicable",
                "recovery": "caller_retry_deterministic_merge",
            }
            if sentry_enabled():
                sentry_sdk.capture_message("GRAPH_UPSERT_FAILED", level="error")
            raise GraphUpsertError("GRAPH_UPSERT_FAILED") from None
        self.last_lifecycle = {
            **_STAGE_IDENTITY,
            "operation": "graph_upsert",
            "status": "degraded" if rejected else "complete",
            "latency_ms": elapsed_latency_ms(started_at),
            "document_count": len(graph_docs),
            "entity_count": nodes_merged,
            "relation_count": rels_merged,
            "skipped_count": skipped,
            "rejected_count": rejected,
            "rejection_codes": sorted(rejection_codes),
            "degraded": bool(rejected),
            "error_code": "GRAPH_V2_WRITE_REJECTED" if rejected else None,
            "resume": "not_applicable",
            "recovery": "caller_retry_deterministic_merge",
        }
        return {"nodes_merged": nodes_merged, "rels_merged": rels_merged, "skipped": skipped}

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    async def _extract_one(self, text: str, ticker: str, source_url: str) -> GraphDocument:
        prompt = _EXTRACTION_PROMPT.format(text=text, ticker=ticker)
        try:
            response = await self._claude.messages.create(
                model=self.EXTRACT_MODEL, max_tokens=self.max_tokens, temperature=0.0,
                messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "{"}],
            )
            raw = "{" + response.content[0].text
        except Exception:
            self._extraction_failures += 1
            logger.error("GRAPH_EXTRACTION_FAILED")
            if sentry_enabled():
                sentry_sdk.capture_message("GRAPH_EXTRACTION_FAILED", level="error")
            return GraphDocument(
                source_url="", ticker=ticker, is_quarantined=True,
                write_mode="temporal_v2", provenance="stage03_temporal",
            )
        return self._parse_graph_doc(raw, source_url=source_url, ticker=ticker)

    @staticmethod
    def _extract_json_block(raw: str) -> Optional[str]:
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                """
                if escaped:
                    escaped = False
                elif char == "\":
                    escaped = True
                elif char == '"':
                    in_string = False
                """
                if escaped:
                    escaped = False
                elif char == chr(92):
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:index + 1]
        return None

    def _parse_graph_doc(self, raw: str, source_url: str, ticker: str) -> GraphDocument:
        clean = self._extract_json_block(raw)
        if clean is None:
            return GraphDocument(source_url=source_url, ticker=ticker)
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return GraphDocument(source_url=source_url, ticker=ticker)
        entities: list[Entity] = []
        for item in data.get("entities", []):
            name = (item.get("name") or "").strip()
            entity_type = (item.get("type") or "").strip()
            if not name:
                continue
            if entity_type not in self.VALID_ENTITY_TYPES:
                entity_type = "Company"
            entities.append(Entity(
                name=name, type=entity_type,
                ticker=(item.get("ticker") or "").strip().upper(),
                description=(item.get("description") or "")[:200],
            ))
        relations: list[Relation] = []
        for item in data.get("relations", []):
            source = (item.get("source") or "").strip()
            target = (item.get("target") or "").strip()
            relation_type = (item.get("rel_type") or "").strip().upper()
            if not source or not target:
                continue
            if relation_type not in self.VALID_REL_TYPES:
                relation_type = "RELATED_TO"
            relations.append(Relation(
                source=source, target=target, rel_type=relation_type,
                weight=float(item.get("weight", 1.0)),
                evidence=(item.get("evidence") or "")[:200],
            ))
        return GraphDocument(entities, relations, source_url, ticker)

    @staticmethod
    def _merge_entity(tx, entity: Entity) -> None:
        cypher = f"""
        MERGE (e:{entity.type} {{name: $name}})
        ON CREATE SET e.ticker=$ticker, e.description=$description, e.created_at=timestamp()
        ON MATCH SET e.ticker=CASE WHEN $ticker <> '' THEN $ticker ELSE e.ticker END,
                     e.updated_at=timestamp()
        """
        tx.run(cypher, name=entity.name, ticker=entity.ticker, description=entity.description)

    @staticmethod
    def _merge_relation(tx, relation: Relation, source_url: str) -> None:
        cypher = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{relation.rel_type}]->(b)
        ON CREATE SET r.weight=$weight, r.evidence=$evidence,
                      r.source_url=$source_url, r.created_at=timestamp()
        ON MATCH SET r.weight=($weight + r.weight) / 2.0, r.updated_at=timestamp()
        """
        tx.run(cypher, source=relation.source, target=relation.target,
               weight=relation.weight, evidence=relation.evidence, source_url=source_url)

    @staticmethod
    def _merge_temporal_entity(tx, entity: Entity) -> None:
        tx.run(
            """
            MERGE (e:TemporalEntityV2 {entity_id: $entity_id})
            ON CREATE SET e.name=$name, e.type=$type, e.ticker=$ticker,
                          e.description=$description, e.created_at=timestamp()
            """,
            entity_id=entity.entity_id, name=entity.name, type=entity.type,
            ticker=entity.ticker, description=entity.description,
        )

    @staticmethod
    def _merge_temporal_observation(tx, relation: Relation, document: GraphDocument) -> None:
        observation_id = _sha256("|".join((
            document.evidence_id, relation.source.strip().casefold(),
            relation.target.strip().casefold(), relation.rel_type, relation.evidence,
        )))
        tx.run(
            """
            MERGE (d:AlphaDocumentV2 {document_id: $document_id})
            ON CREATE SET d.canonical_source_id=$canonical_source_id,
                          d.content_hash=$content_hash, d.document_version=$document_version,
                          d.dataset_version=$dataset_version, d.source_url=$source_url
            MERGE (c:AlphaChunkV2 {chunk_id: $chunk_id})
            ON CREATE SET c.document_id=$document_id, c.chunk_hash=$chunk_hash,
                          c.chunk_index=$chunk_index, c.dataset_version=$dataset_version
            MERGE (e:EvidenceObservationV2 {evidence_id: $evidence_id})
            ON CREATE SET e.chunk_id=$chunk_id, e.ticker=$ticker,
                          e.dataset_version=$dataset_version,
                          e.published_at=CASE WHEN $published_at IS NULL THEN null ELSE datetime($published_at) END,
                          e.retrieved_at=CASE WHEN $retrieved_at IS NULL THEN null ELSE datetime($retrieved_at) END,
                          e.filed_at=CASE WHEN $filed_at IS NULL THEN null ELSE datetime($filed_at) END,
                          e.effective_from=CASE WHEN $effective_from IS NULL THEN null ELSE datetime($effective_from) END,
                          e.effective_to=CASE WHEN $effective_to IS NULL THEN null ELSE datetime($effective_to) END,
                          e.materialized_at=CASE WHEN $materialized_at IS NULL THEN null ELSE datetime($materialized_at) END,
                          e.is_quarantined=$is_quarantined
            MERGE (o:TemporalRelationObservationV2 {observation_id: $observation_id})
            ON CREATE SET o.relation_type=$relation_type, o.weight=$weight,
                          o.evidence_text=$evidence_text
            MERGE (d)-[:HAS_CHUNK]->(c)
            MERGE (e)-[:OBSERVES_CHUNK]->(c)
            MERGE (e)-[:SUPPORTS_OBSERVATION]->(o)
            WITH e, o
            MATCH (a:TemporalEntityV2 {name: $source})
            MATCH (b:TemporalEntityV2 {name: $target})
            MERGE (o)-[:SOURCE_ENTITY]->(a)
            MERGE (o)-[:TARGET_ENTITY]->(b)
            MERGE (e)-[:MENTIONS_ENTITY]->(a)
            MERGE (e)-[:MENTIONS_ENTITY]->(b)
            """,
            document_id=document.document_id, canonical_source_id=document.canonical_source_id,
            content_hash=document.content_hash, document_version=document.document_version,
            dataset_version=document.dataset_version, source_url=document.source_url,
            chunk_id=document.chunk_id, chunk_hash=document.chunk_hash,
            chunk_index=document.chunk_index, evidence_id=document.evidence_id,
            ticker=document.ticker.upper(), published_at=document.published_at,
            retrieved_at=document.retrieved_at, filed_at=document.filed_at,
            effective_from=document.effective_from, effective_to=document.effective_to,
            materialized_at=document.materialized_at,
            is_quarantined=document.is_quarantined or not document.temporal_time_complete,
            observation_id=observation_id, relation_type=relation.rel_type,
            weight=relation.weight, evidence_text=relation.evidence,
            source=relation.source, target=relation.target,
        )

    def _ensure_constraints(self) -> None:
        # Legacy schema only. Stage-02 owns application of additive V2 constraints.
        statements = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:GeopoliticalEvent) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MacroEvent) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
        ]
        with self._driver.session() as session:
            for statement in statements:
                try:
                    session.run(statement)
                except Exception:
                    logger.debug("LEGACY_GRAPH_CONSTRAINT_FAILED")


__all__ = [
    "AlphaGraphStore", "Entity", "GraphDocument", "GraphUpsertError", "Relation",
    "_prepare_temporal_document",
]
