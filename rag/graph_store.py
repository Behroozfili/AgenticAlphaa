"""
rag/graph_store.py — AlphaGraphStore
Entity extraction via Claude + Neo4j knowledge graph population.

Pipeline:
    RawDocument
        │
        ▼
    Claude (entity + relation extraction)
        │
        ▼
    Neo4j (MERGE nodes + relationships)

Entity types recognized:
    Company, Person, GeopoliticalEvent, MacroEvent, Product, Sector

Relationship types:
    COMPETES_WITH, SUPPLIES_TO, AFFECTED_BY, LED_BY,
    PART_OF, RELATED_TO, ACQUIRED_BY
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

from core.observability import sentry_enabled

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data schemas
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    name: str
    type: str       # Company | Person | GeopoliticalEvent | MacroEvent | Product | Sector
    ticker: str = ""
    description: str = ""


@dataclass
class Relation:
    source: str     # entity name
    target: str     # entity name
    rel_type: str   # COMPETES_WITH | SUPPLIES_TO | AFFECTED_BY | ...
    weight: float = 1.0
    evidence: str = ""  # short quote from text supporting this relation


@dataclass
class GraphDocument:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    source_url: str = ""
    ticker: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a financial knowledge graph extractor.

Given the following financial news text, extract:
1. Named entities (companies, people, events, products, sectors)
2. Relationships between those entities

TEXT:
{text}

TICKER CONTEXT: {ticker}

Rules:
- Only extract entities clearly mentioned in the text
- Use the canonical company name (e.g. "NVIDIA" not "Nvidia Corporation")
- Relationship types must be one of:
  COMPETES_WITH, SUPPLIES_TO, AFFECTED_BY, LED_BY, PART_OF, RELATED_TO, ACQUIRED_BY
- Keep evidence under 15 words
- If nothing meaningful can be extracted, return empty lists

Return ONLY valid JSON, no markdown fences:
{{
  "entities": [
    {{"name": "NVIDIA", "type": "Company", "ticker": "NVDA", "description": "GPU manufacturer"}},
    {{"name": "Jensen Huang", "type": "Person", "ticker": "", "description": "CEO of NVIDIA"}}
  ],
  "relations": [
    {{
      "source": "NVIDIA",
      "target": "AMD",
      "rel_type": "COMPETES_WITH",
      "weight": 1.0,
      "evidence": "NVIDIA and AMD compete in the GPU market"
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# AlphaGraphStore
# ─────────────────────────────────────────────────────────────────────────────

class AlphaGraphStore:
    """
    Extracts entities and relationships from RawDocuments using Claude,
    then persists them to Neo4j as a knowledge graph.

    Usage:
        store = AlphaGraphStore()
        graph_docs = store.extract_batch(raw_docs)
        store.upsert_batch(graph_docs)
    """

    EXTRACT_MODEL = "claude-sonnet-4-20250514"

    VALID_ENTITY_TYPES = {
        "Company", "Person", "GeopoliticalEvent",
        "MacroEvent", "Product", "Sector",
    }
    VALID_REL_TYPES = {
        "COMPETES_WITH", "SUPPLIES_TO", "AFFECTED_BY",
        "LED_BY", "PART_OF", "RELATED_TO", "ACQUIRED_BY",
    }

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        self._claude = anthropic.Anthropic(
            api_key=anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self.max_tokens = max_tokens

        # Neo4j — optional; graceful degradation if not configured
        self._driver = None
        uri      = neo4j_uri      or os.environ.get("NEO4J_URI")
        user     = neo4j_user     or os.environ.get("NEO4J_USER", "neo4j")
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD")

        if uri and password:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(uri, auth=(user, password))
                self._driver.verify_connectivity()
                logger.info("AlphaGraphStore connected to Neo4j at %s", uri)
                self._ensure_constraints()
            except Exception as exc:
                logger.warning("Neo4j connection failed: %s — graph writes disabled.", exc)
        else:
            logger.warning("NEO4J_URI / NEO4J_PASSWORD not set — graph writes disabled.")

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def extract_batch(
        self,
        raw_docs,                       # list[RawDocument]
        max_chars_per_doc: int = 1500,
    ) -> list[GraphDocument]:
        """
        Run Claude entity extraction on each RawDocument.
        Skips documents that are too short to contain useful entities.
        Returns a list of GraphDocuments (may be empty lists inside).
        """
        results: list[GraphDocument] = []

        for doc in raw_docs:
            text = f"{doc.title}\n\n{doc.content}".strip()
            if len(text) < 80:
                logger.debug("Skipping short doc: %s", doc.url)
                continue

            graph_doc = self._extract_one(
                text=text[:max_chars_per_doc],
                ticker=doc.ticker,
                source_url=doc.url,
            )
            results.append(graph_doc)

        logger.info(
            "Extracted %d graph documents from %d raw docs.",
            len(results), len(raw_docs),
        )
        return results

    def upsert_batch(self, graph_docs: list[GraphDocument]) -> dict[str, int]:
        """
        Write all entities and relationships to Neo4j.
        Uses MERGE so it's fully idempotent — safe to run repeatedly.

        Returns a summary dict: {nodes_merged, rels_merged, skipped}.
        """
        if self._driver is None:
            logger.warning("Neo4j not available — skipping graph upsert.")
            return {"nodes_merged": 0, "rels_merged": 0, "skipped": len(graph_docs)}

        nodes_merged = 0
        rels_merged  = 0

        with self._driver.session() as session:
            for graph_doc in graph_docs:
                # 1. Upsert entities
                for entity in graph_doc.entities:
                    session.execute_write(self._merge_entity, entity)
                    nodes_merged += 1

                # 2. Upsert relationships
                for rel in graph_doc.relations:
                    session.execute_write(self._merge_relation, rel, graph_doc.source_url)
                    rels_merged += 1

        logger.info(
            "Graph upsert complete: %d nodes, %d relationships.",
            nodes_merged, rels_merged,
        )
        return {"nodes_merged": nodes_merged, "rels_merged": rels_merged, "skipped": 0}

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver:
            self._driver.close()

    # ─────────────────────────────────────────────────────────────────
    # Extraction (Claude)
    # ─────────────────────────────────────────────────────────────────

    def _extract_one(
        self, text: str, ticker: str, source_url: str
    ) -> GraphDocument:
        prompt = _EXTRACTION_PROMPT.format(text=text, ticker=ticker)

        try:
            response = self._claude.messages.create(
                model=self.EXTRACT_MODEL,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        except Exception as exc:
            logger.error("Claude extraction failed for %s: %s", source_url, exc)
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "graph_store.extract")
                    scope.set_tag("source_url", source_url[:100])
                    sentry_sdk.capture_exception(exc)
            return GraphDocument(source_url=source_url, ticker=ticker)

        return self._parse_graph_doc(raw, source_url=source_url, ticker=ticker)

    # ─────────────────────────────────────────────────────────────────
    # Parsing & validation
    # ─────────────────────────────────────────────────────────────────

    def _parse_graph_doc(
        self, raw: str, source_url: str, ticker: str
    ) -> GraphDocument:
        """Parse Claude's JSON response into a validated GraphDocument."""
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed for %s: %s", source_url, raw[:200])
            return GraphDocument(source_url=source_url, ticker=ticker)

        entities = []
        for e in data.get("entities", []):
            name  = (e.get("name") or "").strip()
            etype = (e.get("type") or "").strip()
            if not name:
                continue
            if etype not in self.VALID_ENTITY_TYPES:
                etype = "Company"   # safe default
            entities.append(Entity(
                name=name,
                type=etype,
                ticker=(e.get("ticker") or "").strip().upper(),
                description=(e.get("description") or "")[:200],
            ))

        relations = []
        entity_names = {e.name for e in entities}
        for r in data.get("relations", []):
            src      = (r.get("source") or "").strip()
            tgt      = (r.get("target") or "").strip()
            rel_type = (r.get("rel_type") or "").strip().upper()

            if not src or not tgt:
                continue
            if rel_type not in self.VALID_REL_TYPES:
                rel_type = "RELATED_TO"  # safe default

            relations.append(Relation(
                source=src,
                target=tgt,
                rel_type=rel_type,
                weight=float(r.get("weight", 1.0)),
                evidence=(r.get("evidence") or "")[:200],
            ))

        return GraphDocument(
            entities=entities,
            relations=relations,
            source_url=source_url,
            ticker=ticker,
        )

    # ─────────────────────────────────────────────────────────────────
    # Neo4j write transactions
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_entity(tx, entity: Entity) -> None:
        """MERGE on name — idempotent node upsert."""
        cypher = f"""
        MERGE (e:{entity.type} {{name: $name}})
        ON CREATE SET
            e.ticker      = $ticker,
            e.description = $description,
            e.created_at  = timestamp()
        ON MATCH SET
            e.ticker      = CASE WHEN $ticker <> '' THEN $ticker ELSE e.ticker END,
            e.updated_at  = timestamp()
        """
        tx.run(cypher, name=entity.name, ticker=entity.ticker,
               description=entity.description)

    @staticmethod
    def _merge_relation(tx, rel: Relation, source_url: str) -> None:
        """MERGE relationship — idempotent edge upsert."""
        cypher = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[r:{rel.rel_type}]->(b)
        ON CREATE SET
            r.weight     = $weight,
            r.evidence   = $evidence,
            r.source_url = $source_url,
            r.created_at = timestamp()
        ON MATCH SET
            r.weight     = ($weight + r.weight) / 2.0,
            r.updated_at = timestamp()
        """
        tx.run(
            cypher,
            source=rel.source,
            target=rel.target,
            weight=rel.weight,
            evidence=rel.evidence,
            source_url=source_url,
        )

    # ─────────────────────────────────────────────────────────────────
    # Schema constraints (run once on first connect)
    # ─────────────────────────────────────────────────────────────────

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints and indexes if they don't exist."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company)          REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person)           REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:GeopoliticalEvent) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MacroEvent)       REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product)         REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector)           REQUIRE s.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
        ]
        with self._driver.session() as session:
            for stmt in constraints:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug("Constraint already exists or failed: %s", exc)
        logger.info("Neo4j constraints ensured.")