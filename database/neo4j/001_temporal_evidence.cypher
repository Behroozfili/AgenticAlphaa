// Stage 02 / Database V2
// Additive Neo4j 5.x schema for immutable temporal evidence observations.
// No legacy label, relationship, constraint, index, or data is removed.

CREATE CONSTRAINT alpha_document_v2_id_unique IF NOT EXISTS
FOR (document:AlphaDocumentV2)
REQUIRE document.document_id IS UNIQUE;

CREATE CONSTRAINT alpha_chunk_v2_id_unique IF NOT EXISTS
FOR (chunk:AlphaChunkV2)
REQUIRE chunk.chunk_id IS UNIQUE;

CREATE CONSTRAINT evidence_observation_v2_id_unique IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
REQUIRE evidence.evidence_id IS UNIQUE;

CREATE CONSTRAINT temporal_entity_v2_id_unique IF NOT EXISTS
FOR (entity:TemporalEntityV2)
REQUIRE entity.entity_id IS UNIQUE;

CREATE CONSTRAINT schema_route_component_unique IF NOT EXISTS
FOR (route:SchemaRoute)
REQUIRE route.component IS UNIQUE;

CREATE INDEX alpha_document_v2_source_identity IF NOT EXISTS
FOR (document:AlphaDocumentV2)
ON (
    document.canonical_source_id,
    document.content_hash,
    document.document_version
);

CREATE INDEX alpha_chunk_v2_document_position IF NOT EXISTS
FOR (chunk:AlphaChunkV2)
ON (chunk.document_id, chunk.chunk_index, chunk.chunk_hash);

CREATE INDEX evidence_observation_v2_ticker_dataset IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
ON (evidence.ticker, evidence.dataset_version);

CREATE INDEX evidence_observation_v2_published_at IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
ON (evidence.published_at);

CREATE INDEX evidence_observation_v2_filed_at IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
ON (evidence.filed_at);

CREATE INDEX evidence_observation_v2_effective_from IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
ON (evidence.effective_from);

CREATE INDEX evidence_observation_v2_effective_to IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
ON (evidence.effective_to);

CREATE INDEX evidence_observation_v2_quarantine IF NOT EXISTS
FOR (evidence:EvidenceObservationV2)
ON (evidence.is_quarantined, evidence.dataset_version);

MERGE (route:SchemaRoute {component: 'temporal_evidence'})
ON CREATE SET
    route.active_version = 'legacy-v1',
    route.previous_version = null,
    route.reconciliation_hash = null,
    route.activated_at = null,
    route.updated_at = datetime();

// Required V2 write shape for Stage 04A:
// - AlphaDocumentV2.document_id is the canonical source/content/version SHA-256.
// - AlphaChunkV2.chunk_id is shared with alpha_documents_v2.chunk_id.
// - EvidenceObservationV2.evidence_id is unique per source observation/version.
// - Every observation stores ticker, dataset_version, published_at, filed_at,
//   retrieved_at, effective_from, effective_to, materialized_at, and quarantine.
// - Unknown publication/filing/effective time remains null and quarantined.
// - Each observation links to one immutable chunk with OBSERVES_CHUNK and may
//   link to entities with MENTIONS_ENTITY. Relationship properties do not
//   overwrite observation history.
//
// Historical traversal eligibility:
//   evidence.is_quarantined = false
//   evidence.ticker = $ticker
//   evidence.dataset_version = $dataset_version
//   coalesce(evidence.filed_at, evidence.published_at, evidence.effective_from)
//       <= $as_of
//   (evidence.effective_from IS NULL OR evidence.effective_from <= $as_of)
//   (evidence.effective_to IS NULL OR evidence.effective_to > $as_of)
//
// The route remains legacy-v1. A later authorized reconciliation may set
// active_version to temporal-evidence-v2 while preserving previous_version.
// Rollback restores the previous pointer only; V2 nodes remain immutable.

MATCH (route:SchemaRoute {component: 'temporal_evidence'})
RETURN
    route.active_version AS active_version,
    route.previous_version AS previous_version,
    route.reconciliation_hash AS reconciliation_hash;

// Read-only count/hash reconciliation. The query returns a canonical ordered
// identity digest input because core Neo4j has no portable SHA-256 function.
// The migration runner records SHA-256(UTF-8(ordered_identity_digest_input))
// alongside these counts before any route switch is authorized.
MATCH (document:AlphaDocumentV2)
WITH
    'AlphaDocumentV2' AS relation_name,
    coalesce(document.dataset_version, 'unknown') AS dataset_version,
    document.document_id + ':' +
        coalesce(document.content_hash, '') + ':' +
        coalesce(document.document_version, '') AS identity_row
ORDER BY identity_row
WITH relation_name, dataset_version, collect(identity_row) AS identity_rows
RETURN
    relation_name,
    dataset_version,
    size(identity_rows) AS row_count,
    0 AS quarantined_count,
    0 AS unknown_time_count,
    'SHA-256' AS reconciliation_algorithm,
    reduce(
        digest_input = '',
        identity IN identity_rows |
        digest_input + CASE WHEN digest_input = '' THEN '' ELSE '|' END + identity
    ) AS ordered_identity_digest_input
UNION ALL
MATCH (chunk:AlphaChunkV2)
WITH
    'AlphaChunkV2' AS relation_name,
    coalesce(chunk.dataset_version, 'unknown') AS dataset_version,
    chunk.chunk_id + ':' +
        coalesce(chunk.document_id, '') + ':' +
        coalesce(toString(chunk.chunk_index), '') + ':' +
        coalesce(chunk.chunk_hash, '') AS identity_row
ORDER BY identity_row
WITH relation_name, dataset_version, collect(identity_row) AS identity_rows
RETURN
    relation_name,
    dataset_version,
    size(identity_rows) AS row_count,
    0 AS quarantined_count,
    0 AS unknown_time_count,
    'SHA-256' AS reconciliation_algorithm,
    reduce(
        digest_input = '',
        identity IN identity_rows |
        digest_input + CASE WHEN digest_input = '' THEN '' ELSE '|' END + identity
    ) AS ordered_identity_digest_input
UNION ALL
MATCH (evidence:EvidenceObservationV2)
WITH
    'EvidenceObservationV2' AS relation_name,
    coalesce(evidence.dataset_version, 'unknown') AS dataset_version,
    evidence,
    evidence.evidence_id + ':' +
        coalesce(evidence.chunk_id, '') + ':' +
        coalesce(toString(evidence.effective_from), '') + ':' +
        coalesce(toString(evidence.effective_to), '') AS identity_row
ORDER BY identity_row
WITH
    relation_name,
    dataset_version,
    collect(identity_row) AS identity_rows,
    sum(CASE WHEN evidence.is_quarantined = true THEN 1 ELSE 0 END)
        AS quarantined_count,
    sum(
        CASE
            WHEN coalesce(
                evidence.filed_at,
                evidence.published_at,
                evidence.effective_from
            ) IS NULL
            THEN 1
            ELSE 0
        END
    ) AS unknown_time_count
RETURN
    relation_name,
    dataset_version,
    size(identity_rows) AS row_count,
    quarantined_count,
    unknown_time_count,
    'SHA-256' AS reconciliation_algorithm,
    reduce(
        digest_input = '',
        identity IN identity_rows |
        digest_input + CASE WHEN digest_input = '' THEN '' ELSE '|' END + identity
    ) AS ordered_identity_digest_input;
