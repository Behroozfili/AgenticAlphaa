-- Stage 02 revision 2 / evidence-scoped V2 chunk identity amendment.
-- This additive, versioned migration changes constraints only. It does not
-- mutate rows, switch routes, broaden grants, or apply itself automatically.

begin;

do $forward$
declare
    current_definition text;
begin
    if to_regclass('public.alpha_documents_v2') is null then
        raise exception '001_alpha_documents_v2.sql must be applied before 004';
    end if;

    select lower(regexp_replace(pg_get_constraintdef(c.oid), '\s+', ' ', 'g'))
      into current_definition
      from pg_constraint as c
     where c.conrelid = 'public.alpha_documents_v2'::regclass
       and c.conname = 'alpha_documents_v2_document_identity';

    if current_definition is not null and current_definition <>
       'unique (canonical_source_id, content_hash, document_version, chunk_index, chunk_hash, embedding_version)' then
        raise exception 'alpha_documents_v2_document_identity has an unexpected definition';
    end if;

    if current_definition is not null then
        alter table public.alpha_documents_v2
            drop constraint if exists alpha_documents_v2_document_identity;
    end if;

    select lower(regexp_replace(pg_get_constraintdef(c.oid), '\s+', ' ', 'g'))
      into current_definition
      from pg_constraint as c
     where c.conrelid = 'public.alpha_documents_v2'::regclass
       and c.conname = 'alpha_documents_v2_evidence_identity';

    if current_definition is not null and current_definition <>
       'unique (evidence_id, dataset_version)' then
        raise exception 'alpha_documents_v2_evidence_identity has an unexpected definition';
    end if;

    if current_definition is not null then
        alter table public.alpha_documents_v2
            drop constraint if exists alpha_documents_v2_evidence_identity;
    end if;

    select lower(regexp_replace(pg_get_constraintdef(c.oid), '\s+', ' ', 'g'))
      into current_definition
      from pg_constraint as c
     where c.conrelid = 'public.alpha_documents_v2'::regclass
       and c.conname = 'alpha_documents_v2_document_evidence_chunk_identity';

    if current_definition is null then
        alter table public.alpha_documents_v2
            add constraint alpha_documents_v2_document_evidence_chunk_identity
            unique (
                canonical_source_id,
                content_hash,
                document_version,
                document_id,
                evidence_id,
                dataset_version,
                chunk_index,
                chunk_hash,
                embedding_version
            );
    elsif current_definition <>
          'unique (canonical_source_id, content_hash, document_version, document_id, evidence_id, dataset_version, chunk_index, chunk_hash, embedding_version)' then
        raise exception 'alpha_documents_v2_document_evidence_chunk_identity has an unexpected definition';
    end if;

    select lower(regexp_replace(pg_get_constraintdef(c.oid), '\s+', ' ', 'g'))
      into current_definition
      from pg_constraint as c
     where c.conrelid = 'public.alpha_documents_v2'::regclass
       and c.conname = 'alpha_documents_v2_evidence_chunk_identity';

    if current_definition is null then
        alter table public.alpha_documents_v2
            add constraint alpha_documents_v2_evidence_chunk_identity
            unique (
                evidence_id,
                dataset_version,
                chunk_index,
                chunk_hash,
                embedding_version
            );
    elsif current_definition <>
          'unique (evidence_id, dataset_version, chunk_index, chunk_hash, embedding_version)' then
        raise exception 'alpha_documents_v2_evidence_chunk_identity has an unexpected definition';
    end if;
end
$forward$;

comment on constraint alpha_documents_v2_document_evidence_chunk_identity
    on public.alpha_documents_v2 is
    'Evidence-scoped immutable document/chunk identity. Distinct dataset, retrieval, and effective observations coexist.';

comment on constraint alpha_documents_v2_evidence_chunk_identity
    on public.alpha_documents_v2 is
    'One immutable chunk position per evidence observation, dataset, chunk hash, and embedding version.';

commit;

-- ROLLBACK SQL (execute separately and only with explicit authorization):
-- The preflight checks deliberately fail closed if rows accepted by the new
-- coexistence contract would collide under the revision-1 constraints. No row
-- is deleted or rewritten; reconcile/preserve those observations before retry.
--
-- begin;
-- do $rollback$
-- declare
--     collision_count bigint;
-- begin
--     if to_regclass('public.alpha_documents_v2') is null then
--         raise exception 'alpha_documents_v2 is missing';
--     end if;
--
--     select count(*)
--       into collision_count
--       from (
--           select 1
--             from public.alpha_documents_v2
--            group by canonical_source_id, content_hash, document_version,
--                     chunk_index, chunk_hash, embedding_version
--           having count(*) > 1
--       ) as revision_1_document_conflicts;
--     if collision_count > 0 then
--         raise exception 'rollback blocked: revision-1 document identity would collide';
--     end if;
--
--     select count(*)
--       into collision_count
--       from (
--           select 1
--             from public.alpha_documents_v2
--            group by evidence_id, dataset_version
--           having count(*) > 1
--       ) as revision_1_evidence_conflicts;
--     if collision_count > 0 then
--         raise exception 'rollback blocked: revision-1 evidence identity would collide';
--     end if;
--
--     alter table public.alpha_documents_v2
--         drop constraint if exists alpha_documents_v2_document_evidence_chunk_identity;
--     alter table public.alpha_documents_v2
--         drop constraint if exists alpha_documents_v2_evidence_chunk_identity;
--
--     if not exists (
--         select 1 from pg_constraint
--          where conrelid = 'public.alpha_documents_v2'::regclass
--            and conname = 'alpha_documents_v2_document_identity'
--     ) then
--         alter table public.alpha_documents_v2
--             add constraint alpha_documents_v2_document_identity
--             unique (
--                 canonical_source_id,
--                 content_hash,
--                 document_version,
--                 chunk_index,
--                 chunk_hash,
--                 embedding_version
--             );
--     end if;
--
--     if not exists (
--         select 1 from pg_constraint
--          where conrelid = 'public.alpha_documents_v2'::regclass
--            and conname = 'alpha_documents_v2_evidence_identity'
--     ) then
--         alter table public.alpha_documents_v2
--             add constraint alpha_documents_v2_evidence_identity
--             unique (evidence_id, dataset_version);
--     end if;
-- end
-- $rollback$;
-- commit;
