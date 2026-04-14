"""
OnePilot – CDC Auto-Reindexer §2.2.5
Réindexation MeiliSearch automatique après breaking change
+ Préservation des relations validées manuellement lors d'un rollback
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

MEILI_INDEX = os.environ.get("MEILI_INDEX", "onepilot_entities") if False else "onepilot_entities"


class CDCReindexer:
    """
    Réindexe automatiquement MeiliSearch après un breaking change CDC.
    Préserve les relations validées manuellement lors d'un rollback.
    """

    def __init__(self, pg_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.pg    = pg_pool
        self.redis = redis_client

    # ── 1. Réindexation automatique après breaking change ────

    async def reindex_after_breaking_change(
        self, source_id: UUID, version: int, changes: list[dict]
    ) -> dict:
        """
        Déclenché automatiquement par CDCEngine après un breaking change.
        - Supprime les entités obsolètes de MeiliSearch
        - Réindexe les entités affectées
        - Préserve les relations validées
        """
        from .semantic_enricher import enrich_source, _get_meili_client, MEILI_INDEX

        results = {
            "source_id":      str(source_id),
            "version":        version,
            "deleted":        0,
            "reindexed":      0,
            "relations_preserved": 0,
            "errors":         [],
        }

        meili = _get_meili_client()
        if not meili:
            logger.warning("[CDCReindexer] MeiliSearch non disponible, skip réindexation")
            results["errors"].append("MeiliSearch non disponible")
            return results

        # ── Identifie les tables affectées par les breaking changes ──
        dropped_tables = [
            c["table"] for c in changes
            if c["type"] == "DROP_TABLE"
        ]
        altered_tables = [
            c["table"] for c in changes
            if c["type"] in (
                "ALTER_DROP_COLUMN", "ALTER_COLUMN_TYPE",
                "ALTER_COLUMN_NULLABLE", "CREATE_TABLE"
            )
        ]

        # ── Supprime les entités DROP_TABLE de MeiliSearch ──────────
        if dropped_tables:
            try:
                idx = meili.index(MEILI_INDEX)
                # Récupère les IDs MeiliSearch des entités supprimées
                entity_rows = await self.pg.fetch("""
                    SELECT id FROM source_entities
                    WHERE  source_id   = $1
                      AND  name        = ANY($2)
                """, source_id, dropped_tables)

                if entity_rows:
                    ids_to_delete = [str(r["id"]) for r in entity_rows]
                    idx.delete_documents(ids_to_delete)
                    results["deleted"] = len(ids_to_delete)
                    logger.info(
                        f"[CDCReindexer] {len(ids_to_delete)} entités supprimées "
                        f"de MeiliSearch (source {source_id})"
                    )
            except Exception as e:
                logger.error(f"[CDCReindexer] Erreur suppression MeiliSearch: {e}")
                results["errors"].append(f"Suppression MeiliSearch: {e}")

        # ── Réindexe les entités altérées ────────────────────────────
        if altered_tables:
            try:
                source_row = await self.pg.fetchrow(
                    "SELECT name, connector_type FROM data_sources WHERE id=$1",
                    source_id
                )
                if source_row:
                    # Lance un enrichissement partiel sur les tables affectées
                    reindex_result = await self._reindex_entities(
                        source_id,
                        source_row["name"],
                        source_row["connector_type"],
                        altered_tables,
                        meili,
                    )
                    results["reindexed"] = reindex_result
            except Exception as e:
                logger.error(f"[CDCReindexer] Erreur réindexation: {e}")
                results["errors"].append(f"Réindexation: {e}")

        # ── Préserve les relations validées manuellement ─────────────
        preserved = await self._preserve_validated_relations(
            source_id, dropped_tables + altered_tables
        )
        results["relations_preserved"] = preserved

        # ── Enregistre l'événement de réindexation ───────────────────
        await self._log_reindex_event(source_id, version, results)

        logger.info(
            f"[CDCReindexer] source {source_id} v{version} — "
            f"deleted={results['deleted']}, reindexed={results['reindexed']}, "
            f"relations_preserved={results['relations_preserved']}"
        )

        return results

    # ── 2. Réindexation partielle ────────────────────────────

    async def _reindex_entities(
        self,
        source_id: UUID,
        source_name: str,
        source_type: str,
        table_names: list[str],
        meili_client,
    ) -> int:
        """Réindexe uniquement les entités dont le nom est dans table_names."""
        from .semantic_enricher import (
            _get_meili_client, MEILI_INDEX,
            classify_entity, detect_dimensions,
            build_semantic_tags, index_entity_meili,
            _normalize_name,
        )

        count = 0
        rows = await self.pg.fetch("""
            SELECT
                se.id, se.name, se.entity_type, se.description, se.row_count,
                array_agg(ef.name ORDER BY ef.position) AS field_names
            FROM   source_entities se
            LEFT JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE  se.source_id  = $1
              AND  se.name       = ANY($2)
              AND  se.is_visible = TRUE
            GROUP  BY se.id, se.name, se.entity_type, se.description, se.row_count
        """, source_id, table_names)

        for r in rows:
            try:
                field_names = [f for f in (r["field_names"] or []) if f]
                classification = classify_entity(r["name"], field_names)
                tags = build_semantic_tags(
                    r["name"], field_names,
                    classification["domain"],
                    classification["concept"],
                )

                index_entity_meili(
                    meili_client,
                    str(r["id"]),
                    str(source_id),
                    source_name,
                    source_type,
                    r["name"],
                    r["entity_type"],
                    r["description"],
                    classification["domain"],
                    classification["concept"],
                    classification["entity_class"],
                    tags,
                    field_names,
                )
                count += 1
            except Exception as e:
                logger.warning(
                    f"[CDCReindexer] Réindexation {r['name']}: {e}"
                )

        return count

    # ── 3. Préservation des relations validées ───────────────

    async def _preserve_validated_relations(
        self, source_id: UUID, affected_tables: list[str]
    ) -> int:
        """
        Lors d'un ALTER ou DROP, préserve les relations validées manuellement
        en les marquant comme "needs_review" plutôt que de les supprimer.
        Évite de perdre le travail de validation expert.
        """
        if not affected_tables:
            return 0

        # Marque les relations validées des tables affectées comme "needs_review"
        result = await self.pg.execute("""
            UPDATE entity_relations
            SET    needs_review   = TRUE,
                   review_reason  = 'CDC breaking change detected',
                   reviewed_at    = NULL
            WHERE  source_id     = $1
              AND  is_confirmed   = TRUE
              AND  (
                  source_entity = ANY($2)
                  OR target_entity = ANY($2)
              )
        """, source_id, affected_tables)

        count = int(result.split()[-1]) if result else 0

        if count > 0:
            # Publie notification pour que les experts revalidient
            payload = json.dumps({
                "event":           "RELATIONS_NEED_REVIEW",
                "source_id":       str(source_id),
                "affected_tables": affected_tables,
                "relations_count": count,
                "timestamp":       datetime.now(timezone.utc).isoformat(),
                "message":         (
                    f"{count} relation(s) validée(s) nécessitent une "
                    f"révision suite à un breaking change CDC"
                ),
            })
            try:
                await self.redis.publish(f"cdc:relations:{source_id}", payload)
                key = f"cdc:relations:notifications:{source_id}"
                await self.redis.lpush(key, payload)
                await self.redis.ltrim(key, 0, 49)
                await self.redis.expire(key, 60 * 60 * 24 * 30)
            except Exception:
                pass

            logger.warning(
                f"[CDCReindexer] {count} relations validées marquées "
                f"'needs_review' pour source {source_id}"
            )

        return count

    # ── 4. Log événement réindexation ────────────────────────

    async def _log_reindex_event(
        self, source_id: UUID, version: int, results: dict
    ) -> None:
        """Persiste l'événement de réindexation dans l'historique."""
        try:
            await self.pg.execute("""
                INSERT INTO cdc_reindex_log
                    (source_id, schema_version, deleted_count, reindexed_count,
                     relations_preserved, errors, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
            """,
                source_id,
                version,
                results["deleted"],
                results["reindexed"],
                results["relations_preserved"],
                json.dumps(results["errors"]),
                datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning(f"[CDCReindexer] Log error: {e}")

    # ── 5. Réindexation complète (force) ─────────────────────

    async def full_reindex(self, source_id: UUID) -> dict:
        """
        Réindexation complète forcée d'une source dans MeiliSearch.
        Utile après un rollback ou une resync complète.
        """
        source_row = await self.pg.fetchrow(
            "SELECT name, connector_type FROM data_sources WHERE id=$1",
            source_id
        )
        if not source_row:
            raise ValueError(f"Source {source_id} introuvable")

        from .semantic_enricher import enrich_source
        result = await enrich_source(
            source_id,
            source_row["name"],
            source_row["connector_type"],
        )

        return {
            "status":         "reindexed",
            "source_id":      str(source_id),
            "enriched":       result.get("enriched", 0),
            "meili_indexed":  result.get("meili_indexed", 0),
        }