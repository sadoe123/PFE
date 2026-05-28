"""
OnePilot – LLM Description Generator
Génère automatiquement des descriptions métier pour les tables/colonnes
de n'importe quelle source ERP via Ollama, puis les sauvegarde dans le
catalogue et réindexe dans MeiliSearch.

Objectif : rendre le RAG générique — le LLM comprend "solde bancaire"
même si la table s'appelle CS_BAL_ACC_BNK.

Usage :
    POST /sources/{source_id}/generate-descriptions
    → lance l'enrichissement en arrière-plan
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
import httpx

logger = logging.getLogger(__name__)

# ── Config Ollama ────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")

# Timeout adaptatif selon le modèle :
# - qwen2.5-coder:3b sur CPU  → 60s
# - mistral:7b sur CPU        → 120s
# - llama3:8b sur GPU         → 15s
# En prod avec GPU, réduire à 15-20s via env var OLLAMA_TIMEOUT
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

# Taille du batch — tables traitées en parallèle
# CPU: 3 (évite saturation mémoire)
# GPU: 8-10 (parallélisme maximal)
# Configurable via env var DESC_BATCH_SIZE
BATCH_SIZE = int(os.environ.get("DESC_BATCH_SIZE", "3"))

# Nombre de colonnes max envoyées au LLM par table
# Limité par le context window du modèle :
# - qwen2.5-coder:3b : 2048 tokens → max ~15 colonnes
# - mistral:7b       : 8192 tokens → max ~50 colonnes
# - llama3:8b        : 8192 tokens → max ~50 colonnes
# On envoie les N premières colonnes (souvent les plus importantes)
# Configurable via env var DESC_MAX_COLS
MAX_COLS_PER_TABLE = int(os.environ.get("DESC_MAX_COLS", "12"))


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_description_prompt(
    table_name: str,
    columns: List[str],
    source_name: str,
    source_type: str,
    existing_relations: List[str] = None,
) -> str:
    """
    Construit le prompt pour générer une description métier d'une table.
    Le LLM doit retourner un JSON structuré.
    """
    cols_str = ", ".join(columns[:MAX_COLS_PER_TABLE])
    relations_str = ""
    if existing_relations:
        relations_str = f"\nKnown FK relations: {', '.join(existing_relations[:5])}"

    return f"""You are a business analyst expert in ERP systems.
Analyze this database table and generate a business description in French.

Source ERP: {source_name} ({source_type})
Table name: {table_name}
Columns: {cols_str}{relations_str}

Respond ONLY with a valid JSON object (no markdown, no explanation):
{{
  "table_description": "Description métier courte de la table en 1-2 phrases",
  "domain": "one of: finance, treasury, banking, hr, sales, procurement, accounting, logistics, customer, configuration, audit, other",
  "keywords": ["mot1", "mot2", "mot3"],
  "column_descriptions": {{
    "COLUMN_NAME": "description courte"
  }}
}}

Rules:
- table_description: 1 sentence in French describing the business data
- keywords: 3 French business terms users would search
- column_descriptions: max 3 columns only
- Table prefixes: CS=Cash/Settlement, DI=Data Integration, GS=General Settings, AA=Access/Auth, TH=Config, RC=Reconciliation, PY=Payment
- Return ONLY the JSON, nothing else
"""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALLER
# ═══════════════════════════════════════════════════════════════════════════════

async def _call_ollama(prompt: str, timeout: int = None) -> Optional[str]:
    """Appelle Ollama et retourne la réponse brute."""
    try:
        async with httpx.AsyncClient(timeout=timeout or OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":   OLLAMA_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.2, "num_predict": 600},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except httpx.ConnectError:
        logger.warning("[DescGen] Ollama non disponible")
        return None
    except Exception as e:
        logger.warning(f"[DescGen] Ollama erreur: {e}")
        return None


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """Parse la réponse JSON du LLM avec fallback robuste."""
    if not raw:
        return None

    # Nettoyer markdown si présent
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Extraire le premier bloc JSON
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None

    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        # Tentative de réparation : enlever les virgules trailing
        import re
        fixed = re.sub(r",\s*([}\]])", r"\1", raw[start:end])
        try:
            return json.loads(fixed)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

async def _load_tables_without_descriptions(
    pg_pool: asyncpg.Pool,
    source_id: UUID,
    limit: int = 1000,
) -> List[Dict]:
    """
    Charge les tables qui n'ont pas encore de description LLM.
    Retourne une liste de dicts: {id, name, description, field_names}
    """
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                se.id,
                se.name,
                se.description,
                se.entity_type,
                COALESCE(
                    array_agg(ef.name ORDER BY ef.position) FILTER (WHERE ef.name IS NOT NULL),
                    ARRAY[]::text[]
                ) AS field_names
            FROM source_entities se
            LEFT JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE se.source_id = $1
              AND se.is_visible = TRUE
              AND (se.description IS NULL OR se.description = '' OR se.description NOT LIKE '%[LLM]%')
            GROUP BY se.id, se.name, se.description, se.entity_type
            ORDER BY se.name
            LIMIT $2
        """, source_id, limit)

    return [dict(r) for r in rows]


async def _load_entity_relations(
    pg_pool: asyncpg.Pool,
    source_id: UUID,
    entity_name: str,
) -> List[str]:
    """Charge les relations FK connues pour une entité."""
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT source_field, target_entity, target_field
            FROM entity_relations
            WHERE source_id = $1
              AND source_entity = $2
            LIMIT 5
        """, source_id, entity_name)

    return [
        f"{entity_name}.{r['source_field']} → {r['target_entity']}.{r['target_field']}"
        for r in rows
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════════

async def _save_description(
    pg_pool: asyncpg.Pool,
    entity_id: UUID,
    source_id: UUID,
    entity_name: str,
    parsed: Dict,
) -> None:
    """Sauvegarde la description et les mots-clés dans source_entities."""
    table_desc = parsed.get("table_description", "")
    keywords   = parsed.get("keywords", [])
    domain     = parsed.get("domain", "")

    # Description complète avec mots-clés intégrés en texte
    kw_text = ", ".join(keywords[:5]) if keywords else ""
    full_desc = f"[LLM] {table_desc}"
    if kw_text:
        full_desc += f" | {kw_text}"

    # Metadata simple en JSON string
    meta_update = json.dumps({
        "llm_description": table_desc,
        "llm_keywords": keywords,
        "llm_domain": domain,
    }, ensure_ascii=False)

    async with pg_pool.acquire() as conn:
        # Étape 1 : mise à jour description (simple, pas de jsonb complexe)
        await conn.execute(
            "UPDATE source_entities SET description = $1 WHERE id = $2",
            full_desc, entity_id
        )

        # Étape 2 : mise à jour metadata via merge JSON simple
        existing = await conn.fetchval(
            "SELECT metadata FROM source_entities WHERE id = $1", entity_id
        )
        try:
            meta = json.loads(existing) if existing else {}
        except Exception:
            meta = {}
        meta.update(json.loads(meta_update))
        await conn.execute(
            "UPDATE source_entities SET metadata = $1 WHERE id = $2",
            json.dumps(meta, ensure_ascii=False), entity_id
        )

        # Étape 3 : descriptions colonnes si présentes
        col_descs = parsed.get("column_descriptions", {})
        for col_name, col_desc in col_descs.items():
            if col_desc:
                await conn.execute(
                    """UPDATE entity_fields ef SET display_name = $1
                       FROM source_entities se
                       WHERE ef.entity_id = se.id AND se.id = $2
                         AND ef.name = $3
                         AND (ef.display_name IS NULL OR ef.display_name = ef.name)""",
                    col_desc[:255], entity_id, col_name
                )


async def _reindex_in_meili(
    pg_pool: asyncpg.Pool,
    source_id: UUID,
    entity_id: UUID,
    entity_name: str,
    description: str,
    keywords: List[str],
    source_name: str,
    source_type: str,
    field_names: List[str],
) -> None:
    """Réindexe l'entité dans MeiliSearch avec la nouvelle description."""
    try:
        from .semantic_enricher import _get_meili_client, _ensure_meili_index, MEILI_INDEX, _normalize_name, _resolve_synonyms

        client = _get_meili_client()
        if not client:
            return

        _ensure_meili_index(client)

        # Construire le document enrichi
        tokens   = _normalize_name(entity_name)
        resolved = _resolve_synonyms(tokens)
        # Ajouter les keywords LLM aux synonymes → améliore le recall RAG
        all_synonyms = resolved + tokens + keywords

        doc = {
            "id":            str(entity_id),
            "source_id":     str(source_id),
            "source_name":   source_name,
            "source_type":   source_type,
            "name":          entity_name,
            "display_name":  entity_name.replace("_", " ").title(),
            "description":   description,
            "entity_type":   "table",
            "domain":        "",
            "concept":       "",
            "entity_class":  "",
            "tags":          " ".join(keywords),
            "field_names":   " ".join(field_names[:50]),
            "synonyms_text": " ".join(all_synonyms),
            "sample_values": "",
        }

        idx = client.index(MEILI_INDEX)
        idx.add_documents([doc])
        logger.debug(f"[DescGen] Réindexé: {entity_name}")

    except Exception as e:
        logger.warning(f"[DescGen] Réindexation MeiliSearch {entity_name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_descriptions_for_source(
    pg_pool: asyncpg.Pool,
    source_id: UUID,
    source_name: str,
    source_type: str,
    batch_size: int = BATCH_SIZE,
    limit: int = 500,
    progress_callback=None,
) -> Dict:
    """
    Génère des descriptions LLM pour toutes les tables sans description.
    Générique — fonctionne pour n'importe quelle source ERP.

    Args:
        pg_pool: pool PostgreSQL OnePilot
        source_id: UUID de la source
        source_name: nom affiché de la source
        source_type: type connector (mssql, postgresql, etc.)
        batch_size: tables traitées en parallèle
        limit: nombre max de tables à traiter
        progress_callback: coroutine optionnelle appelée après chaque batch

    Returns:
        dict avec stats: {total, enriched, failed, skipped, duration_s}
    """
    t0 = time.time()
    logger.info(f"[DescGen] Démarrage enrichissement: {source_name} ({source_id})")

    # 1. Charger les tables sans description
    tables = await _load_tables_without_descriptions(pg_pool, source_id, limit)
    total = len(tables)
    logger.info(f"[DescGen] {total} tables à enrichir")

    if total == 0:
        return {"total": 0, "enriched": 0, "failed": 0, "skipped": 0, "duration_s": 0}

    enriched = 0
    failed   = 0
    skipped  = 0

    # 2. Traitement par batch
    for batch_start in range(0, total, batch_size):
        batch = tables[batch_start:batch_start + batch_size]

        async def process_table(table: Dict) -> Tuple[str, bool]:
            entity_name = table["name"]
            field_names = list(table["field_names"] or [])

            # Charger les relations FK pour contexte
            relations = await _load_entity_relations(pg_pool, source_id, entity_name)

            # Construire et envoyer le prompt
            prompt = _build_description_prompt(
                table_name    = entity_name,
                columns       = field_names,
                source_name   = source_name,
                source_type   = source_type,
                existing_relations = relations,
            )

            raw = await _call_ollama(prompt)
            if not raw:
                return entity_name, False

            parsed = _parse_llm_response(raw)
            if not parsed or not parsed.get("table_description"):
                logger.warning(f"[DescGen] Réponse invalide pour {entity_name}: {raw[:100]}")
                return entity_name, False

            # Sauvegarder description
            await _save_description(
                pg_pool     = pg_pool,
                entity_id   = table["id"],
                source_id   = source_id,
                entity_name = entity_name,
                parsed      = parsed,
            )

            # Réindexer dans MeiliSearch
            keywords = parsed.get("keywords", [])
            full_desc = parsed.get("table_description", "")
            await _reindex_in_meili(
                pg_pool     = pg_pool,
                source_id   = source_id,
                entity_id   = table["id"],
                entity_name = entity_name,
                description = full_desc,
                keywords    = keywords,
                source_name = source_name,
                source_type = source_type,
                field_names = field_names,
            )

            logger.info(f"[DescGen] ✓ {entity_name}: {full_desc[:60]}")
            return entity_name, True

        # Exécution parallèle du batch
        results = await asyncio.gather(
            *[process_table(t) for t in batch],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[DescGen] Exception: {result}")
                failed += 1
            elif result[1]:
                enriched += 1
            else:
                failed += 1

        # Progress callback optionnel
        if progress_callback:
            await progress_callback({
                "processed": batch_start + len(batch),
                "total":     total,
                "enriched":  enriched,
                "failed":    failed,
            })

        # Petite pause entre batches pour ne pas saturer Ollama
        await asyncio.sleep(0.5)

    duration = round(time.time() - t0, 1)
    logger.info(
        f"[DescGen] Terminé: {enriched}/{total} enrichies, "
        f"{failed} échecs, {duration}s"
    )

    return {
        "total":      total,
        "enriched":   enriched,
        "failed":     failed,
        "skipped":    skipped,
        "duration_s": duration,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT FASTAPI (à intégrer dans main.py)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_description_generation_bg(
    source_id: UUID,
    source_name: str,
    source_type: str,
    pg_pool: asyncpg.Pool,
    limit: int = 500,
) -> None:
    """
    Tâche de fond : génère les descriptions sans bloquer l'API.
    Lancée via BackgroundTasks FastAPI.
    """
    try:
        stats = await generate_descriptions_for_source(
            pg_pool     = pg_pool,
            source_id   = source_id,
            source_name = source_name,
            source_type = source_type,
            limit       = limit,
        )
        logger.info(f"[DescGen BG] Terminé pour {source_name}: {stats}")
    except Exception as e:
        logger.error(f"[DescGen BG] Erreur pour {source_name}: {e}", exc_info=True)
