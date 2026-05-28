"""
OnePilot — RAG Engine §Sprint 7A + §Sprint 7B (Graph RAG)
Schema-Aware RAG : Hybrid Search (BM25 + pgvector) + Reranking + Graph BFS

Fichier : api/rag_engine.py
Auteur  : Samar (PFE OnePilot)

Architecture Sprint 7A :
    Question → MeiliSearch (BM25) ──┐
                                     ├→ Fusion RRF → Reranker → Contexte → Prompt LLM
    Question → pgvector (dense)   ──┘

Architecture Sprint 7B (Graph RAG) :
    Tables seed (7A) → BFS sur entity_relations (FK)
                          ├── Hop 1 : FK directes
                          └── Hop 2 : FK des FK
                       → Contexte enrichi avec JOINs possibles → Prompt LLM
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import asyncpg
import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MEILI_HOST    = os.environ.get("MEILI_HOST", "http://onepilot_meili:7700")
MEILI_API_KEY = os.environ.get("MEILI_API_KEY", "onepilot_meili_key")
MEILI_INDEX   = os.environ.get("MEILI_INDEX", "onepilot_entities")

TOP_K_RETRIEVAL = 20
TOP_K_FINAL     = 3
RRF_K           = 60
EMBEDDING_DIM   = 384

# ── Graph RAG config ──────────────────────────────────────────────────────────
# GRAPH_MAX_HOPS : profondeur BFS maximale.
#   2 = Hop1 (FK directes) + Hop2 (FK des FK). Au-delà, le graphe explose.
GRAPH_MAX_HOPS = 2

# GRAPH_MAX_NEIGHBORS : max voisins explorés PAR TABLE PAR HOP.
#   Évite l'explosion combinatoire sur les super-hubs (ex: TH_REVDATEX = 451 FK).
#   5 = équilibre couverture/performance pour SXA (1223 tables, 3383 FK).
GRAPH_MAX_NEIGHBORS = 5

# GRAPH_MAX_TABLES : max tables totales injectées dans le prompt LLM.
#   Calcul contexte LLM : 12 tables × 15 cols × ~60 chars = ~10 800 chars ≈ 2700 tokens
#   Compatible avec qwen2.5-coder:3b (4096 tokens) et claude-3 (200k tokens).
#   Augmenter si modèle plus grand (ex: 20 pour claude-3-sonnet).
GRAPH_MAX_TABLES = 12

# GRAPH_MAX_JOIN_HINTS : max JOIN hints injectés dans le prompt.
#   Au-delà, le LLM se perd dans trop de choix possibles.
GRAPH_MAX_JOIN_HINTS = 15

# GRAPH_MIN_CONFIDENCE : seuil minimum de confiance pour inclure une FK.
#   0.5 = inclut FK prédites (ML) + FK confirmées (explicit_fk).
#   Mettre 0.9 pour n'inclure que les FK certaines (explicit_fk uniquement).
GRAPH_MIN_CONFIDENCE = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — RECHERCHE BM25 VIA MEILI SEARCH
# ─────────────────────────────────────────────────────────────────────────────

async def search_meili(
    question: str,
    source_id: UUID,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Recherche BM25 dans MeiliSearch.

    BM25 (Best Match 25) = algorithme de recherche par mots-clés.
    Il calcule un score de pertinence basé sur :
      - La fréquence du mot dans le document (TF)
      - L'inverse de la fréquence dans tous les documents (IDF)
      - La longueur du document

    Retourne une liste de dicts :
        [
          {"id": "uuid", "name": "Dernière intégration bancaire",
           "score": 0.92, "rank": 1, "columns": [...], "description": "..."},
          ...
        ]
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {}
            if MEILI_API_KEY:
                headers["Authorization"] = f"Bearer {MEILI_API_KEY}"

            response = await client.post(
                f"{MEILI_HOST}/indexes/{MEILI_INDEX}/search",
                headers=headers,
                json={
                    "q": question,
                    "filter": f'source_id = "{source_id}"',
                    "limit": top_k,
                    "showRankingScore": True,
                    "attributesToRetrieve": [
                        "id", "name", "description",
                        "field_names", "domain", "concept",
                        "entity_type", "row_count"
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for rank, hit in enumerate(data.get("hits", []), start=1):
            raw_cols = hit.get("field_names", [])
            if isinstance(raw_cols, str):
                cols = [c.strip() for c in raw_cols.split() if c.strip()]
            elif isinstance(raw_cols, list):
                cols = raw_cols
            else:
                cols = []

            domain = (
                hit.get("domain") or
                hit.get("business_domain") or
                ""
            )

            results.append({
                "id":          hit.get("id"),
                "name":        hit.get("name", ""),
                "description": hit.get("description", ""),
                "columns":     cols,
                "domain":      domain,
                "concept":     hit.get("concept") or hit.get("business_concept") or "",
                "entity_type": hit.get("entity_type", "table"),
                "row_count":   hit.get("row_count"),
                "bm25_score":  hit.get("_rankingScore", 0.0),
                "bm25_rank":   rank,
                "source":      "bm25",
            })

        # Déduplication par nom
        seen_names = set()
        deduped = []
        for r in results:
            if r["name"] not in seen_names:
                seen_names.add(r["name"])
                deduped.append(r)
        for i, r in enumerate(deduped, 1):
            r["bm25_rank"] = i
        results = deduped

        logger.info(
            f"[RAG] BM25 : {len(results)} résultats uniques pour '{question[:40]}'"
            f" (source={source_id})"
        )
        return results

    except httpx.ConnectError:
        logger.warning("[RAG] MeiliSearch non disponible")
        return []
    except Exception as e:
        logger.error(f"[RAG] Erreur BM25 : {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — RECHERCHE DENSE VIA PGVECTOR
# ─────────────────────────────────────────────────────────────────────────────

async def search_pgvector(
    question: str,
    source_id: UUID,
    pg_pool: asyncpg.Pool,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Recherche sémantique dense via pgvector.

    Contrairement à BM25 qui cherche des mots exacts, pgvector cherche
    par "sens". Il convertit la question en vecteur (embedding) et trouve
    les entités dont les vecteurs sont les plus proches dans l'espace sémantique.

    Opérateur pgvector :
        <=>  = distance cosinus (entre 0 et 2, 0 = identique)
    """
    try:
        embedding = await _compute_embedding(question)
        if embedding is None:
            logger.warning("[RAG] Embedding non disponible, skip pgvector")
            return []

        vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

        rows = await pg_pool.fetch(f"""
            SELECT
                se.id::text            AS id,
                se.name                AS name,
                se.description         AS description,
                se.entity_type         AS entity_type,
                se.row_count           AS row_count,
                se.business_domain              AS domain,
                se.business_concept             AS concept,
                (se.embedding <=> $1::vector) AS cosine_distance,
                COALESCE(
                    array_agg(ef.name ORDER BY ef.position) FILTER (WHERE ef.name IS NOT NULL),
                    ARRAY[]::text[]
                ) AS field_names
            FROM source_entities se
            LEFT JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE se.source_id = $2
              AND se.embedding IS NOT NULL
              AND se.is_visible = TRUE
            GROUP BY se.id, se.name, se.description, se.entity_type,
                     se.row_count, se.business_domain, se.business_concept, se.embedding
            ORDER BY se.embedding <=> $1::vector
            LIMIT $3
        """, vec_str, source_id, top_k)

        results = []
        for rank, row in enumerate(rows, start=1):
            similarity = 1.0 - float(row["cosine_distance"])

            results.append({
                "id":           str(row["id"]),
                "name":         row["name"],
                "description":  row["description"] or "",
                "columns":      list(row["field_names"]),
                "domain":       row["domain"] or "",
                "concept":      row["concept"] or "",
                "entity_type":  row["entity_type"],
                "row_count":    row["row_count"],
                "dense_score":  similarity,
                "dense_rank":   rank,
                "source":       "dense",
            })

        logger.info(
            f"[RAG] pgvector : {len(results)} résultats pour '{question[:40]}'"
        )
        return results

    except Exception as e:
        logger.error(f"[RAG] Erreur pgvector : {e}")
        return []


async def _compute_embedding(text: str) -> Optional[list[float]]:
    """
    Calcule l'embedding d'un texte via le service d'embedding local.
    Utilise le modèle all-MiniLM-L6-v2 (384 dimensions).

    Si le service n'est pas disponible, retourne None et le système
    continue avec BM25 uniquement (dégradation gracieuse).
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                "http://onepilot_embed:8001/embed",
                json={"text": text},
            )
            if resp.status_code == 200:
                return resp.json()["embedding"]
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        import asyncio

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def _sync_embed():
            return model.encode(text, normalize_embeddings=True).tolist()

        return await asyncio.to_thread(_sync_embed)

    except ImportError:
        logger.warning(
            "[RAG] sentence-transformers non installé. "
            "pip install sentence-transformers --break-system-packages"
        )
        return None
    except Exception as e:
        logger.error(f"[RAG] Erreur embedding : {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — FUSION RRF (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_rrf(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = RRF_K,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Fusionne les résultats BM25 et dense via Reciprocal Rank Fusion.

    Formule RRF :
        score(doc) = Σ 1 / (k + rang(doc, liste_i))

    Standard de la littérature (papier Cormack 2009).
    """
    scores: dict[str, dict] = {}

    for item in bm25_results:
        entity_id = item["id"]
        rank = item["bm25_rank"]
        rrf_contribution = 1.0 / (k + rank)

        if entity_id not in scores:
            scores[entity_id] = {"rrf_score": 0.0, **item}
        scores[entity_id]["rrf_score"] += rrf_contribution
        scores[entity_id]["bm25_rank"]  = rank
        scores[entity_id]["bm25_score"] = item.get("bm25_score", 0.0)

    for item in dense_results:
        entity_id = item["id"]
        rank = item["dense_rank"]
        rrf_contribution = 1.0 / (k + rank)

        if entity_id not in scores:
            scores[entity_id] = {"rrf_score": 0.0, **item}
        scores[entity_id]["rrf_score"] += rrf_contribution
        scores[entity_id]["dense_rank"]  = rank
        scores[entity_id]["dense_score"] = item.get("dense_score", 0.0)

    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    for i, item in enumerate(fused, start=1):
        item["rrf_rank"] = i
        has_bm25  = "bm25_rank"  in item
        has_dense = "dense_rank" in item
        if has_bm25 and has_dense:
            item["found_in"] = "both"
        elif has_bm25:
            item["found_in"] = "bm25"
        else:
            item["found_in"] = "dense"

    result = fused[:top_k]

    logger.info(
        f"[RAG] RRF : {len(bm25_results)} BM25 + {len(dense_results)} dense"
        f" → {len(result)} fusionnés"
        f" ({sum(1 for x in result if x.get('found_in')=='both')} dans les deux)"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — RERANKER
# ─────────────────────────────────────────────────────────────────────────────

async def rerank(
    question: str,
    candidates: list[dict],
    top_k: int = TOP_K_FINAL,
) -> list[dict]:
    """
    Reranke les candidats RRF avec un cross-encoder.

    Cross-encoder : encode [question + document] ENSEMBLE → plus précis que
    bi-encoder mais plus lent. Utilisé uniquement sur les top-20 RRF.

    Fallback heuristique si sentence-transformers absent.
    """
    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder
        import asyncio

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = []
        for c in candidates:
            entity_text = (
                f"Table: {c['name']}. "
                f"{c.get('description', '')}. "
                f"Colonnes: {', '.join(c.get('columns', [])[:10])}. "
                f"Domaine: {c.get('business_domain', '')}."
            )
            pairs.append((question, entity_text))

        def _sync_rerank():
            scores = model.predict(pairs)
            return scores.tolist()

        scores = await asyncio.to_thread(_sync_rerank)

        scored = []
        for candidate, score in zip(candidates, scores):
            scored.append({**candidate, "rerank_score": float(score)})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = scored[:top_k]

        logger.info(
            f"[RAG] Reranker : {len(candidates)} candidats → top {len(result)}"
            f" | top scores: {[round(x['rerank_score'],2) for x in result]}"
        )
        return result

    except ImportError:
        logger.warning("[RAG] CrossEncoder non disponible, fallback heuristique")
        return _rerank_heuristic(question, candidates, top_k)

    except Exception as e:
        logger.error(f"[RAG] Erreur reranker : {e}")
        return _rerank_heuristic(question, candidates, top_k)


def _rerank_heuristic(
    question: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Reranking heuristique — pénalité maximale pour tables infrastructure.
    Score = RRF_score (40%) + bonus_mot_clé + bonus_found_both
    """
    INFRA_PREFIXES = (
        "QRTZ_", "qrtz_", "sys", "SYS", "dt_", "DT_",
        "MSreplication", "msreplication", "sysdiagram", "__",
    )
    q_words = set(question.lower().split())
    scored = []

    for c in candidates:
        name = c.get("name", "")

        if any(name.startswith(p) for p in INFRA_PREFIXES):
            scored.append({**c, "rerank_score": -10.0})
            continue

        score = c.get("rrf_score", 0.0) * 0.4
        entity_text = (
            f"{name} {c.get('description','')} "
            f"{' '.join(c.get('columns',[]))}"
        ).lower()
        matching_words = sum(1 for w in q_words if w in entity_text and len(w) > 3)
        score += matching_words * 0.1

        if c.get("found_in") == "both":
            score += 0.2
        if c.get("entity_type") in ("view", "business_view"):
            score += 0.15

        scored.append({**c, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    business_only = [c for c in scored if c["rerank_score"] >= 0]
    return (business_only if business_only else scored)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# SPRINT 7B — GRAPH RAG : BFS SUR FK
# ─────────────────────────────────────────────────────────────────────────────

async def _load_fk_graph(
    source_id: UUID,
    pg_pool: asyncpg.Pool,
) -> Dict[str, List[Dict]]:
    """
    Charge toutes les relations FK de la source depuis entity_relations.

    Retourne un graphe d'adjacence BIDIRECTIONNEL — exactement comme le notebook
    (G.successors + G.predecessors) mais stocké en dict Python au lieu de NetworkX.

    Structure par table :
    {
        "TableA": [
            {
                "neighbor":          "TableB",
                "source_field":      "id_b",
                "target_field":      "id",
                "relation_type":     "many_to_one",
                "confidence":        1.0,      ← score [0.5 → 1.0]
                "weight":            1.0,      ← ALIAS de confidence (cohérence notebook)
                "detection_method":  "explicit_fk",
                "direction":         "out",    ← "out" = A→B, "in" = B←A
                "join_hint":         "TableA.id_b = TableB.id",
                "is_confirmed":      True,
            },
            ...
        ],
        "TableB": [
            { ..., "direction": "in", ... }   ← arc inverse
        ]
    }

    Bidirectionnel car un JOIN SQL peut s'écrire dans les deux sens :
        FROM A JOIN B ON A.fk = B.pk  ≡  FROM B JOIN A ON B.pk = A.fk
    """
    try:
        rows = await pg_pool.fetch("""
            SELECT
                source_entity,
                target_entity,
                source_field,
                target_field,
                relation_type,
                confidence,
                detection_method,
                is_confirmed
            FROM entity_relations
            WHERE source_id = $1
              AND confidence >= $2
            ORDER BY confidence DESC, is_confirmed DESC
        """, source_id, GRAPH_MIN_CONFIDENCE)

        graph: Dict[str, List[Dict]] = {}

        for row in rows:
            src       = row["source_entity"]
            tgt       = row["target_entity"]
            s_f       = row["source_field"] or ""
            t_f       = row["target_field"] or ""
            rtype     = row["relation_type"] or "fk"
            conf      = float(row["confidence"] or 0.5)
            method    = row["detection_method"] or "unknown"
            confirmed = bool(row["is_confirmed"])

            # JOIN hints dans les deux sens (SQL peut s'écrire dans les deux)
            hint_out = f"{src}.{s_f} = {tgt}.{t_f}" if s_f and t_f else ""
            hint_in  = f"{tgt}.{t_f} = {src}.{s_f}" if s_f and t_f else ""

            # ── Arc sortant : src → tgt ──────────────────────────────────────
            if src not in graph:
                graph[src] = []
            graph[src].append({
                "neighbor":          tgt,
                "source_field":      s_f,
                "target_field":      t_f,
                "relation_type":     rtype,
                "confidence":        conf,
                "weight":            conf,   # alias — cohérence avec notebook NetworkX
                "detection_method":  method,
                "direction":         "out",
                "join_hint":         hint_out,
                "is_confirmed":      confirmed,
            })

            # ── Arc entrant : tgt → src (bidirectionnel) ─────────────────────
            if tgt not in graph:
                graph[tgt] = []
            graph[tgt].append({
                "neighbor":          src,
                "source_field":      t_f,
                "target_field":      s_f,
                "relation_type":     rtype,
                "confidence":        conf,
                "weight":            conf,   # alias — cohérence avec notebook NetworkX
                "detection_method":  method,
                "direction":         "in",
                "join_hint":         hint_in,
                "is_confirmed":      confirmed,
            })

        logger.info(
            f"[GraphRAG] Graphe FK chargé : {len(graph)} tables"
            f" | {sum(len(v) for v in graph.values())//2} relations"
            f" | seuil confiance={GRAPH_MIN_CONFIDENCE}"
            f" (source={source_id})"
        )
        return graph

    except Exception as e:
        logger.error(f"[GraphRAG] Erreur chargement graphe FK : {e}")
        return {}


async def _bfs_fk_neighbors(
    seed_tables: List[str],
    source_id: UUID,
    pg_pool: asyncpg.Pool,
    max_hops: int = GRAPH_MAX_HOPS,
    max_neighbors: int = GRAPH_MAX_NEIGHBORS,
    max_total: int = GRAPH_MAX_TABLES,
) -> Tuple[List[str], List[Dict], Dict[str, List[Dict]]]:
    """
    BFS (Breadth-First Search) sur le graphe FK — identique au notebook.

    Algorithme aligné avec bfs_subgraph() du notebook :
        - Bidirectionnel : explore arcs "out" ET "in" (même résultat que
          G.successors() + G.predecessors() dans NetworkX)
        - Tri par (is_confirmed DESC, confidence DESC) — FK certaines en priorité
        - Déduplique les voisins avant de trier (évite double-comptage)
        - Enregistre TOUS les join_hints (même vers tables déjà visitées)

    Différence intentionnelle vs notebook :
        - max_total=12 (LLM a une fenêtre de contexte limitée)
        - Le notebook n'a pas de limite → visualisation complète

    Retourne :
        - all_tables  : seed + tables découvertes (ordonnées par distance BFS)
        - join_paths  : tous les chemins JOIN utiles (max GRAPH_MAX_JOIN_HINTS)
        - adjacency   : sous-graphe filtré sur les tables retournées
    """
    if not seed_tables:
        return [], [], {}

    graph = await _load_fk_graph(source_id, pg_pool)
    if not graph:
        logger.warning("[GraphRAG] Graphe FK vide — pas de relations disponibles")
        return seed_tables, [], {}

    # BFS — même logique que le notebook
    visited:    Set[str]    = set(seed_tables)
    queue                   = deque((t, 0) for t in seed_tables)
    join_paths: List[Dict]  = []
    all_tables: List[str]   = list(seed_tables)

    while queue and len(all_tables) < max_total:
        current_table, hop = queue.popleft()

        if hop >= max_hops:
            continue

        neighbors = graph.get(current_table, [])

        # ── Déduplication + tri identique au notebook ────────────────────────
        # Le notebook fait : set(successors + predecessors) puis tri par max(weight)
        # Ici : le graphe est déjà bidirectionnel (arcs in+out), on déduplique
        # par nom de voisin en gardant l'arc de plus haute confiance
        seen_neighbors: Dict[str, Dict] = {}
        for edge in neighbors:
            n = edge["neighbor"]
            if n not in seen_neighbors or edge["confidence"] > seen_neighbors[n]["confidence"]:
                seen_neighbors[n] = edge

        # Tri : FK confirmées d'abord, puis par confiance décroissante
        # (identique au notebook qui trie par max(weight) décroissant)
        neighbors_sorted = sorted(
            seen_neighbors.values(),
            key=lambda e: (int(e["is_confirmed"]), e["confidence"]),
            reverse=True
        )[:max_neighbors]

        added_this_hop = 0
        for edge in neighbors_sorted:
            if added_this_hop >= max_neighbors:
                break
            if len(all_tables) >= max_total:
                break

            neighbor = edge["neighbor"]

            # Enregistrer JOIN hint — même si table déjà visitée
            # (le notebook fait pareil : enregistre hint même pour voisins déjà vus)
            if edge.get("join_hint") and len(join_paths) < GRAPH_MAX_JOIN_HINTS:
                join_paths.append({
                    "from":         current_table,
                    "to":           neighbor,
                    "join_hint":    edge["join_hint"],
                    "hop":          hop + 1,
                    "confidence":   edge["confidence"],
                    "weight":       edge["weight"],       # alias — cohérence notebook
                    "method":       edge["detection_method"],
                    "direction":    edge["direction"],
                    "is_confirmed": edge["is_confirmed"],
                })

            # Ajouter au BFS uniquement si pas encore visité
            if neighbor not in visited:
                visited.add(neighbor)
                all_tables.append(neighbor)
                queue.append((neighbor, hop + 1))
                added_this_hop += 1

    # Sous-graphe filtré sur les tables retournées
    adjacency: Dict[str, List[Dict]] = {}
    for table in all_tables:
        if table in graph:
            adjacency[table] = [
                e for e in graph[table]
                if e["neighbor"] in visited
            ]

    confirmed_paths = sum(1 for p in join_paths if p["is_confirmed"])
    logger.info(
        f"[GraphRAG] BFS terminé : {len(seed_tables)} seed"
        f" → {len(all_tables)} tables totales"
        f" | {len(join_paths)} JOIN paths"
        f" ({confirmed_paths} confirmés, {len(join_paths)-confirmed_paths} ML)"
        f" | hops={max_hops}, neighbors={max_neighbors}, max_tables={max_total}"
    )
    return all_tables, join_paths, adjacency


async def _fetch_graph_entities(
    table_names: List[str],
    source_id: UUID,
    pg_pool: asyncpg.Pool,
) -> List[Dict]:
    """
    Récupère les métadonnées complètes des tables découvertes par BFS
    qui ne sont pas déjà dans les seed entities.

    Ces tables sont nouvelles (trouvées via FK) donc pas encore enrichies.
    """
    if not table_names:
        return []

    try:
        rows = await pg_pool.fetch("""
            SELECT
                se.id::text       AS id,
                se.name           AS name,
                se.description    AS description,
                se.entity_type    AS entity_type,
                se.row_count      AS row_count,
                se.business_domain AS domain,
                se.business_concept AS concept,
                COALESCE(
                    array_agg(ef.name ORDER BY ef.position)
                    FILTER (WHERE ef.name IS NOT NULL),
                    ARRAY[]::text[]
                ) AS field_names
            FROM source_entities se
            LEFT JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE se.source_id = $1
              AND se.name = ANY($2::text[])
              AND se.is_visible = TRUE
            GROUP BY se.id, se.name, se.description, se.entity_type,
                     se.row_count, se.business_domain, se.business_concept
        """, source_id, table_names)

        results = []
        for row in rows:
            results.append({
                "id":          str(row["id"]),
                "name":        row["name"],
                "description": row["description"] or "",
                "columns":     list(row["field_names"]),
                "domain":      row["domain"] or "",
                "concept":     row["concept"] or "",
                "entity_type": row["entity_type"],
                "row_count":   row["row_count"],
                "source":      "graph",    # ← marqué comme découvert par Graph RAG
                "rrf_score":   0.0,
                "rerank_score": 0.0,
            })

        logger.info(
            f"[GraphRAG] {len(results)} entités FK récupérées depuis PostgreSQL"
        )
        return results

    except Exception as e:
        logger.error(f"[GraphRAG] Erreur fetch graph entities : {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — CONSTRUCTION DU CONTEXTE POUR LE PROMPT LLM
# ─────────────────────────────────────────────────────────────────────────────

async def get_schema_context(
    question: str,
    source_id: UUID,
    pg_pool: asyncpg.Pool,
    top_k_final: int = TOP_K_FINAL,
) -> dict:
    """
    Fonction principale Sprint 7A (inchangée).
    Orchestre BM25 + pgvector + RRF + Reranker.
    """
    # ── Étape 1 : BM25 ───────────────────────────────────────────────────────
    bm25_results = await search_meili(question, source_id)

    # ── Étape 2 : Dense / pgvector ────────────────────────────────────────────
    dense_results = await search_pgvector(question, source_id, pg_pool)

    # ── Étape 3 : Fusion RRF ─────────────────────────────────────────────────
    if bm25_results or dense_results:
        fused = fuse_rrf(bm25_results, dense_results)
    else:
        logger.warning("[RAG] Aucun résultat BM25 ni dense — contexte vide")
        return _empty_context()

    # ── Étape 4 : Reranking ───────────────────────────────────────────────────
    top_entities = await rerank(question, fused, top_k=top_k_final)

    if not top_entities:
        return _empty_context()

    # ── Étape 5 : Enrichissement colonnes complètes ───────────────────────────
    enriched = await _enrich_with_full_columns(top_entities, pg_pool)

    # ── Étape 6 : Formatage ───────────────────────────────────────────────────
    context_text = _build_context_text(enriched)

    method = "hybrid" if dense_results else "bm25_only"

    logger.info(
        f"[RAG] Contexte final : {len(enriched)} tables"
        f" | méthode={method}"
        f" | tables={[e['name'] for e in enriched]}"
    )

    return {
        "context_text": context_text,
        "tables_found": [e["name"] for e in enriched],
        "table_count":  len(enriched),
        "method":       method,
        "entities":     enriched,
        "debug": {
            "bm25_count":   len(bm25_results),
            "dense_count":  len(dense_results),
            "fused_count":  len(fused),
            "top_scores":   [
                {
                    "name":         e["name"],
                    "rrf_score":    round(e.get("rrf_score", 0), 4),
                    "rerank_score": round(e.get("rerank_score", 0), 4),
                    "found_in":     e.get("found_in", "?"),
                }
                for e in top_entities
            ],
        },
    }


async def get_schema_context_with_graph(
    question: str,
    source_id: UUID,
    pg_pool: asyncpg.Pool,
    top_k_final: int = TOP_K_FINAL,
    max_hops: int = GRAPH_MAX_HOPS,
) -> dict:
    """
    SPRINT 7B — Fonction principale Graph RAG.

    Étend get_schema_context() avec l'exploration BFS du graphe FK.

    Pipeline complet :
        1. BM25 + pgvector + RRF + Reranker  (Sprint 7A)
              → tables seed pertinentes
        2. BFS sur entity_relations           (Sprint 7B)
              → tables liées par FK (2 sauts)
        3. Enrichissement colonnes complètes
        4. Formatage contexte avec JOIN hints

    Retourne le même format que get_schema_context() + champs graph_*.
    """
    # ── Phase 7A : RAG standard ───────────────────────────────────────────────
    bm25_results  = await search_meili(question, source_id)
    dense_results = await search_pgvector(question, source_id, pg_pool)

    if bm25_results or dense_results:
        fused = fuse_rrf(bm25_results, dense_results)
    else:
        logger.warning("[GraphRAG] Aucun résultat BM25 ni dense — contexte vide")
        return _empty_context()

    top_entities = await rerank(question, fused, top_k=top_k_final)
    if not top_entities:
        return _empty_context()

    # Tables trouvées par 7A = seeds pour le BFS
    seed_names = [e["name"] for e in top_entities]

    # ── Phase 7B : BFS Graph RAG ──────────────────────────────────────────────
    all_table_names, join_paths, adjacency = await _bfs_fk_neighbors(
        seed_tables  = seed_names,
        source_id    = source_id,
        pg_pool      = pg_pool,
        max_hops     = max_hops,
        max_neighbors = GRAPH_MAX_NEIGHBORS,
        max_total    = GRAPH_MAX_TABLES,
    )

    # Tables nouvellement découvertes par BFS (pas dans seed)
    new_table_names = [t for t in all_table_names if t not in seed_names]

    # Récupérer métadonnées des nouvelles tables
    graph_entities = await _fetch_graph_entities(new_table_names, source_id, pg_pool)

    # ── Enrichissement colonnes complètes (seed + graph) ─────────────────────
    seed_enriched  = await _enrich_with_full_columns(top_entities, pg_pool)
    graph_enriched = await _enrich_with_full_columns(graph_entities, pg_pool)

    # Toutes les entités : seed en premier (priorité), puis graph
    all_entities = seed_enriched + graph_enriched

    # ── Formatage contexte avec graphe ───────────────────────────────────────
    context_text = _build_context_text_with_graph(
        seed_entities  = seed_enriched,
        graph_entities = graph_enriched,
        join_paths     = join_paths,
        adjacency      = adjacency,
    )

    method = "graph_hybrid" if dense_results else "graph_bm25_only"

    logger.info(
        f"[GraphRAG] Contexte final : {len(seed_enriched)} seed"
        f" + {len(graph_enriched)} graph"
        f" = {len(all_entities)} tables totales"
        f" | {len(join_paths)} JOIN paths"
        f" | méthode={method}"
    )

    return {
        "context_text":    context_text,
        "tables_found":    [e["name"] for e in all_entities],
        "table_count":     len(all_entities),
        "method":          method,
        "entities":        all_entities,
        # Champs spécifiques Graph RAG
        "graph_seed_tables":   seed_names,
        "graph_new_tables":    new_table_names,
        "graph_join_paths":    join_paths,
        "graph_hops":          max_hops,
        "debug": {
            "bm25_count":       len(bm25_results),
            "dense_count":      len(dense_results),
            "fused_count":      len(fused),
            "seed_count":       len(seed_enriched),
            "graph_count":      len(graph_enriched),
            "join_path_count":  len(join_paths),
            "top_scores": [
                {
                    "name":         e["name"],
                    "rrf_score":    round(e.get("rrf_score", 0), 4),
                    "rerank_score": round(e.get("rerank_score", 0), 4),
                    "found_in":     e.get("found_in", "?"),
                    "source":       e.get("source", "rag"),
                }
                for e in all_entities
            ],
        },
    }


async def _enrich_with_full_columns(
    entities: list[dict],
    pg_pool: asyncpg.Pool,
) -> list[dict]:
    """
    Récupère les colonnes complètes (avec types, FK, description) depuis PostgreSQL.
    MeiliSearch ne stocke que les noms — PostgreSQL a tous les détails.
    """
    enriched = []
    for entity in entities:
        try:
            entity_id = entity["id"]

            fields = await pg_pool.fetch("""
                SELECT
                    ef.name                AS col_name,
                    ef.data_type           AS data_type,
                    ef.native_type         AS native_type,
                    ef.is_primary_key      AS is_pk,
                    ef.is_foreign_key      AS is_fk,
                    ef.is_nullable         AS is_nullable,
                    ef.description         AS col_description,
                    ef.position            AS position
                FROM entity_fields ef
                WHERE ef.entity_id = $1
                ORDER BY ef.position
                LIMIT 20
            """, entity_id)

            enriched.append({
                **entity,
                "fields": [dict(f) for f in fields],
            })

        except Exception as e:
            logger.warning(f"[RAG] Enrichissement colonnes échoué pour {entity.get('name')}: {e}")
            enriched.append(entity)

    return enriched


def _build_context_text(entities: list[dict]) -> str:
    """
    Formate les entités en texte structuré pour le prompt LLM (Sprint 7A).
    """
    if not entities:
        return ""

    lines = [
        f"=== SCHÉMA SOURCE ({len(entities)} table(s) récupérées par RAG) ===\n"
    ]

    for entity in entities:
        lines.append(f"Table [{entity['name']}]")

        if entity.get("description"):
            lines.append(f"  Description: {entity['description']}")

        if entity.get("business_domain"):
            lines.append(f"  Domaine métier: {entity['business_domain']}")

        fields = entity.get("fields") or []
        if fields:
            lines.append("  Colonnes:")
            for field in fields[:15]:
                col_name  = field.get("col_name", "")
                data_type = field.get("data_type", "")
                is_pk     = field.get("is_pk", False)
                is_fk     = field.get("is_fk", False)
                col_desc  = field.get("col_description", "")

                indicators = []
                if is_pk:
                    indicators.append("PK")
                if is_fk:
                    indicators.append("FK")

                indicator_str = f"  [{', '.join(indicators)}]" if indicators else ""
                desc_str      = f"  ← {col_desc}" if col_desc else ""

                lines.append(
                    f"    - {col_name:<35} {data_type:<12}"
                    f"{indicator_str}{desc_str}"
                )
        elif entity.get("columns"):
            lines.append(
                f"  Colonnes: {', '.join(entity['columns'][:10])}"
            )

        lines.append("")

    return "\n".join(lines)


def _build_context_text_with_graph(
    seed_entities:  List[Dict],
    graph_entities: List[Dict],
    join_paths:     List[Dict],
    adjacency:      Dict[str, List[Dict]],
) -> str:
    """
    Formate le contexte enrichi avec Graph RAG pour le prompt LLM (Sprint 7B).

    Les JOIN hints sont triés par priorité :
        1. FK confirmées (explicit_fk, is_confirmed=True) en premier
        2. FK haute confiance (≥0.9) ensuite
        3. FK prédites ML (<0.9) en dernier

    Structure injectée dans le prompt :
        === SCHÉMA SOURCE (RAG + Graph RAG) ===

        [TABLES PRINCIPALES]
        Table [AA_AU2CMP]
          Colonnes: AU2CMP_ID (PK), ...

        [TABLES LIÉES via FK]
        Table [GS_CMP]  [FK]
          Colonnes: CMP_ID (PK), ...

        [CHEMINS JOIN — utiliser pour les requêtes multi-tables]
        → AA_AU2CMP.AU2CMP_ID = GS_CMP.CMP_ID  (Direct, 100%, explicit_fk)
        → AA_AU2EMP.REV = TH_REVDATEX.REVDATEX_ID  (Direct, 100%, explicit_fk)
    """
    if not seed_entities and not graph_entities:
        return ""

    total = len(seed_entities) + len(graph_entities)
    lines = [
        f"=== SCHÉMA SOURCE — RAG + Graph RAG"
        f" ({total} table(s) : {len(seed_entities)} seed + {len(graph_entities)} FK) ===\n"
    ]

    # ── Tables principales (seed) ─────────────────────────────────────────────
    if seed_entities:
        lines.append("[TABLES PRINCIPALES — pertinence directe avec la question]")
        for entity in seed_entities:
            _append_entity_block(lines, entity, adjacency)

    # ── Tables liées par FK (graph) ───────────────────────────────────────────
    if graph_entities:
        lines.append("[TABLES LIÉES — découvertes via relations FK (Graph RAG)]")
        for entity in graph_entities:
            _append_entity_block(lines, entity, adjacency, is_graph=True)

    # ── Chemins JOIN triés par priorité ──────────────────────────────────────
    if join_paths:
        lines.append("[CHEMINS JOIN — utiliser ces conditions exactes pour les JOINs]")

        # Déduplication + tri : confirmed first, puis confidence décroissante
        seen_hints: Set[str] = set()
        unique_paths = []
        for path in join_paths:
            hint = path.get("join_hint", "")
            if hint and hint not in seen_hints:
                seen_hints.add(hint)
                unique_paths.append(path)

        # Tri final : FK confirmées > haute confiance > ML prédites
        unique_paths.sort(
            key=lambda p: (int(p.get("is_confirmed", False)), p.get("confidence", 0)),
            reverse=True
        )

        for path in unique_paths[:GRAPH_MAX_JOIN_HINTS]:
            hint      = path.get("join_hint", "")
            conf      = path.get("confidence", 0)
            hop       = path.get("hop", 1)
            method    = path.get("method", "")
            confirmed = path.get("is_confirmed", False)

            hop_str    = "Direct" if hop == 1 else f"Hop {hop}"
            conf_str   = f"{conf:.0%}"
            method_str = f", {method}" if method not in ("unknown", "") else ""
            cert_str   = " ✓" if confirmed else ""

            lines.append(
                f"  → {hint}"
                f"  ({hop_str}, {conf_str}{method_str}{cert_str})"
            )

        lines.append("")
        lines.append(
            "  RÈGLE SQL : Utilise UNIQUEMENT les conditions JOIN ci-dessus."
            " Ne devine pas les colonnes de jointure."
        )
        lines.append("")

    return "\n".join(lines)


def _append_entity_block(
    lines: List[str],
    entity: Dict,
    adjacency: Dict[str, List[Dict]],
    is_graph: bool = False,
) -> None:
    """Ajoute le bloc texte d'une entité dans le contexte."""
    name   = entity.get("name", "")
    prefix = "  [FK]" if is_graph else ""

    lines.append(f"Table [{name}]{prefix}")

    if entity.get("description"):
        lines.append(f"  Description: {entity['description']}")

    if entity.get("domain"):
        lines.append(f"  Domaine: {entity['domain']}")

    # Voisins FK de cette table (pour contexte JOIN)
    neighbors = adjacency.get(name, [])
    if neighbors:
        neighbor_names = list({n["neighbor"] for n in neighbors})[:4]
        lines.append(f"  Liée à: {', '.join(neighbor_names)}")

    fields = entity.get("fields") or []
    if fields:
        lines.append("  Colonnes:")
        for field in fields[:15]:
            col_name  = field.get("col_name", "")
            data_type = field.get("data_type", "")
            is_pk     = field.get("is_pk", False)
            is_fk_col = field.get("is_fk", False)
            col_desc  = field.get("col_description", "")

            indicators = []
            if is_pk:
                indicators.append("PK")
            if is_fk_col:
                indicators.append("FK")

            indicator_str = f"  [{', '.join(indicators)}]" if indicators else ""
            desc_str      = f"  ← {col_desc}" if col_desc else ""

            lines.append(
                f"    - {col_name:<35} {data_type:<12}"
                f"{indicator_str}{desc_str}"
            )
    elif entity.get("columns"):
        lines.append(
            f"  Colonnes: {', '.join(entity['columns'][:10])}"
        )

    lines.append("")


def _empty_context() -> dict:
    """Retourne un contexte vide avec la structure attendue."""
    return {
        "context_text":      "",
        "tables_found":      [],
        "table_count":       0,
        "method":            "empty",
        "entities":          [],
        "graph_seed_tables": [],
        "graph_new_tables":  [],
        "graph_join_paths":  [],
        "graph_hops":        0,
        "debug": {
            "bm25_count":      0,
            "dense_count":     0,
            "fused_count":     0,
            "seed_count":      0,
            "graph_count":     0,
            "join_path_count": 0,
            "top_scores":      [],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTÉGRATION DANS LE PROMPT LLM
# ─────────────────────────────────────────────────────────────────────────────

def inject_rag_context_into_prompt(
    original_prompt: str,
    rag_context: dict,
) -> str:
    """
    Injecte le contexte RAG (7A ou 7B) dans le prompt LLM.

    Le contexte Graph RAG inclut les JOIN hints qui permettent au LLM
    de générer des requêtes multi-tables correctes automatiquement.
    """
    if not rag_context.get("context_text"):
        return original_prompt

    method = rag_context.get("method", "hybrid")
    if "graph" in method:
        method_label = "RAG + Graph RAG (FK)"
    else:
        method_label = "RAG"

    rag_block = (
        f"\nContexte schéma récupéré automatiquement par {method_label} :\n"
        + rag_context["context_text"]
        + "\nIMPORTANT: Utilise en priorité les tables du contexte ci-dessus."
        + " Si des chemins JOIN sont indiqués, utilise-les pour les requêtes multi-tables.\n"
    )

    lines = original_prompt.split("\n")
    insert_pos = min(3, len(lines))
    lines.insert(insert_pos, rag_block)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SPRINT 7C — CORRECTIVE RAG
# Sub-querying + Routing + SQL Column Validator
# ═══════════════════════════════════════════════════════════════════════════════

# ── Routing : questions conceptuelles qui ne nécessitent PAS de SQL ──────────

# Patterns indiquant une question conceptuelle/explicative → réponse texte
_CONCEPTUAL_PATTERNS = [
    r"^quels?\s+sont\s+les?\s+(erp|sap|logiciel|syst[eè]me|module|applicat|fonctionnalit)",
    r"^qu['‘’e]est[- ]ce\\s+(qu['‘’e]|que)",
    r"^c['‘’e]est\\s+quoi",
    r"^explique[sz]?\s+",
    r"^comment\s+(fonctionne|marche|utilise|configur)",
    r"^d[eé]finis?\s+",
    r"^que\s+signifie",
    r"^quelle\s+est\s+la\s+diff[eé]rence",
    r"^quels?\s+sont\s+les?\s+(avantage|concept|principe|caract)",
    r"(erp|sap|dynamics|odoo|sage|progiciel)\s+(est|sont|signifie|fonctionne)",
    r"^d[eé]cris?\s+(moi\s+)?le[s]?\s+",
    r"^pr[eé]sente[z]?\s+",
    r"^donne[sz]?\s+moi\s+une?\s+(explication|description|d[eé]finition|pr[eé]sentation)",
]

import re as _re

_CONCEPTUAL_RE = [_re.compile(p, _re.IGNORECASE) for p in _CONCEPTUAL_PATTERNS]


def is_conceptual_question(question: str) -> bool:
    """
    Détecte si la question est conceptuelle/explicative et ne nécessite PAS de SQL.

    Sprint 7C — Routing :
        "quels sont les ERP ?" → True → répondre en texte
        "jointure entre Orders et Customers" → False → générer SQL
    """
    q = question.strip()
    for pattern in _CONCEPTUAL_RE:
        if pattern.search(q):
            return True
    return False


# ── Sub-querying : décomposer une question complexe multi-entités ─────────────

def decompose_question(question: str) -> List[str]:
    """
    Sprint 7C — Sub-querying :
    Décompose une question complexe multi-entités en sous-questions simples.

    Exemple :
        "liste les employés avec leur entreprise et leur banque"
        → ["employés personnel", "entreprise société", "banque établissement bancaire"]

    Pourquoi : le RAG cherche les tables par similarité sémantique.
    Avec une question longue, le vecteur "moyen" est dilué.
    En décomposant, chaque sous-query cible précisément une entité.
    """
    q = question.lower().strip()

    sub_queries = []

    # Patterns de décomposition : "X avec leur Y et leur Z"
    # Capture les entités principales de la question
    entity_patterns = [
        # "employés avec leur entreprise et leur banque"
        (r"employ[eé][s]?\s+avec\s+", ["employés personnel utilisateurs", "entreprise société compagnie", "banque établissement bancaire"]),
        (r"employ[eé][s]?\s+et\s+", ["employés personnel", "entreprise société"]),
        # "clients avec leurs commandes"
        (r"client[s]?\s+avec\s+", ["clients customers", "commandes orders"]),
        (r"client[s]?\s+et\s+", ["clients customers", "commandes orders"]),
        # "ventes avec les produits"
        (r"vente[s]?\s+avec\s+", ["ventes sales", "produits products"]),
        # jointure complexe
        (r"jointure[s]?\s+complexe[s]?", ["relations FK jointures tables", "clés étrangères foreign keys"]),
        (r"relation[s]?\s+entre\s+", ["relations FK entity_relations", "tables liées"]),
    ]

    for pattern, entities in entity_patterns:
        if _re.search(pattern, q):
            return entities

    # Décomposition générique par "avec", "et leur", "ainsi que"
    separators = [r"\s+avec\s+(leurs?\s+)?", r"\s+et\s+(leurs?\s+)?", r"\s+ainsi\s+que\s+", r"\s+,\s+"]
    parts = [q]
    for sep in separators:
        new_parts = []
        for part in parts:
            split = _re.split(sep, part)
            new_parts.extend([p.strip() for p in split if p and p.strip()])
        parts = new_parts

    if len(parts) > 1:
        return parts[:4]  # max 4 sous-queries

    return [question]  # question simple → pas de décomposition


async def get_schema_context_corrective(
    question: str,
    source_id: UUID,
    pg_pool,
    max_hops: int = GRAPH_MAX_HOPS,
) -> dict:
    """
    Sprint 7C — Corrective RAG Pipeline complet :

    1. Routing : si question conceptuelle → retourne contexte vide (pas de SQL)
    2. Sub-querying : décompose la question → plusieurs RAG searches
    3. Fusion : combine tous les contextes → meilleur coverage
    4. Graph RAG : BFS sur les seeds fusionnés
    5. Retourne contexte enrichi

    Amélioration vs 7B :
        - Sub-querying garantit de meilleures seeds même pour questions complexes
        - Le routing évite de générer du SQL pour des questions conceptuelles
    """
    # ── Étape 1 : Routing ─────────────────────────────────────────────────────
    if is_conceptual_question(question):
        logger.info(f"[CRAG] Question conceptuelle détectée → pas de SQL : '{question[:60]}'")
        return {
            **_empty_context(),
            "method":           "conceptual",
            "is_conceptual":    True,
            "conceptual_question": question,
        }

    # ── Étape 2 : Sub-querying ────────────────────────────────────────────────
    sub_queries = decompose_question(question)
    logger.info(f"[CRAG] Sub-queries : {sub_queries}")

    # ── Étape 3 : RAG multi-query ─────────────────────────────────────────────
    all_bm25:  List[dict] = []
    all_dense: List[dict] = []
    seen_ids:  set = set()

    for sub_q in sub_queries:
        # BM25
        bm25 = await search_meili(sub_q, source_id)
        for item in bm25:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_bm25.append(item)

        # Dense (pgvector)
        dense = await search_pgvector(sub_q, source_id, pg_pool)
        for item in dense:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_dense.append(item)

    # Re-numéroter les rangs après fusion
    for i, item in enumerate(all_bm25, 1):
        item["bm25_rank"] = i
    for i, item in enumerate(all_dense, 1):
        item["dense_rank"] = i

    if not all_bm25 and not all_dense:
        logger.warning("[CRAG] Aucun résultat BM25 ni dense après sub-querying")
        return _empty_context()

    # ── Étape 4 : Fusion RRF + Reranking ─────────────────────────────────────
    fused = fuse_rrf(all_bm25, all_dense, top_k=TOP_K_RETRIEVAL)
    top_entities = await rerank(question, fused, top_k=TOP_K_FINAL)

    if not top_entities:
        return _empty_context()

    seed_names = [e["name"] for e in top_entities]

    # ── Étape 5 : Graph RAG BFS ───────────────────────────────────────────────
    all_table_names, join_paths, adjacency = await _bfs_fk_neighbors(
        seed_tables   = seed_names,
        source_id     = source_id,
        pg_pool       = pg_pool,
        max_hops      = max_hops,
        max_neighbors = GRAPH_MAX_NEIGHBORS,
        max_total     = GRAPH_MAX_TABLES,
    )

    new_table_names = [t for t in all_table_names if t not in seed_names]
    graph_entities  = await _fetch_graph_entities(new_table_names, source_id, pg_pool)

    # ── Étape 6 : Enrichissement ──────────────────────────────────────────────
    seed_enriched  = await _enrich_with_full_columns(top_entities, pg_pool)
    graph_enriched = await _enrich_with_full_columns(graph_entities, pg_pool)
    all_entities   = seed_enriched + graph_enriched

    context_text = _build_context_text_with_graph(
        seed_entities  = seed_enriched,
        graph_entities = graph_enriched,
        join_paths     = join_paths,
        adjacency      = adjacency,
    )

    method = "crag_graph_hybrid" if all_dense else "crag_graph_bm25"

    logger.info(
        f"[CRAG] Résultat : {len(sub_queries)} sub-queries"
        f" | {len(seed_enriched)} seed + {len(graph_enriched)} graph"
        f" | {len(join_paths)} JOIN paths"
        f" | méthode={method}"
    )

    return {
        "context_text":        context_text,
        "tables_found":        [e["name"] for e in all_entities],
        "table_count":         len(all_entities),
        "method":              method,
        "entities":            all_entities,
        "graph_seed_tables":   seed_names,
        "graph_new_tables":    new_table_names,
        "graph_join_paths":    join_paths,
        "graph_hops":          max_hops,
        "sub_queries":         sub_queries,
        "is_conceptual":       False,
        "debug": {
            "sub_query_count":  len(sub_queries),
            "bm25_count":       len(all_bm25),
            "dense_count":      len(all_dense),
            "fused_count":      len(fused),
            "seed_count":       len(seed_enriched),
            "graph_count":      len(graph_enriched),
            "join_path_count":  len(join_paths),
        },
    }


# ── SQL Column Validator : vérifier colonnes contre le vrai schéma ───────────

async def validate_sql_columns(
    sql: str,
    source_id: UUID,
    pg_pool,
) -> Dict[str, List[str]]:
    """
    Sprint 7C — SQL Column Validator.

    Vérifie que les colonnes utilisées dans le SQL existent dans le vrai schéma.
    Retourne un dict :
    {
        "invalid_columns": ["BANQUE", "CTPRLAST"],
        "suggestions":     {"BANQUE": "DESCRIPTION", "CTPRLAST": "CTPRSLAST"},
        "valid":           False,
    }

    Pourquoi : le LLM "invente" parfois des colonnes qui n'existent pas.
    Ce validator intercepte ces erreurs avant exécution.
    """
    import re

    # Extraire les tables utilisées dans le SQL
    table_pattern = re.compile(
        r'\bFROM\s+\[?(\w+)\]?|\bJOIN\s+\[?(\w+)\]?',
        re.IGNORECASE
    )
    tables_used = set()
    for m in table_pattern.finditer(sql):
        t = m.group(1) or m.group(2)
        if t:
            tables_used.add(t.strip('[]'))

    if not tables_used:
        return {"invalid_columns": [], "suggestions": {}, "valid": True}

    # Récupérer les colonnes réelles de ces tables
    try:
        rows = await pg_pool.fetch("""
            SELECT
                se.name   AS table_name,
                ef.name   AS col_name,
                ef.data_type
            FROM source_entities se
            JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE se.source_id = $1
              AND se.name = ANY($2::text[])
              AND se.is_visible = TRUE
        """, source_id, list(tables_used))
    except Exception as e:
        logger.warning(f"[CRAG] validate_sql_columns error: {e}")
        return {"invalid_columns": [], "suggestions": {}, "valid": True}

    # Construire dict {table: [col1, col2, ...]}
    real_schema: Dict[str, List[str]] = {}
    for row in rows:
        t = row["table_name"]
        if t not in real_schema:
            real_schema[t] = []
        real_schema[t].append(row["col_name"].lower())

    import unicodedata as _ud
    def _nc(s):
        s = _ud.normalize('NFD', s.lower())
        return ''.join(c for c in s if _ud.category(c) != 'Mn')
    # Index normalise : "état" → "etat", "Devises" → "devises"
    all_real_cols: set = {col for cols in real_schema.values() for col in cols}
    all_real_cols_norm: set = {_nc(col) for col in all_real_cols}

    # Extraire les colonnes utilisées dans le SQL SELECT
    col_pattern = re.compile(r'\[([^\]]+)\]|(?<!\w)([A-Z_][A-Z0-9_]{2,})(?!\w)', re.IGNORECASE)
    sql_keywords = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'ON', 'AND', 'OR', 'NOT', 'TOP', 'ORDER', 'BY', 'GROUP', 'HAVING',
        'WITH', 'NOLOCK', 'AS', 'IN', 'IS', 'NULL', 'LIKE', 'BETWEEN',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'CASE', 'WHEN',
        'THEN', 'ELSE', 'END', 'ASC', 'DESC', 'GETDATE', 'DATEADD', 'MONTH',
        'YEAR', 'DAY', 'CAST', 'CONVERT', 'ISNULL', 'COALESCE', 'LEN',
        'REVTYPE', 'REVDATEX_ID', 'NOLOCK',
    }

    # Ajouter les noms de tables à ignorer
    sql_keywords.update({t.upper() for t in tables_used})

    # Filtrer les alias définis par AS xxx (ex: AS PRENOM → PRENOM n'est pas une colonne)
    alias_pattern = re.compile(r'\bAS\s+(\w+)', re.IGNORECASE)
    alias_names = {m.group(1).upper() for m in alias_pattern.finditer(sql)}
    sql_keywords.update(alias_names)

    # Filtrer les préfixes de table utilisés comme alias (ex: e.CTPRSFIRST → ignorer 'e')
    prefix_pattern = re.compile(r'\b(\w{1,5})\.', re.IGNORECASE)
    table_aliases = {m.group(1).upper() for m in prefix_pattern.finditer(sql)}
    sql_keywords.update(table_aliases)

    invalid_cols = []
    suggestions  = {}

    for m in col_pattern.finditer(sql):
        col = (m.group(1) or m.group(2) or "").strip()
        if not col or col.upper() in sql_keywords:
            continue
        if col.lower() not in all_real_cols and _nc(col) not in all_real_cols_norm:
            invalid_cols.append(col)
            # Suggérer la colonne la plus proche (distance Levenshtein approx)
            best = _find_closest_column(col.lower(), all_real_cols)
            if best:
                suggestions[col] = best

    logger.info(
        f"[CRAG] Validation SQL : {len(tables_used)} tables"
        f" | {len(invalid_cols)} colonnes invalides : {invalid_cols[:5]}"
    )

    return {
        "invalid_columns": invalid_cols,
        "suggestions":     suggestions,
        "valid":           len(invalid_cols) == 0,
        "tables_checked":  list(tables_used),
    }


def _find_closest_column(col: str, real_cols: set, max_dist: int = 3) -> str:
    """
    Trouve la colonne réelle la plus proche par distance d'édition simplifiée.
    Utilisé pour suggérer des corrections dans validate_sql_columns.
    """
    if not real_cols:
        return ""

    # D'abord chercher par préfixe commun (rapide)
    prefix = col[:4]
    prefix_matches = [c for c in real_cols if c.startswith(prefix)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        # Parmi les matches par préfixe, prendre le plus court
        return min(prefix_matches, key=len)

    # Fallback : contient le col comme substring
    substr_matches = [c for c in real_cols if col in c or c in col]
    if substr_matches:
        return min(substr_matches, key=lambda x: abs(len(x) - len(col)))

    return ""

