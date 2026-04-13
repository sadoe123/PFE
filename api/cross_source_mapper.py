"""
OnePilot — cross_source_mapper.py
===================================
Phase 3 — Relations cross-sources (API ↔ DB ↔ fichiers)

Permet de détecter et naviguer des relations entre entités
appartenant à des sources différentes.

Exemple :
    API Endpoint /orders/{order_id}
        ↓ (order_id maps to)
    DB Table: sales_orders.sap_order_number
        ↓ (joins to)
    Excel File: monthly_report.xlsx → column "Order#"

Algorithme :
  1. Normalisation des identifiants (trim, uppercase, padding)
  2. Analyse des chevauchements de valeurs entre sources
  3. Scoring de compatibilité (type + pattern + overlap)
  4. Stockage des mappings validés
  5. Graph traversal multi-sources pour path-finding
"""

from __future__ import annotations

import logging
import json
import re
import asyncio
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# NORMALISATION DES IDENTIFIANTS
# ══════════════════════════════════════════════════════════════════════

def normalize_identifier(value: Any) -> str:
    """
    Normalise un identifiant pour comparaison cross-sources.
    Règles : strip, uppercase, suppression des zéros de padding, tirets.
    """
    if value is None:
        return ""
    s = str(value).strip()
    # Supprimer les zéros de padding à gauche pour les IDs numériques
    if s.isdigit():
        return s.lstrip("0") or "0"
    # Normaliser les UUIDs
    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.I
    )
    if UUID_RE.match(s.replace("-", "")):
        return s.lower().replace("-", "")
    # Codes alphanumériques : uppercase, sans tirets/espaces
    return s.upper().replace("-", "").replace(" ", "").replace("_", "")


def normalize_values(values: List[Any]) -> Set[str]:
    """Normalise une liste de valeurs pour comparaison."""
    return {normalize_identifier(v) for v in values if v is not None}


def compute_cross_source_overlap(
    values_a: List[Any],
    values_b: List[Any],
    sample_size: int = 500,
) -> Dict:
    """
    Calcule le chevauchement de valeurs entre deux colonnes de sources différentes
    après normalisation.

    Retourne:
    {
        "overlap_ratio": float,   # |A∩B| / |A|
        "coverage_a":   float,   # % de A présents dans B
        "coverage_b":   float,   # % de B présents dans A
        "common_count": int,
        "pattern_match": bool,   # même format détecté
    }
    """
    # Sous-échantillonnage
    sample_a = values_a[:sample_size]
    sample_b = values_b[:sample_size]

    norm_a = normalize_values(sample_a)
    norm_b = normalize_values(sample_b)

    if not norm_a or not norm_b:
        return {"overlap_ratio": 0.0, "coverage_a": 0.0, "coverage_b": 0.0,
                "common_count": 0, "pattern_match": False}

    common = norm_a & norm_b
    overlap = len(common) / len(norm_a | norm_b)
    cov_a = len(common) / len(norm_a)
    cov_b = len(common) / len(norm_b)

    # Détecter le format dominant
    def _dominant_pattern(vals: Set[str]) -> str:
        if not vals:
            return "unknown"
        sample = list(vals)[:20]
        patterns = {
            "uuid": re.compile(r"^[0-9a-f]{32}$"),
            "numeric": re.compile(r"^\d+$"),
            "alpha_code": re.compile(r"^[A-Z0-9]{2,20}$"),
        }
        for name, rx in patterns.items():
            if sum(1 for v in sample if rx.match(v)) / len(sample) > 0.7:
                return name
        return "mixed"

    pat_a = _dominant_pattern(norm_a)
    pat_b = _dominant_pattern(norm_b)

    return {
        "overlap_ratio": round(overlap, 4),
        "coverage_a":    round(cov_a, 4),
        "coverage_b":    round(cov_b, 4),
        "common_count":  len(common),
        "pattern_match": pat_a == pat_b and pat_a != "unknown",
        "pattern_a":     pat_a,
        "pattern_b":     pat_b,
    }


# ══════════════════════════════════════════════════════════════════════
# DÉTECTION DE MAPPINGS CROSS-SOURCES
# ══════════════════════════════════════════════════════════════════════

CROSS_SOURCE_OVERLAP_THRESHOLD = 0.30  # 30% de chevauchement minimum

async def detect_cross_source_mappings(
    source_id_a: UUID,
    source_id_b: UUID,
    min_overlap: float = CROSS_SOURCE_OVERLAP_THRESHOLD,
    sample_size: int = 500,
) -> List[Dict]:
    """
    Détecte les mappings entre colonnes de deux sources différentes.

    Algorithme :
    1. Charger les profils de données (top_values) des deux sources
    2. Comparer toutes les paires de colonnes avec overlap normalisé
    3. Filtrer par seuil et scorer

    Retourne une liste de mappings candidats.
    """
    try:
        from .database import get_pg_pool  # type: ignore
    except ImportError:
        from database import get_pg_pool  # type: ignore

    pool = await get_pg_pool()

    # Charger les profils des deux sources
    async def _load_profiles(source_id: UUID) -> List[Dict]:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT entity_name, profile_data
                    FROM entity_profiles
                    WHERE source_id = $1
                      AND profile_data->>'error' IS NULL
                """, source_id)
            profiles = []
            for row in rows:
                data = row["profile_data"]
                if isinstance(data, str):
                    data = json.loads(data)
                for col in data.get("columns", []):
                    top_values = [
                        tv.get("value") for tv in col.get("top_values", [])
                        if tv.get("value") is not None
                    ]
                    if top_values:
                        profiles.append({
                            "entity":       row["entity_name"],
                            "column":       col["name"],
                            "data_type":    col.get("data_type", "string"),
                            "top_values":   top_values,
                            "unique_count": col.get("unique_count", 0),
                            "null_rate":    col.get("null_rate", 1.0),
                        })
            return profiles
        except Exception as e:
            logger.warning(f"[cross_source] load_profiles {source_id}: {e}")
            return []

    profiles_a, profiles_b = await asyncio.gather(
        _load_profiles(source_id_a),
        _load_profiles(source_id_b),
    )

    if not profiles_a or not profiles_b:
        return []

    logger.info(
        f"[cross_source] Comparaison {len(profiles_a)} × {len(profiles_b)} colonnes"
    )

    candidates = []

    for col_a in profiles_a:
        for col_b in profiles_b:
            # ── Filtre 1 : compatibilité de type ─────────────────────────
            if not _types_compatible(col_a["data_type"], col_b["data_type"]):
                continue

            # ── Filtre 2 : skip colonnes trop nulles ──────────────────────
            if col_a["null_rate"] > 0.9 or col_b["null_rate"] > 0.9:
                continue

            # ── Filtre 3 : skip booléens (unique_count <= 2) ─────────────
            if col_a["unique_count"] <= 2 or col_b["unique_count"] <= 2:
                continue

            # ── Filtre 4 : skip colonnes peu discriminantes ───────────────
            if col_a["unique_count"] < 3 or col_b["unique_count"] < 3:
                continue

            overlap = compute_cross_source_overlap(
                col_a["top_values"],
                col_b["top_values"],
                sample_size,
            )

            # ── Filtre 5 : minimum de valeurs communes ────────────────────
            if overlap["common_count"] < 3:
                continue

            # ── Filtre 6 : rejeter les IDs séquentiels [1,2,3,4,5...] ────
            # Ces petits entiers consécutifs se retrouvent dans toutes les
            # tables de référence → faux positifs systématiques
            def _is_sequential_ints(values) -> bool:
                try:
                    nums = sorted([int(float(v)) for v in values if str(v).strip()])
                    if len(nums) < 3:
                        return False
                    # Séquentiels si max-min == len-1 et tous consécutifs
                    return (nums[-1] - nums[0] == len(nums) - 1
                            and nums[-1] <= 50)  # petits entiers consécutifs
                except (ValueError, TypeError):
                    return False

            common_vals = set(normalize_values(col_a["top_values"])) & \
                          set(normalize_values(col_b["top_values"]))
            if _is_sequential_ints(list(common_vals)):
                continue

            if overlap["coverage_a"] < min_overlap:
                continue

            # ── Filtre 6 : pénaliser les noms de colonnes trop génériques ─
            # Ex: ID, CODE, NUM seuls sont trop ambigus
            generic_names = {"id", "code", "num", "number", "value",
                             "flag", "type", "status", "active", "enabled"}
            col_a_lower = col_a["column"].lower().strip("_")
            col_b_lower = col_b["column"].lower().strip("_")
            name_penalty = 0.3 if (col_a_lower in generic_names or
                                    col_b_lower in generic_names) else 0.0

            # Score de confiance composite
            score = (
                0.5 * overlap["coverage_a"] +
                0.3 * overlap["overlap_ratio"] +
                0.2 * (1.0 if overlap["pattern_match"] else 0.0)
                - name_penalty
            )

            # ── Filtre 7 : score minimum après pénalité ───────────────────
            if score < 0.40:
                continue

            candidates.append({
                "source_id_a":    str(source_id_a),
                "entity_a":       col_a["entity"],
                "column_a":       col_a["column"],
                "source_id_b":    str(source_id_b),
                "entity_b":       col_b["entity"],
                "column_b":       col_b["column"],
                "confidence":     round(score, 4),
                "overlap_ratio":  overlap["overlap_ratio"],
                "coverage_a":     overlap["coverage_a"],
                "coverage_b":     overlap["coverage_b"],
                "common_count":   overlap["common_count"],
                "pattern_match":  overlap["pattern_match"],
                "detection_method": "cross_source_value_overlap",
            })

    # Trier par confiance
    candidates.sort(key=lambda x: -x["confidence"])

    # Dédupliquer : garder le meilleur mapping par (entity_a, column_a)
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["entity_a"], c["column_a"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    logger.info(f"[cross_source] {len(deduped)} mappings candidats")
    return deduped


def _types_compatible(type_a: str, type_b: str) -> bool:
    """Vérifie la compatibilité de deux types de données."""
    INT_TYPES  = {"integer", "int", "bigint", "smallint", "numeric", "float", "number"}
    STR_TYPES  = {"string", "varchar", "nvarchar", "char", "text"}
    DATE_TYPES = {"date", "datetime", "timestamp"}

    def _group(t: str) -> str:
        t = t.lower()
        if any(x in t for x in INT_TYPES):  return "numeric"
        if any(x in t for x in STR_TYPES):  return "string"
        if any(x in t for x in DATE_TYPES): return "date"
        return "other"

    ga, gb = _group(type_a), _group(type_b)
    # Numeric ↔ String compatible si IDs peuvent être stockés comme strings
    if ga == "other" or gb == "other":
        return True
    if ga == gb:
        return True
    if {ga, gb} == {"numeric", "string"}:
        return True  # IDs souvent stockés dans les deux formats
    return False


# ══════════════════════════════════════════════════════════════════════
# PERSISTANCE DES MAPPINGS CROSS-SOURCES
# ══════════════════════════════════════════════════════════════════════

async def save_cross_source_mappings(mappings: List[Dict]) -> int:
    """Sauvegarde les mappings cross-sources en base."""
    try:
        from .database import get_pg_pool  # type: ignore
    except ImportError:
        from database import get_pg_pool  # type: ignore

    if not mappings:
        return 0

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # Créer la table si elle n'existe pas
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_source_mappings (
                id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_id_a     UUID NOT NULL,
                entity_a        VARCHAR(255) NOT NULL,
                column_a        VARCHAR(255) NOT NULL,
                source_id_b     UUID NOT NULL,
                entity_b        VARCHAR(255) NOT NULL,
                column_b        VARCHAR(255) NOT NULL,
                confidence      FLOAT NOT NULL,
                overlap_ratio   FLOAT,
                coverage_a      FLOAT,
                coverage_b      FLOAT,
                common_count    INTEGER,
                pattern_match   BOOLEAN,
                detection_method VARCHAR(100),
                is_validated    BOOLEAN DEFAULT FALSE,
                validated_by    VARCHAR(255),
                created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                UNIQUE(source_id_a, entity_a, column_a, source_id_b, entity_b, column_b)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cross_source_a
                ON cross_source_mappings(source_id_a, entity_a);
            CREATE INDEX IF NOT EXISTS idx_cross_source_b
                ON cross_source_mappings(source_id_b, entity_b);
            CREATE INDEX IF NOT EXISTS idx_cross_source_confidence
                ON cross_source_mappings(confidence DESC);
        """)

        inserted = 0
        for m in mappings:
            try:
                await conn.execute("""
                    INSERT INTO cross_source_mappings
                        (source_id_a, entity_a, column_a,
                         source_id_b, entity_b, column_b,
                         confidence, overlap_ratio, coverage_a, coverage_b,
                         common_count, pattern_match, detection_method)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    ON CONFLICT (source_id_a, entity_a, column_a, source_id_b, entity_b, column_b)
                    DO UPDATE SET
                        confidence    = EXCLUDED.confidence,
                        overlap_ratio = EXCLUDED.overlap_ratio,
                        coverage_a    = EXCLUDED.coverage_a,
                        coverage_b    = EXCLUDED.coverage_b,
                        common_count  = EXCLUDED.common_count,
                        pattern_match = EXCLUDED.pattern_match,
                        created_at    = NOW()
                """,
                    m["source_id_a"], m["entity_a"], m["column_a"],
                    m["source_id_b"], m["entity_b"], m["column_b"],
                    m["confidence"],  m["overlap_ratio"], m["coverage_a"],
                    m["coverage_b"],  m["common_count"],  m["pattern_match"],
                    m["detection_method"],
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"[cross_source] insert error: {e}")

    logger.info(f"[cross_source] {inserted} mappings sauvegardés")
    return inserted


# ══════════════════════════════════════════════════════════════════════
# GRAPHE UNIFIÉ MULTI-SOURCES
# ══════════════════════════════════════════════════════════════════════

class MultiSourceGraph:
    """
    Graphe de relations unifié combinant :
    - Relations intra-source (entity_relations de chaque source)
    - Relations cross-sources (cross_source_mappings)

    Permet le path-finding entre entités de sources différentes.
    Nœuds : "{source_id}::{entity_name}"
    """

    def __init__(self):
        self.graph: Dict[str, List[Tuple]] = defaultdict(list)
        # Tuple : (neighbor_node, via_field, tgt_field, confidence, relation_type)

    def _node(self, source_id: str, entity: str) -> str:
        return f"{source_id}::{entity}"

    def add_intra_relation(
        self,
        source_id: str,
        src_entity: str, src_field: str,
        tgt_entity: str, tgt_field: str,
        confidence: float,
    ):
        src_node = self._node(source_id, src_entity)
        tgt_node = self._node(source_id, tgt_entity)
        self.graph[src_node].append((tgt_node, src_field, tgt_field, confidence, "intra"))
        self.graph[tgt_node].append((src_node, tgt_field, src_field, confidence, "intra"))

    def add_cross_relation(
        self,
        source_id_a: str, entity_a: str, col_a: str,
        source_id_b: str, entity_b: str, col_b: str,
        confidence: float,
    ):
        node_a = self._node(source_id_a, entity_a)
        node_b = self._node(source_id_b, entity_b)
        self.graph[node_a].append((node_b, col_a, col_b, confidence, "cross"))
        self.graph[node_b].append((node_a, col_b, col_a, confidence, "cross"))

    def find_paths(
        self,
        start_source: str, start_entity: str,
        end_source: str,   end_entity: str,
        max_depth: int = 4,
    ) -> List[Dict]:
        """
        BFS pour trouver tous les chemins entre deux entités de sources potentiellement
        différentes.
        """
        start = self._node(start_source, start_entity)
        end   = self._node(end_source, end_entity)

        if start == end:
            return []
        if start not in self.graph and end not in self.graph:
            return []

        queue = deque([(start, [start], [], 1.0)])
        paths = []
        seen_paths: Set[str] = set()
        MAX_RESULTS = 10
        MAX_VISITED = 20_000
        visited_count = 0

        while queue and len(paths) < MAX_RESULTS and visited_count < MAX_VISITED:
            node, path, edges, path_conf = queue.popleft()
            visited_count += 1

            if len(path) - 1 >= max_depth:
                continue

            for (neighbor, via_f, tgt_f, conf, rel_type) in self.graph.get(node, []):
                if neighbor in path:
                    continue
                new_conf  = round(path_conf * conf, 4)
                new_path  = path + [neighbor]
                new_edges = edges + [{
                    "from":      node,
                    "via":       via_f,
                    "to":        neighbor,
                    "tgt_field": tgt_f,
                    "conf":      conf,
                    "type":      rel_type,
                }]

                if neighbor == end:
                    path_key = "->".join(new_path)
                    if path_key not in seen_paths:
                        seen_paths.add(path_key)
                        # Décomposer les nœuds en (source_id, entity)
                        path_decoded = [
                            {"source_id": n.split("::")[0], "entity": n.split("::")[-1]}
                            for n in new_path
                        ]
                        paths.append({
                            "path":        new_path,
                            "path_decoded": path_decoded,
                            "edges":       new_edges,
                            "length":      len(new_path) - 1,
                            "confidence":  new_conf,
                            "cross_source": any(e["type"] == "cross" for e in new_edges),
                            "cross_source_count": sum(1 for e in new_edges if e["type"] == "cross"),
                        })
                else:
                    queue.append((neighbor, new_path, new_edges, new_conf))

        paths.sort(key=lambda p: (p["length"], -p["confidence"]))
        return paths[:MAX_RESULTS]

    @property
    def node_count(self) -> int:
        return len(self.graph)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.graph.values()) // 2


async def build_multi_source_graph(source_ids: List[UUID]) -> MultiSourceGraph:
    """
    Construit le graphe unifié pour un ensemble de sources.

    1. Charge les relations intra-source depuis entity_relations
    2. Charge les mappings cross-sources depuis cross_source_mappings
    3. Retourne le graphe complet
    """
    try:
        from .database import get_pg_pool  # type: ignore
    except ImportError:
        from database import get_pg_pool  # type: ignore

    pool = await get_pg_pool()
    g = MultiSourceGraph()

    async with pool.acquire() as conn:
        # 1. Relations intra-source
        for sid in source_ids:
            try:
                rows = await conn.fetch("""
                    SELECT source_entity, source_field, target_entity, target_field, confidence
                    FROM entity_relations
                    WHERE source_id = $1
                      AND source_entity IS NOT NULL
                      AND target_entity IS NOT NULL
                      AND confidence >= 0.50
                """, sid)
                for row in rows:
                    g.add_intra_relation(
                        str(sid),
                        row["source_entity"], row["source_field"],
                        row["target_entity"], row["target_field"],
                        row["confidence"],
                    )
            except Exception as e:
                logger.warning(f"[multi_source_graph] intra {sid}: {e}")

        # 2. Relations cross-sources
        try:
            sid_strs = [str(s) for s in source_ids]
            cross_rows = await conn.fetch("""
                SELECT source_id_a, entity_a, column_a,
                       source_id_b, entity_b, column_b, confidence
                FROM cross_source_mappings
                WHERE source_id_a = ANY($1::uuid[])
                  AND source_id_b = ANY($1::uuid[])
                  AND confidence >= 0.30
                ORDER BY confidence DESC
            """, source_ids)

            for row in cross_rows:
                g.add_cross_relation(
                    str(row["source_id_a"]), row["entity_a"], row["column_a"],
                    str(row["source_id_b"]), row["entity_b"], row["column_b"],
                    row["confidence"],
                )
        except Exception as e:
            logger.warning(f"[multi_source_graph] cross: {e}")

    logger.info(
        f"[multi_source_graph] {g.node_count} nœuds, {g.edge_count} arêtes "
        f"pour {len(source_ids)} sources"
    )
    return g


# ══════════════════════════════════════════════════════════════════════
# API PUBLIQUE — endpoints à ajouter dans main.py
# ══════════════════════════════════════════════════════════════════════
# Ces fonctions sont des wrappers prêts à câbler dans FastAPI :
#
# POST /cross-source/detect
#   body: {source_id_a, source_id_b, min_overlap=0.30}
#   → detect_cross_source_mappings() + save_cross_source_mappings()
#
# GET /cross-source/mappings
#   params: source_id_a, source_id_b
#   → query cross_source_mappings
#
# POST /cross-source/path
#   body: {source_ids: [...], from: {source_id, entity}, to: {source_id, entity}}
#   → build_multi_source_graph() + graph.find_paths()
#
# PUT /cross-source/mappings/{id}/validate
#   → UPDATE cross_source_mappings SET is_validated=TRUE, validated_by=...

async def get_cross_source_mappings(
    source_id_a: UUID,
    source_id_b: Optional[UUID] = None,
    min_confidence: float = 0.30,
) -> List[Dict]:
    """Récupère les mappings cross-sources existants."""
    try:
        from .database import get_pg_pool  # type: ignore
    except ImportError:
        from database import get_pg_pool  # type: ignore

    pool = await get_pg_pool()

    try:
        async with pool.acquire() as conn:
            if source_id_b:
                rows = await conn.fetch("""
                    SELECT * FROM cross_source_mappings
                    WHERE source_id_a = $1 AND source_id_b = $2
                      AND confidence >= $3
                    ORDER BY confidence DESC
                """, source_id_a, source_id_b, min_confidence)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM cross_source_mappings
                    WHERE (source_id_a = $1 OR source_id_b = $1)
                      AND confidence >= $2
                    ORDER BY confidence DESC
                """, source_id_a, min_confidence)

        return [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"[cross_source] get_mappings: {e}")
        return []