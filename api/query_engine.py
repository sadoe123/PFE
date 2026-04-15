"""
OnePilot – Query Engine §2.3.3
SQL Generator + API Query Builder + Universal Query Planner
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from .nlu_engine import Intent, QuerySlots

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SQL GENERATOR §2.3.3.A
# ══════════════════════════════════════════════════════════════════════

class SQLGenerator:
    """
    Génère du SQL depuis les slots NLU.
    Approche hybride : templates pour les cas standards, LLM pour les complexes.
    """

    MAX_ROWS = 1000  # Sécurité : jamais de SELECT * sans LIMIT

    def generate(
        self,
        slots: QuerySlots,
        schema_context: Dict,  # {table: [fields]}
        dialect: str = "sql",  # sql, mssql, postgresql, mysql
    ) -> Dict:
        """
        Génère une requête SQL depuis les slots.
        Retourne {sql, explanation, params, warnings}
        """
        intent = slots.intent
        tables = slots.table_names
        schema = schema_context

        try:
            if intent == Intent.GENERATE_AGG:
                return self._generate_aggregate(slots, schema, dialect)
            elif intent == Intent.GENERATE_JOIN:
                return self._generate_join(slots, schema, dialect)
            elif intent == Intent.GENERATE_FILTER:
                return self._generate_filter(slots, schema, dialect)
            elif intent in (Intent.GENERATE_SQL, Intent.LIST_ENTITIES):
                return self._generate_select(slots, schema, dialect)
            elif intent == Intent.COUNT_ENTITIES:
                return self._generate_count(slots, schema, dialect)
            elif intent == Intent.PROFILE_ENTITY:
                return self._generate_profile(slots, schema, dialect)
            else:
                return self._generate_select(slots, schema, dialect)
        except Exception as e:
            logger.warning(f"[SQLGen] Error: {e}")
            return {
                "sql":         self._fallback_sql(tables, dialect),
                "explanation": f"Requête générique (erreur: {e})",
                "params":      {},
                "warnings":    [str(e)],
            }

    def _generate_select(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        """SELECT simple avec filtres optionnels."""
        table = slots.table_names[0] if slots.table_names else list(schema.keys())[0] if schema else "table"
        fields = schema.get(table, [])

        select_cols = "*"
        if fields and len(fields) <= 20:
            select_cols = ", ".join(f[:15] for f in fields[:15])

        sql = f"SELECT {select_cols}\nFROM {self._quote(table, dialect)}"

        where_clauses, params = self._build_where(slots, table, fields, dialect)
        if where_clauses:
            sql += f"\nWHERE {' AND '.join(where_clauses)}"

        limit = slots.top_n or 100
        sql += f"\n{self._limit(limit, dialect)}"

        return {
            "sql":         sql,
            "explanation": f"Sélection des données de '{table}'" + (f" — top {limit}" if slots.top_n else ""),
            "params":      params,
            "warnings":    self._check_warnings(slots, table),
        }

    def _generate_aggregate(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        """SELECT avec agrégation (SUM, AVG, COUNT...)."""
        table = slots.table_names[0] if slots.table_names else list(schema.keys())[0] if schema else "table"
        fields = schema.get(table, [])
        metric = slots.metric or "COUNT"
        group_by = slots.group_by

        # Cherche le champ numérique le plus probable pour l'agrégation
        numeric_field = self._find_numeric_field(fields, slots)
        agg_field = f"{metric}({numeric_field})" if numeric_field and metric != "COUNT" else f"{metric}(*)"

        if group_by:
            # Cherche le vrai nom du champ group_by dans le schéma
            group_field = self._match_field(group_by, fields) or group_by
            sql = (
                f"SELECT {self._quote(group_field, dialect)},\n"
                f"       {agg_field} AS {metric.lower()}_total\n"
                f"FROM {self._quote(table, dialect)}"
            )
            where_clauses, params = self._build_where(slots, table, fields, dialect)
            if where_clauses:
                sql += f"\nWHERE {' AND '.join(where_clauses)}"
            sql += f"\nGROUP BY {self._quote(group_field, dialect)}"
            sql += f"\nORDER BY {metric.lower()}_total DESC"
            sql += f"\n{self._limit(slots.top_n or 20, dialect)}"
            explanation = f"{metric} de '{table}' groupé par '{group_field}'"
        else:
            sql = (
                f"SELECT {agg_field} AS result\n"
                f"FROM {self._quote(table, dialect)}"
            )
            where_clauses, params = self._build_where(slots, table, fields, dialect)
            if where_clauses:
                sql += f"\nWHERE {' AND '.join(where_clauses)}"
            explanation = f"{metric} global de '{table}'"
            params = {}

        return {
            "sql": sql, "explanation": explanation,
            "params": params, "warnings": self._check_warnings(slots, table),
        }

    def _generate_join(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        """JOIN entre deux tables."""
        tables = slots.table_names
        if len(tables) < 2:
            return self._generate_select(slots, schema, dialect)

        t1, t2 = tables[0], tables[1]
        f1 = schema.get(t1, [])
        f2 = schema.get(t2, [])

        # Cherche la clé de jointure
        join_col = self._find_join_key(t1, t2, f1, f2)

        if join_col:
            sql = (
                f"SELECT a.*, b.*\n"
                f"FROM {self._quote(t1, dialect)} a\n"
                f"JOIN {self._quote(t2, dialect)} b\n"
                f"  ON a.{join_col[0]} = b.{join_col[1]}\n"
                f"{self._limit(100, dialect)}"
            )
            explanation = f"Jointure entre '{t1}' et '{t2}' via {join_col[0]} → {join_col[1]}"
        else:
            sql = (
                f"SELECT a.*, b.*\n"
                f"FROM {self._quote(t1, dialect)} a\n"
                f"-- ⚠️ Clé de jointure non trouvée automatiquement\n"
                f"JOIN {self._quote(t2, dialect)} b ON a.id = b.{t1.lower()}_id\n"
                f"{self._limit(100, dialect)}"
            )
            explanation = f"Jointure entre '{t1}' et '{t2}' — clé déduite (à vérifier)"

        return {
            "sql": sql, "explanation": explanation,
            "params": {}, "warnings": ["Vérifiez la clé de jointure"] if not join_col else [],
        }

    def _generate_filter(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        """SELECT avec filtre."""
        return self._generate_select(slots, schema, dialect)

    def _generate_count(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        """COUNT(*)."""
        table = slots.table_names[0] if slots.table_names else list(schema.keys())[0] if schema else "table"
        sql = f"SELECT COUNT(*) AS total\nFROM {self._quote(table, dialect)}"
        return {
            "sql": sql, "explanation": f"Nombre de lignes dans '{table}'",
            "params": {}, "warnings": [],
        }

    def _generate_profile(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        """Profiling statistique d'une table."""
        table = slots.table_names[0] if slots.table_names else list(schema.keys())[0] if schema else "table"
        fields = schema.get(table, [])[:5]

        if dialect in ("mssql", "sql_server"):
            stats = "\n  UNION ALL\n  ".join([
                f"SELECT '{f}' AS col, COUNT(*) AS total, "
                f"COUNT({f}) AS non_null, "
                f"COUNT(DISTINCT {f}) AS distinct_vals "
                f"FROM {self._quote(table, dialect)}"
                for f in fields
            ]) if fields else f"SELECT COUNT(*) FROM {self._quote(table, dialect)}"
        else:
            stats = f"SELECT COUNT(*) AS total_rows FROM {self._quote(table, dialect)}"

        return {
            "sql": stats, "explanation": f"Profil statistique de '{table}'",
            "params": {}, "warnings": [],
        }

    def _build_where(self, slots: QuerySlots, table: str, fields: List[str], dialect: str) -> Tuple[List[str], Dict]:
        """Construit les clauses WHERE depuis les slots."""
        clauses = []
        params = {}

        # Filtre date
        if slots.date_filter:
            df = slots.date_filter
            date_field = self._find_date_field(fields)
            if date_field:
                if df.get("type") == "range":
                    clauses.append(f"{self._quote(date_field, dialect)} BETWEEN '{df['from']}' AND '{df['to']}'")
                elif df.get("type") == "date":
                    clauses.append(f"{self._quote(date_field, dialect)} = '{df['from']}'")

        # Filtre montant
        if slots.amount_filter:
            af = slots.amount_filter
            amount_field = self._find_amount_field(fields)
            op_map = {"gt": ">", "lt": "<", "eq": "=", "between": "BETWEEN"}
            op = op_map.get(af.get("op", "eq"), "=")
            if amount_field:
                clauses.append(f"{self._quote(amount_field, dialect)} {op} {af['value']}")

        return clauses, params

    def _find_date_field(self, fields: List[str]) -> Optional[str]:
        """Trouve le champ date le plus probable."""
        date_keywords = ["date", "at", "time", "created", "modified", "updated", "when"]
        for kw in date_keywords:
            for f in fields:
                if kw in f.lower():
                    return f
        return None

    def _find_amount_field(self, fields: List[str]) -> Optional[str]:
        """Trouve le champ montant le plus probable."""
        amount_keywords = ["amount", "total", "price", "cost", "value", "montant", "prix", "sum"]
        for kw in amount_keywords:
            for f in fields:
                if kw in f.lower():
                    return f
        return None

    def _find_numeric_field(self, fields: List[str], slots: QuerySlots) -> Optional[str]:
        """Trouve le champ numérique pour l'agrégation."""
        # 1. Champ explicitement mentionné dans les slots
        if slots.field_names:
            matched = self._match_field(slots.field_names[0], fields)
            if matched:
                return matched

        # 2. Cherche dans les entités brutes (noms de champs dans la question)
        if slots.raw_entities:
            for ent in slots.raw_entities:
                if ent.entity_type == "table":
                    matched = self._match_field(ent.value, fields)
                    if matched and matched != slots.table_names[0] if slots.table_names else True:
                        return matched

        # 3. Recherche par mot-clé dans le nom du champ
        numeric_kw = ["freight", "amount", "total", "price", "qty", "quantity",
                      "value", "montant", "prix", "qte", "sum", "cost", "revenue",
                      "shipping", "tax", "discount", "salary", "wage",
                      "budget", "invoice", "payment", "fee", "charge", "rate",
                      "weight", "poids", "unitprice", "extended"]
        for kw in numeric_kw:
            for f in fields:
                if kw in f.lower():
                    return f

        # 4. Fallback : premier champ non-PK non-FK non-string
        id_kw = ["id", "code", "name", "country", "city", "address", "phone", "email", "desc"]
        for f in fields:
            if not any(kw in f.lower() for kw in id_kw):
                return f

        return fields[0] if fields else None

    def _find_join_key(self, t1: str, t2: str, f1: List[str], f2: List[str]) -> Optional[Tuple[str, str]]:
        """Trouve la clé de jointure entre deux tables."""
        # t1_id dans f2
        for f in f2:
            if t1.lower().rstrip("s") in f.lower() and "id" in f.lower():
                for pk in f1:
                    if pk.lower() in ("id", f"{t1.lower()}_id", "pk"):
                        return (pk, f)

        # t2_id dans f1
        for f in f1:
            if t2.lower().rstrip("s") in f.lower() and "id" in f.lower():
                for pk in f2:
                    if pk.lower() in ("id", f"{t2.lower()}_id", "pk"):
                        return (f, pk)

        # id commun
        common = set(fn.lower() for fn in f1) & set(fn.lower() for fn in f2)
        id_common = [c for c in common if "id" in c]
        if id_common:
            return (id_common[0], id_common[0])

        return None

    def _match_field(self, name: str, fields: List[str]) -> Optional[str]:
        """Trouve le champ correspondant dans le schéma."""
        for f in fields:
            if name.lower() == f.lower() or name.lower() in f.lower():
                return f
        return None

    def _quote(self, name: str, dialect: str) -> str:
        """Quote un identifiant selon le dialecte."""
        if dialect in ("mssql", "sql_server"):
            return f"[{name}]"
        elif dialect in ("mysql",):
            return f"`{name}`"
        return f'"{name}"'

    def _limit(self, n: int, dialect: str) -> str:
        if dialect in ("mssql", "sql_server"):
            return ""  # TOP N mis dans SELECT pour MSSQL
        return f"LIMIT {min(n, self.MAX_ROWS)}"

    def _fallback_sql(self, tables: List[str], dialect: str) -> str:
        table = tables[0] if tables else "your_table"
        return f"SELECT *\nFROM {self._quote(table, dialect)}\n{self._limit(100, dialect)}"

    def _check_warnings(self, slots: QuerySlots, table: str) -> List[str]:
        warnings = []
        if slots.ambiguities:
            warnings.extend(slots.ambiguities)
        return warnings


# ══════════════════════════════════════════════════════════════════════
# UNIVERSAL QUERY PLANNER §2.3.3.C
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QueryStep:
    step_id:     int
    source_id:   str
    source_type: str  # database, webservice, file
    action:      str  # sql, api_call, merge, filter, sort
    query:       str
    depends_on:  List[int] = field(default_factory=list)
    parallel:    bool = False
    cache_ttl:   int = 300


@dataclass
class QueryPlan:
    question:     str
    steps:        List[QueryStep]
    merge_strategy: str  # join, union, enrich
    explanation:  str
    estimated_ms: int = 0


class UniversalQueryPlanner:
    """
    Décompose une question complexe en sous-requêtes atomiques
    sur plusieurs sources hétérogènes (DB + API + Files).
    """

    def plan(
        self,
        slots: QuerySlots,
        sources: List[Dict],
        schemas: Dict[str, Dict],
    ) -> QueryPlan:
        """
        Crée un plan d'exécution multi-source.
        sources = [{id, name, connector_type, entity_count}]
        schemas = {source_id: {table: [fields]}}
        """
        sql_gen = SQLGenerator()
        steps   = []
        step_id = 0

        # ── Cas 1 : Source unique ────────────────────────────────────
        if len(slots.table_names) <= 2 and len(sources) == 1:
            src = sources[0]
            schema = schemas.get(str(src["id"]), {})
            dialect = self._get_dialect(src["connector_type"])
            result  = sql_gen.generate(slots, schema, dialect)

            steps.append(QueryStep(
                step_id     = step_id,
                source_id   = str(src["id"]),
                source_type = src.get("category", "database"),
                action      = "sql",
                query       = result["sql"],
                parallel    = False,
                cache_ttl   = 60,
            ))

            return QueryPlan(
                question      = "",
                steps         = steps,
                merge_strategy= "none",
                explanation   = result["explanation"],
            )

        # ── Cas 2 : Cross-source ─────────────────────────────────────
        db_sources  = [s for s in sources if s.get("category") == "database"]
        api_sources = [s for s in sources if s.get("category") == "webservice"]

        # Step 1 : Requête DB principale (parallélisable)
        for src in db_sources[:2]:
            schema = schemas.get(str(src["id"]), {})
            dialect = self._get_dialect(src["connector_type"])
            result  = sql_gen.generate(slots, schema, dialect)
            steps.append(QueryStep(
                step_id     = step_id,
                source_id   = str(src["id"]),
                source_type = "database",
                action      = "sql",
                query       = result["sql"],
                parallel    = True,
                cache_ttl   = 300,
            ))
            step_id += 1

        # Step 2 : Enrichissement API (dépend du step DB)
        for src in api_sources[:1]:
            steps.append(QueryStep(
                step_id     = step_id,
                source_id   = str(src["id"]),
                source_type = "webservice",
                action      = "api_call",
                query       = self._build_api_query(slots, src),
                depends_on  = [s.step_id for s in steps if s.source_type == "database"],
                parallel    = False,
                cache_ttl   = 120,
            ))
            step_id += 1

        # Step final : Merge
        if len(steps) > 1:
            steps.append(QueryStep(
                step_id     = step_id,
                source_id   = "merger",
                source_type = "internal",
                action      = "merge",
                query       = json.dumps({
                    "strategy": "join",
                    "on":       self._find_merge_key(slots, schemas),
                }),
                depends_on  = [s.step_id for s in steps],
                parallel    = False,
                cache_ttl   = 0,
            ))

        return QueryPlan(
            question      = "",
            steps         = steps,
            merge_strategy= "join" if len(steps) > 1 else "none",
            explanation   = f"Plan cross-source : {len(steps)} étapes",
            estimated_ms  = len(steps) * 200,
        )

    def _get_dialect(self, connector_type: str) -> str:
        if "mssql" in connector_type or "sage_100" in connector_type:
            return "mssql"
        if "mysql" in connector_type:
            return "mysql"
        return "postgresql"

    def _build_api_query(self, slots: QuerySlots, source: Dict) -> str:
        """Construit une URL d'API depuis les slots."""
        base = source.get("base_url", "")
        if slots.table_names:
            return f"{base}/{slots.table_names[0].lower()}"
        return base

    def _find_merge_key(self, slots: QuerySlots, schemas: Dict) -> str:
        """Trouve la clé commune pour le merge cross-source."""
        common_keys = ["id", "customer_id", "order_id", "product_id"]
        for key in common_keys:
            for schema in schemas.values():
                for fields in schema.values():
                    if key in [f.lower() for f in fields]:
                        return key
        return "id"
