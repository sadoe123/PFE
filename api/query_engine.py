"""
OnePilot – Query Engine §2.3.3  ── 100% complet
SQL Generator + CTE/Window/HAVING + API Query Builder (OData) + SQL Validator
Universal Query Planner (cross-source)
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
# §2.3.3.D  SQL VALIDATOR  ── nouveau
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    valid:    bool
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score:    float = 1.0   # 1.0 = parfait, 0.0 = invalide


class SQLValidator:
    """
    Validation syntaxique SQL avant exécution.
    - Protège contre DML/DDL et injections
    - Vérifie structure minimale SELECT … FROM
    - Contrôle parenthèses, guillemets, TOP/LIMIT cohérence
    - Ne nécessite pas de connexion DB
    """

    # ── Mots-clés DML/DDL interdits ──────────────────────────────────
    _FORBIDDEN = re.compile(
        r"\b(INSERT\s+INTO|UPDATE\s+\w|DELETE\s+FROM|DROP\s+\w|TRUNCATE\s+\w|"
        r"ALTER\s+\w|CREATE\s+\w|EXEC(?:UTE)?\s|xp_cmdshell|sp_executesql|"
        r"OPENROWSET|BULK\s+INSERT|MERGE\s+INTO)\b",
        re.IGNORECASE,
    )

    # ── Patterns injection basiques ───────────────────────────────────
    _INJECTION = [
        re.compile(r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP)", re.IGNORECASE),   # stacked queries
        re.compile(r"'\s*OR\s*'?\d+'?\s*=\s*'?\d+'?",          re.IGNORECASE),   # ' OR '1'='1
        re.compile(r"--\s*$",                                   re.MULTILINE),    # commentaire fin de ligne
        re.compile(r"/\*.*?\*/",                                re.DOTALL),       # bloc commentaire
    ]

    # ── Structure minimale ────────────────────────────────────────────
    _HAS_SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)
    _HAS_FROM   = re.compile(r"\bFROM\b",   re.IGNORECASE)

    # ── TOP/LIMIT de sécurité ─────────────────────────────────────────
    _HAS_TOP    = re.compile(r"\bTOP\s+\d+\b",    re.IGNORECASE)
    _HAS_LIMIT  = re.compile(r"\bLIMIT\s+\d+\b",  re.IGNORECASE)
    _MAX_SAFE   = 10_000

    @staticmethod
    def _balanced_parens(sql: str) -> bool:
        depth = 0
        in_str, str_ch = False, None
        for ch in sql:
            if in_str:
                if ch == str_ch:
                    in_str = False
            elif ch in ("'", '"'):
                in_str, str_ch = True, ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    @staticmethod
    def _balanced_quotes(sql: str) -> bool:
        return (sql.count("'") - sql.count("\\'")) % 2 == 0

    @staticmethod
    def _extract_top_n(sql: str) -> Optional[int]:
        m = re.search(r"\bTOP\s+(\d+)\b", sql, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"\bLIMIT\s+(\d+)\b", sql, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def validate(
        self,
        sql: str,
        dialect: str = "mssql",
        allow_no_limit: bool = False,
    ) -> ValidationResult:
        """
        Valide une requête SQL.
        Retourne un ValidationResult avec errors / warnings / score.
        """
        errors:   List[str] = []
        warnings: List[str] = []

        # 1. DML / DDL
        m = self._FORBIDDEN.search(sql)
        if m:
            errors.append(f"Instruction interdite détectée : '{m.group(0).strip()}'. "
                          f"Seules les requêtes SELECT sont autorisées.")

        # 2. Injections
        for pat in self._INJECTION:
            if pat.search(sql):
                errors.append("Pattern d'injection SQL détecté.")
                break

        # 3. Structure SELECT … FROM
        if not self._HAS_SELECT.search(sql):
            errors.append("La requête doit contenir SELECT.")
        if not self._HAS_FROM.search(sql):
            errors.append("La requête doit contenir FROM.")

        # 4. Parenthèses
        if not self._balanced_parens(sql):
            errors.append("Parenthèses non équilibrées.")

        # 5. Guillemets
        if not self._balanced_quotes(sql):
            errors.append("Guillemets simples non fermés.")

        # 6. Limite de lignes
        if not allow_no_limit:
            top_n = self._extract_top_n(sql)
            if top_n is None:
                if dialect in ("mssql", "sql_server"):
                    warnings.append("Aucun TOP N — risque de retour de trop de lignes.")
                else:
                    warnings.append("Aucun LIMIT — risque de retour de trop de lignes.")
            elif top_n > self._MAX_SAFE:
                warnings.append(f"TOP/LIMIT {top_n} élevé (> {self._MAX_SAFE}). Vérifiez.")

        # 7. Score
        score = 1.0 - (len(errors) * 0.4) - (len(warnings) * 0.05)
        score = max(0.0, round(score, 2))

        return ValidationResult(
            valid    = len(errors) == 0,
            errors   = errors,
            warnings = warnings,
            score    = score,
        )


# ══════════════════════════════════════════════════════════════════════
# §2.3.3.A+  SQL OPTIMIZER — Explain Plan + Index Hints
# ══════════════════════════════════════════════════════════════════════

class SQLOptimizer:
    """
    Optimisation SQL post-génération :
    - Ajout WITH(NOLOCK) pour lectures SQL Server (évite les locks)
    - Détection tables sans index sur colonnes filtrées → warning
    - Explain plan textuel basé sur structure de la requête
    """

    # Tables transactionnelles → toujours WITH(NOLOCK)
    _NOLOCK_TABLES = {
        "orders", "order_details", "transactions", "journal",
        "si_tresorerie", "gs_acc", "financement_bi", "rc_bal",
    }

    def optimize(
        self,
        sql: str,
        schema: Dict[str, List[str]],
        dialect: str = "mssql",
        add_nolock: bool = True,
    ) -> Dict[str, Any]:
        """
        Applique les optimisations SQL.
        Retourne {sql, warnings, index_hints, explain_plan}.
        """
        warnings = []
        index_hints = []
        optimized = sql

        if dialect == "mssql" and add_nolock:
            optimized, nolock_applied = self._add_nolock(sql, schema)
            if nolock_applied:
                index_hints.append("WITH(NOLOCK) appliqué sur tables transactionnelles")

        # Détecte absence d'index probable
        missing_idx = self._detect_missing_index(sql, schema)
        if missing_idx:
            warnings.extend(missing_idx)

        explain = self._build_explain_plan(sql, schema)

        return {
            "sql":          optimized,
            "warnings":     warnings,
            "index_hints":  index_hints,
            "explain_plan": explain,
        }

    def _add_nolock(self, sql: str, schema: Dict[str, List[str]]) -> tuple:
        """Ajoute WITH(NOLOCK) sur les tables — évite les doublons."""
        import re
        # D'abord supprimer tous les WITH(NOLOCK) existants pour éviter les doublons
        sql_clean = re.sub(r'\s*WITH\s*\(NOLOCK\)', '', sql, flags=re.IGNORECASE)
        applied = False
        def _replace(m):
            nonlocal applied
            tbl = m.group(1).strip('[]')
            alias = m.group(2) or ''
            if tbl.lower() in self._NOLOCK_TABLES or tbl.lower() in [k.lower() for k in schema.keys()]:
                applied = True
                return f"[{tbl}] WITH(NOLOCK){alias}"
            return m.group(0)
        result = re.sub(
            r'\[([^\]]+)\](\s+(?:AS\s+)?\w+)?',
            _replace, sql_clean, flags=re.IGNORECASE
        )
        return result, applied

    def _detect_missing_index(self, sql: str, schema: Dict[str, List[str]]) -> List[str]:
        """Détecte les colonnes WHERE sans index probable."""
        import re
        warnings = []
        where_cols = re.findall(r'WHERE.*?\[(\w+)\]', sql, re.IGNORECASE | re.DOTALL)
        non_indexed = {"description", "notes", "remarks", "comment", "address"}
        for col in where_cols:
            if col.lower() in non_indexed:
                warnings.append(f"⚠️ Colonne [{col}] probablement sans index — filtre lent")
        return warnings

    def _build_explain_plan(self, sql: str, schema: Dict[str, List[str]]) -> str:
        """Génère un explain plan textuel simplifié."""
        import re
        lines = ["Plan d'exécution estimé:"]
        sql_up = sql.upper()

        # Détecte les opérations
        tables = re.findall(r'FROM\s+\[([^\]]+)\]|JOIN\s+\[([^\]]+)\]', sql, re.IGNORECASE)
        flat_tables = [t[0] or t[1] for t in tables]

        if len(flat_tables) > 1:
            lines.append(f"  1. Hash Join sur {' ⟶ '.join(flat_tables)}")
        elif flat_tables:
            lines.append(f"  1. Table Scan / Index Seek sur [{flat_tables[0]}]")

        if 'GROUP BY' in sql_up:
            lines.append("  2. Hash Aggregate (GROUP BY)")
        if 'ORDER BY' in sql_up:
            lines.append("  3. Sort")
        if 'HAVING' in sql_up:
            lines.append("  4. Filter (HAVING)")

        top_m = re.search(r'TOP\s+(\d+)', sql, re.IGNORECASE)
        if top_m:
            lines.append(f"  → Résultat limité à TOP {top_m.group(1)}")

        return "\n".join(lines)


_sql_optimizer = SQLOptimizer()

def get_sql_optimizer() -> SQLOptimizer:
    return _sql_optimizer


# ══════════════════════════════════════════════════════════════════════
# §2.3.3.E  API QUERY BUILDER (OData-style)  ── nouveau
# ══════════════════════════════════════════════════════════════════════

@dataclass
class APIQueryParams:
    """Paramètres de requête OData-style."""
    select:  List[str]      = field(default_factory=list)
    filter:  Optional[str]  = None
    orderby: List[str]      = field(default_factory=list)
    top:     int            = 100
    skip:    int            = 0
    expand:  List[str]      = field(default_factory=list)
    count:   bool           = False
    search:  Optional[str]  = None


class APIQueryBuilder:
    """
    Construit des requêtes API OData-style depuis les slots NLU.
    Gère : $select, $filter, $orderby, $top, $skip, $expand, pagination.
    Compatible REST / OData v4.
    """

    MAX_PAGE = 1000

    # Mapping opérateurs NLU → OData
    _OP_MAP = {
        "gt": "gt", "lt": "lt", "eq": "eq",
        "gte": "ge", "lte": "le", "ne": "ne",
        ">": "gt", "<": "lt", "=": "eq",
        ">=": "ge", "<=": "le", "!=": "ne",
    }

    def build(
        self,
        slots: QuerySlots,
        endpoint_fields: List[str],
        base_url: str = "",
        page: int = 0,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Construit les paramètres de requête API.
        Retourne {url, params, headers, explanation}.
        """
        params = APIQueryParams()

        # ── $select ──────────────────────────────────────────────────
        if slots.field_names:
            # Champs explicitement demandés
            params.select = [
                f for f in endpoint_fields
                if any(req.lower() in f.lower() for req in slots.field_names)
            ]
        elif endpoint_fields and len(endpoint_fields) <= 20:
            # Sélectionne tous les champs si liste courte
            params.select = endpoint_fields[:]
        # Sinon : pas de $select → API retourne tout

        # ── $filter ───────────────────────────────────────────────────
        filter_parts = []

        if slots.date_filter:
            df = slots.date_filter
            date_field = self._find_field(endpoint_fields, ["date", "at", "created", "time"])
            if date_field:
                if df.get("type") == "range":
                    filter_parts.append(
                        f"{date_field} ge {df['from']}T00:00:00Z and "
                        f"{date_field} le {df['to']}T23:59:59Z"
                    )
                elif df.get("type") == "date":
                    filter_parts.append(f"{date_field} eq {df['from']}T00:00:00Z")

        if slots.amount_filter:
            af = slots.amount_filter
            amount_field = self._find_field(
                endpoint_fields,
                ["amount", "total", "price", "cost", "value"]
            )
            odata_op = self._OP_MAP.get(af.get("op", "eq"), "eq")
            if amount_field:
                filter_parts.append(f"{amount_field} {odata_op} {af['value']}")

        if filter_parts:
            params.filter = " and ".join(filter_parts)

        # ── $search ───────────────────────────────────────────────────
        if slots.search_term:
            params.search = slots.search_term

        # ── $orderby ──────────────────────────────────────────────────
        if slots.group_by:
            order_field = self._find_field(endpoint_fields, [slots.group_by])
            if order_field:
                params.orderby = [f"{order_field} asc"]

        if slots.metric in ("SUM", "COUNT", "MAX"):
            # Tri décroissant sur le champ numérique
            num_field = self._find_field(endpoint_fields, ["amount", "total", "value"])
            if num_field:
                params.orderby = [f"{num_field} desc"]

        # ── $top / $skip (pagination) ─────────────────────────────────
        params.top  = min(slots.top_n or page_size, self.MAX_PAGE)
        params.skip = page * params.top

        # ── $expand (JOIN hints) ──────────────────────────────────────
        if len(slots.table_names) > 1:
            params.expand = slots.table_names[1:]

        # ── $count ────────────────────────────────────────────────────
        params.count = slots.intent == Intent.COUNT_ENTITIES

        # ── Construction URL finale ───────────────────────────────────
        url, qs = self._build_url(base_url, slots.table_names, params)

        explanation = self._explain(slots, params, page)

        return {
            "url":         url,
            "query_string": qs,
            "params":      self._to_dict(params),
            "explanation": explanation,
            "page":        page,
            "page_size":   params.top,
        }

    def build_batch(
        self,
        slots: QuerySlots,
        endpoint_fields: List[str],
        base_url: str = "",
        total_pages: int = 3,
    ) -> List[Dict[str, Any]]:
        """Construit plusieurs requêtes paginées en parallèle."""
        return [
            self.build(slots, endpoint_fields, base_url, page=p, page_size=100)
            for p in range(total_pages)
        ]

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _find_field(fields: List[str], keywords: List[str]) -> Optional[str]:
        for kw in keywords:
            for f in fields:
                if kw.lower() in f.lower():
                    return f
        return None

    @staticmethod
    def _build_url(
        base: str,
        table_names: List[str],
        params: APIQueryParams,
    ) -> Tuple[str, str]:
        endpoint = table_names[0].lower() if table_names else "records"
        url = f"{base.rstrip('/')}/{endpoint}" if base else f"/{endpoint}"

        parts: List[str] = []
        if params.select:
            parts.append(f"$select={','.join(params.select)}")
        if params.filter:
            parts.append(f"$filter={params.filter}")
        if params.orderby:
            parts.append(f"$orderby={','.join(params.orderby)}")
        if params.top:
            parts.append(f"$top={params.top}")
        if params.skip:
            parts.append(f"$skip={params.skip}")
        if params.expand:
            parts.append(f"$expand={','.join(params.expand)}")
        if params.count:
            parts.append("$count=true")
        if params.search:
            parts.append(f"$search={params.search}")

        qs = "&".join(parts)
        return (f"{url}?{qs}" if qs else url), qs

    @staticmethod
    def _to_dict(params: APIQueryParams) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if params.select:  d["$select"]  = ",".join(params.select)
        if params.filter:  d["$filter"]  = params.filter
        if params.orderby: d["$orderby"] = ",".join(params.orderby)
        if params.top:     d["$top"]     = params.top
        if params.skip:    d["$skip"]    = params.skip
        if params.expand:  d["$expand"]  = ",".join(params.expand)
        if params.count:   d["$count"]   = "true"
        if params.search:  d["$search"]  = params.search
        return d

    @staticmethod
    def _explain(slots: QuerySlots, params: APIQueryParams, page: int) -> str:
        parts = []
        if params.select:
            parts.append(f"{len(params.select)} champ(s) sélectionné(s)")
        if params.filter:
            parts.append("avec filtre")
        if params.orderby:
            parts.append(f"trié par {params.orderby[0]}")
        if page > 0:
            parts.append(f"page {page + 1} (skip={params.skip})")
        return "Requête API : " + (", ".join(parts) if parts else "liste complète")


# ══════════════════════════════════════════════════════════════════════
# §2.3.3.F  CTE / WINDOW / HAVING BUILDER  ── nouveau
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CTEDefinition:
    name:  str
    query: str
    description: str = ""


@dataclass
class WindowSpec:
    function:     str
    partition_by: List[str] = field(default_factory=list)
    order_by:     List[str] = field(default_factory=list)
    frame:        Optional[str] = None   # ex: "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    alias:        str = ""


class AdvancedSQLBuilder:
    """
    Génère les constructions SQL avancées :
    - CTEs (WITH … AS)
    - Window functions (ROW_NUMBER, RANK, SUM OVER, LAG, LEAD …)
    - HAVING + sous-requêtes corrélées
    Multi-dialectes : MSSQL / PostgreSQL / MySQL
    """

    def build_cte(
        self,
        ctes: List[CTEDefinition],
        final_select: str,
        dialect: str = "mssql",
    ) -> str:
        """
        Construit une requête WITH … AS.
        ctes        = liste de CTEDefinition
        final_select = requête principale qui référence les CTEs
        """
        if not ctes:
            return final_select

        cte_parts = []
        for cte in ctes:
            cte_parts.append(f"{cte.name} AS (\n  {cte.query}\n)")

        return "WITH " + ",\n".join(cte_parts) + "\n" + final_select

    def build_window(
        self,
        base_select: str,
        table: str,
        windows: List[WindowSpec],
        dialect: str = "mssql",
    ) -> str:
        """
        Ajoute des colonnes calculées avec window functions à un SELECT.
        """
        window_cols = []
        for w in windows:
            over_parts: List[str] = []
            if w.partition_by:
                over_parts.append("PARTITION BY " + ", ".join(w.partition_by))
            if w.order_by:
                over_parts.append("ORDER BY " + ", ".join(w.order_by))
            if w.frame:
                over_parts.append(w.frame)

            over_clause = "OVER (" + " ".join(over_parts) + ")"
            alias = w.alias or f"{w.function.lower()}_col"
            window_cols.append(f"{w.function} {over_clause} AS {alias}")

        extra_cols = ",\n       ".join(window_cols)

        # Injecte les colonnes window dans le SELECT existant
        # On remplace "SELECT " par "SELECT <window_cols>, "
        if re.search(r"\bSELECT\b", base_select, re.IGNORECASE):
            return re.sub(
                r"\bSELECT\b",
                f"SELECT {extra_cols},\n       ",
                base_select,
                count=1,
                flags=re.IGNORECASE,
            )
        return base_select

    def build_having(
        self,
        agg_sql: str,
        having_conditions: List[str],
    ) -> str:
        """
        Ajoute une clause HAVING à une requête GROUP BY existante.
        Si ORDER BY existe, insère HAVING avant.
        """
        if not having_conditions:
            return agg_sql

        having_clause = "HAVING " + " AND ".join(having_conditions)

        if re.search(r"\bORDER\s+BY\b", agg_sql, re.IGNORECASE):
            return re.sub(
                r"(\bORDER\s+BY\b)",
                having_clause + "\n\\1",
                agg_sql,
                count=1,
                flags=re.IGNORECASE,
            )
        return agg_sql + f"\n{having_clause}"

    def detect_complexity(self, slots: QuerySlots) -> str:
        """
        Détermine si on a besoin de SQL avancé.
        Retourne : 'simple' | 'cte' | 'window' | 'having' | 'advanced'
        """
        text = " ".join([
            slots.raw_text if hasattr(slots, "raw_text") and slots.raw_text else "",
            " ".join(slots.table_names),
            slots.metric or "",
            slots.group_by or "",
        ]).lower()

        # Patterns qui déclenchent des constructions avancées
        cte_keywords     = ["étape", "step", "puis", "d'abord", "ensuite", "avec", "sous-total"]
        window_keywords  = ["rang", "rank", "classement", "numéro de ligne", "row number",
                            "running", "cumulatif", "cumul", "précédent", "precedent",
                            "lag", "lead", "glissant", "rolling", "ntile", "percentile",
                            "mois précédent", "mois precedent", "top n par", "top 3 par",
                            "top 5 par", "top 10 par", "par rapport au mois"]
        having_keywords  = ["ayant", "having", "au moins", "au plus", "plus de",
                            "supérieur à", "inférieur à", "dont le total", "where total"]

        has_window  = any(kw in text for kw in window_keywords)
        has_having  = any(kw in text for kw in having_keywords)
        has_cte     = any(kw in text for kw in cte_keywords)

        if has_window and (has_having or has_cte):
            return "advanced"
        if has_window:
            return "window"
        if has_having:
            return "having"
        if has_cte:
            return "cte"
        return "simple"

    # ── Templates prêts à l'emploi ────────────────────────────────────

    def template_top_n_per_group(
        self,
        table: str,
        group_col: str,
        rank_col: str,
        n: int = 3,
        dialect: str = "mssql",
        extra_cols: Optional[List[str]] = None,
    ) -> str:
        """
        TOP N par groupe avec ROW_NUMBER().
        Ex: "Top 3 commandes par client"
        """
        q = self._quote
        cols = ", ".join([q(c, dialect) for c in (extra_cols or [])]) + (", " if extra_cols else "")
        cte = CTEDefinition(
            name="ranked",
            query=(
                f"SELECT {q(group_col, dialect)}, {q(rank_col, dialect)},\n"
                f"       {cols}"
                f"       ROW_NUMBER() OVER (\n"
                f"         PARTITION BY {q(group_col, dialect)}\n"
                f"         ORDER BY {q(rank_col, dialect)} DESC\n"
                f"       ) AS rn\n"
                f"FROM {q(table, dialect)}"
            ),
        )
        final = (
            f"SELECT *\n"
            f"FROM ranked\n"
            f"WHERE rn <= {n}\n"
            f"ORDER BY {q(group_col, dialect)}, rn"
        )
        return self.build_cte([cte], final, dialect)

    def template_running_total(
        self,
        table: str,
        amount_col: str,
        date_col: str,
        partition_col: Optional[str] = None,
        dialect: str = "mssql",
    ) -> str:
        """
        Total cumulatif avec SUM() OVER (ORDER BY date).
        """
        q = self._quote
        partition = f"PARTITION BY {q(partition_col, dialect)} " if partition_col else ""
        return (
            f"SELECT {q(date_col, dialect)},\n"
            f"       {q(amount_col, dialect)},\n"
            f"       SUM({q(amount_col, dialect)}) OVER (\n"
            f"         {partition}ORDER BY {q(date_col, dialect)}\n"
            f"         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
            f"       ) AS running_total\n"
            f"FROM {q(table, dialect)}\n"
            f"ORDER BY {q(date_col, dialect)}"
        )

    def template_lag_comparison(
        self,
        table: str,
        amount_col: str,
        date_col: str,
        partition_col: Optional[str] = None,
        dialect: str = "mssql",
    ) -> str:
        """
        Comparaison période N vs période N-1 avec LAG().
        """
        q = self._quote
        partition = f"PARTITION BY {q(partition_col, dialect)} " if partition_col else ""
        return (
            f"SELECT {q(date_col, dialect)},\n"
            f"       {q(amount_col, dialect)} AS current_value,\n"
            f"       LAG({q(amount_col, dialect)}) OVER (\n"
            f"         {partition}ORDER BY {q(date_col, dialect)}\n"
            f"       ) AS previous_value,\n"
            f"       {q(amount_col, dialect)} - LAG({q(amount_col, dialect)}) OVER (\n"
            f"         {partition}ORDER BY {q(date_col, dialect)}\n"
            f"       ) AS delta\n"
            f"FROM {q(table, dialect)}\n"
            f"ORDER BY {q(date_col, dialect)}"
        )

    def template_rank_by_group(
        self,
        table: str,
        group_col: str,
        rank_col: str,
        dialect: str = "mssql",
    ) -> str:
        """
        RANK() et DENSE_RANK() par groupe.
        """
        q = self._quote
        return (
            f"SELECT {q(group_col, dialect)},\n"
            f"       {q(rank_col, dialect)},\n"
            f"       RANK() OVER (\n"
            f"         ORDER BY {q(rank_col, dialect)} DESC\n"
            f"       ) AS rank_val,\n"
            f"       DENSE_RANK() OVER (\n"
            f"         ORDER BY {q(rank_col, dialect)} DESC\n"
            f"       ) AS dense_rank_val\n"
            f"FROM {q(table, dialect)}\n"
            f"ORDER BY rank_val"
        )

    def template_agg_having(
        self,
        table: str,
        group_col: str,
        agg_col: str,
        agg_func: str = "SUM",
        having_op: str = ">",
        having_value: Any = 0,
        dialect: str = "mssql",
        top_n: int = 100,
    ) -> str:
        """
        GROUP BY avec HAVING.
        Ex: "clients ayant commandé plus de 10 fois"
        """
        q = self._quote
        limit = f"TOP {top_n} " if dialect in ("mssql", "sql_server") else ""
        agg_expr = "*" if agg_col == "*" else q(agg_col, dialect)
        sql = (
            f"SELECT {limit}{q(group_col, dialect)},\n"
            f"       {agg_func}({agg_expr}) AS {agg_func.lower()}_total\n"
            f"FROM {q(table, dialect)}\n"
            f"GROUP BY {q(group_col, dialect)}\n"
            f"HAVING {agg_func}({agg_expr}) {having_op} {having_value}\n"
            f"ORDER BY {agg_func.lower()}_total DESC"
        )
        if dialect not in ("mssql", "sql_server"):
            sql += f"\nLIMIT {top_n}"
        return sql

    @staticmethod
    def _quote(name: str, dialect: str) -> str:
        if dialect in ("mssql", "sql_server"):
            return f"[{name}]"
        if dialect == "mysql":
            return f"`{name}`"
        return f'"{name}"'


# ══════════════════════════════════════════════════════════════════════
# §2.3.3.A  SQL GENERATOR  (existant + intégration avancé)
# ══════════════════════════════════════════════════════════════════════

# ── Blacklist tables infrastructure ──────────────────────────────────────────
_INFRA_PREFIXES = (
    "QRTZ_", "qrtz_", "sys", "SYS", "dt_", "DT_",
    "MSreplication", "msreplication", "sysdiagram", "__",
)

def _is_infra_table(name: str) -> bool:
    return any(name.startswith(p) for p in _INFRA_PREFIXES)

def _first_business_table(schema: dict) -> str:
    """Retourne la premiere table metier du schema (exclut infra)."""
    for t in schema.keys():
        if not _is_infra_table(t):
            return t
    return "table"


class SQLGenerator:
    """
    Génère du SQL depuis les slots NLU.
    Approche hybride : templates pour les cas standards, AdvancedSQLBuilder pour les complexes.
    Validation automatique sur chaque requête générée.
    """

    MAX_ROWS = 1000

    def __init__(self):
        self._adv     = AdvancedSQLBuilder()
        self._val     = SQLValidator()

    def generate(
        self,
        slots: QuerySlots,
        schema_context: Dict,   # {table: [fields]}
        dialect: str = "mssql",
    ) -> Dict:
        """
        Génère une requête SQL depuis les slots.
        Retourne {sql, explanation, params, warnings, validation, complexity}
        """
        intent = slots.intent

        # Détecte si on a besoin de SQL avancé
        complexity = self._adv.detect_complexity(slots)

        try:
            if complexity in ("window", "having", "cte", "advanced"):
                result = self._generate_advanced(slots, schema_context, dialect, complexity)
            elif intent == Intent.GENERATE_AGG:
                result = self._generate_aggregate(slots, schema_context, dialect)
            elif intent == Intent.GENERATE_JOIN:
                result = self._generate_join(slots, schema_context, dialect)
            elif intent == Intent.GENERATE_FILTER:
                result = self._generate_filter(slots, schema_context, dialect)
            elif intent in (Intent.GENERATE_SQL, Intent.LIST_ENTITIES):
                result = self._generate_select(slots, schema_context, dialect)
            elif intent == Intent.COUNT_ENTITIES:
                result = self._generate_count(slots, schema_context, dialect)
            elif intent == Intent.PROFILE_ENTITY:
                result = self._generate_profile(slots, schema_context, dialect)
            else:
                result = self._generate_select(slots, schema_context, dialect)
        except Exception as e:
            logger.warning(f"[SQLGen] Error: {e}")
            result = {
                "sql":         self._fallback_sql(slots.table_names, dialect),
                "explanation": f"Requête générique (erreur: {e})",
                "params":      {},
                "warnings":    [str(e)],
            }

        # ── LLM pour requêtes complexes ───────────────────────────────────
        # Déclenché dès que is_complex_query=True, peu importe le template
        raw_text = getattr(slots, "raw_text", "") or ""
        try:
            from .llm_engine import is_complex_query, generate_sql_with_llm
            _need_llm = is_complex_query(raw_text)
            if _need_llm:
                logger.info(f"[SQLGen] Requête complexe → LLM tables={slots.table_names}")
                # Filtre le schéma aux tables mentionnées + tables connexes
                _tbl = slots.table_names or list(schema_context.keys())[:3]
                _filtered_schema = {t: schema_context[t] for t in _tbl if t in schema_context}
                if not _filtered_schema:
                    _filtered_schema = dict(list(schema_context.items())[:5])
                llm_result = generate_sql_with_llm(
                    question    = raw_text,
                    schema      = _filtered_schema,
                    dialect     = dialect,
                    table_names = _tbl,
                )
                if llm_result.get("sql"):
                    result = llm_result
                    result["complexity"] = "llm"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[SQLGen] LLM fallback error: {e}")

        # Optimisation SQL (WITH NOLOCK + index hints + explain plan)
        try:
            opt = get_sql_optimizer()
            opt_result = opt.optimize(result["sql"], schema_context, dialect)
            result["sql"]          = opt_result["sql"]
            result["explain_plan"] = opt_result["explain_plan"]
            result["index_hints"]  = opt_result["index_hints"]
            if opt_result["warnings"]:
                result.setdefault("warnings", []).extend(opt_result["warnings"])
        except Exception as _e:
            logger.debug(f"[SQLOptimizer] skip: {_e}")

        # Validation automatique
        vr = self._val.validate(result["sql"], dialect)
        result["validation"]  = {
            "valid":    vr.valid,
            "errors":   vr.errors,
            "warnings": vr.warnings + result.get("warnings", []),
            "score":    vr.score,
        }
        result["complexity"] = complexity
        return result

    # ── Génération avancée (CTE / Window / HAVING) ────────────────────

    def _generate_advanced(
        self,
        slots: QuerySlots,
        schema: Dict,
        dialect: str,
        complexity: str,
    ) -> Dict:
        table  = slots.table_names[0] if slots.table_names else _first_business_table(schema)
        fields = schema.get(table, [])

        # Résout le group_by : cherche le vrai champ dans le schéma
        # "client" → CustomerID, "employe" → EmployeeID, "categorie" → CategoryID
        _group_synonyms = {
            "client":       ["customerid","customer_id","companyname","custid"],
            "customer":     ["customerid","customer_id","companyname"],
            "employe":      ["employeeid","employee_id","lastname"],
            "employee":     ["employeeid","employee_id","lastname"],
            "produit":      ["productid","product_id","productname"],
            "product":      ["productid","product_id","productname"],
            "categorie":    ["categoryid","category_id","categoryname"],
            "category":     ["categoryid","category_id","categoryname"],
            "fournisseur":  ["supplierid","supplier_id","companyname"],
            "supplier":     ["supplierid","supplier_id"],
            "pays":         ["shipcountry","country","countrycode"],
            "country":      ["shipcountry","country","countrycode"],
            "region":       ["shipregion","region"],
            "ville":        ["shipcity","city"],
            "city":         ["shipcity","city"],
            "date":         ["orderdate","trndate","date","creationdate"],
            "mois":         ["orderdate","trndate","date"],
            "annee":        ["orderdate","trndate","date"],
            "banque":       ["banque","bank","acc_id"],
            "societe":      ["societe","company","companyname"],
            # ES — Español
            "cliente":      ["customerid","customer_id","companyname"],
            "empleado":     ["employeeid","employee_id","lastname"],
            "producto":     ["productid","product_id","productname"],
            "categoria":    ["categoryid","category_id","categoryname"],
            "proveedor":    ["supplierid","supplier_id"],
            "pais":         ["shipcountry","country","countrycode"],
            "ciudad":       ["shipcity","city"],
            "fecha":        ["orderdate","trndate","date"],
            "monto":        ["amounti","freight","unitprice","amount"],
            "importe":      ["amounti","freight","unitprice","amount"],
            "ventas":       ["freight","unitprice","amounti"],
            # DE — Deutsch
            "kunde":        ["customerid","customer_id","companyname"],
            "mitarbeiter":  ["employeeid","employee_id","lastname"],
            "produkt":      ["productid","product_id","productname"],
            "kategorie":    ["categoryid","category_id","categoryname"],
            "lieferant":    ["supplierid","supplier_id"],
            "land":         ["shipcountry","country","countrycode"],
            "stadt":        ["shipcity","city"],
            "datum":        ["orderdate","trndate","date"],
            "betrag":       ["amounti","freight","unitprice","amount"],
            "umsatz":       ["freight","unitprice","amounti"],
        }
        import unicodedata as _ud
        def _gnorm(s):
            s = ''.join(c for c in _ud.normalize('NFD',s) if _ud.category(c)!='Mn')
            # Dépluralise ES/DE
            if s.endswith('es'): s=s[:-2]
            elif s.endswith('en'): s=s[:-2]
            elif s.endswith('s') and len(s)>3: s=s[:-1]
            return s
        raw_group = _gnorm((slots.group_by or "").lower())
        resolved_group = None
        if raw_group:
            # Cherche d'abord dans les synonymes
            for syn_key, syn_fields in _group_synonyms.items():
                if syn_key in raw_group or raw_group in syn_key:
                    for sf in syn_fields:
                        matched = next((f for f in fields if f.lower() == sf), None)
                        if matched:
                            resolved_group = matched
                            break
                    if resolved_group:
                        break
            # Sinon cherche directement dans le schéma
            if not resolved_group:
                resolved_group = self._match_field(slots.group_by, fields)
        group_col = resolved_group or slots.group_by or (fields[0] if fields else "id")
        amount_col = self._find_numeric_field(fields, slots) or (fields[1] if len(fields) > 1 else "value")
        date_col   = self._find_date_field(fields) or "date"

        raw = getattr(slots, "raw_text", "") or ""

        if complexity == "window":
            raw_low = raw.lower()
            # Détecte le type de window
            if any(kw in raw_low for kw in ["cumulatif", "cumul", "running", "glissant"]):
                sql = self._adv.template_running_total(table, amount_col, date_col, None, dialect)
                exp = f"Total cumulatif de '{amount_col}' dans '{table}' par date"
            elif any(kw in raw_low for kw in ["précédent", "precedent", "lag", "comparaison",
                                               "vs", "mois précédent", "mois precedent",
                                               "par rapport"]):
                sql = self._adv.template_lag_comparison(table, amount_col, date_col, None, dialect)
                exp = f"Comparaison période N vs N-1 sur '{amount_col}'"
            else:
                # ROW_NUMBER / RANK — group_col = champ résolu depuis group_by
                top_n = slots.top_n or 3
                # Priorité : group_col résolu > champ id/name > premier champ
                if group_col and group_col in fields:
                    id_field = group_col
                else:
                    id_field = next(
                        (f for f in fields if any(k in f.lower() for k in
                         ["name","nom","label","banque","societe","customer","employee","product","category","supplier","code"])),
                        group_col or (fields[0] if fields else "id")
                    )
                # Si metric=SUM → utiliser template AGG simple avec ORDER BY montant
                if slots.metric == "SUM":
                    q = self._quote
                    limit_clause = f"TOP {top_n} " if dialect in ("mssql","sql_server") else ""
                    sql = (
                        f"SELECT {limit_clause}{q(id_field, dialect)},\n"
                        f"       SUM({q(amount_col, dialect)}) AS sum_total\n"
                        f"FROM {q(table, dialect)}\n"
                        f"GROUP BY {q(id_field, dialect)}\n"
                        f"ORDER BY sum_total DESC"
                    )
                    exp = f"Top {top_n} '{id_field}' par SUM({amount_col})"
                else:
                    sql = self._adv.template_top_n_per_group(table, id_field, amount_col, top_n, dialect, [])
                    exp = f"Top {top_n} '{amount_col}' par '{id_field}' avec ROW_NUMBER()"

        elif complexity == "having":
            raw_low = raw.lower()
            having_val = 0
            having_op  = ">"
            if slots.amount_filter:
                having_val = slots.amount_filter.get("value", 0)
                having_op  = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "="}.get(
                    slots.amount_filter.get("op", "gt"), ">"
                )
            # "fois", "commandes", "orders", "fois" → COUNT(*) plutôt que SUM
            count_kw = ["fois", "commandes", "orders", "times", "occurrences", "nombre de", "count", "transactions", "lignes", "entrees", "enregistrements"]
            if any(kw in raw_low for kw in count_kw):
                metric = "COUNT"
                agg_col = "*"
            else:
                metric = slots.metric or "SUM"
                agg_col = amount_col
            # group_col = champ résolu depuis group_by (prioritaire) sinon champ identifiant
            if group_col and group_col in fields:
                id_field = group_col
            else:
                id_field = next(
                    (f for f in fields if any(k in f.lower() for k in
                     ["name","nom","banque","societe","customer","employee","product","supplier","company","code"])),
                    group_col or (fields[0] if fields else "id")
                )
            sql = self._adv.template_agg_having(table, id_field, agg_col if metric != "COUNT" else "*", metric, having_op, having_val, dialect, slots.top_n or 100)
            exp = f"{metric}(*) GROUP BY [{id_field}] HAVING {having_op} {having_val}"

        elif complexity == "cte":
            # CTE : calcul intermédiaire + requête finale
            cte1 = CTEDefinition(
                name="base_agg",
                query=(
                    f"SELECT {self._quote(group_col, dialect)},\n"
                    f"       SUM({self._quote(amount_col, dialect)}) AS total\n"
                    f"FROM {self._quote(table, dialect)}\n"
                    f"GROUP BY {self._quote(group_col, dialect)}"
                ),
            )
            if dialect in ("mssql", "sql_server"):
                final = (
                    f"SELECT TOP {slots.top_n or 100} *\n"
                    f"FROM base_agg\n"
                    f"ORDER BY total DESC"
                )
            else:
                final = (
                    f"SELECT *\n"
                    f"FROM base_agg\n"
                    f"ORDER BY total DESC\n"
                    f"LIMIT {slots.top_n or 100}"
                )
            sql = self._adv.build_cte([cte1], final, dialect)
            exp = f"CTE aggregation de '{table}' — total par '{group_col}'"

        else:  # advanced : window + having combinés
            # CTE avec window function + HAVING
            cte1 = CTEDefinition(
                name="ranked_agg",
                query=(
                    f"SELECT {self._quote(group_col, dialect)},\n"
                    f"       SUM({self._quote(amount_col, dialect)}) AS total,\n"
                    f"       RANK() OVER (ORDER BY SUM({self._quote(amount_col, dialect)}) DESC) AS rnk\n"
                    f"FROM {self._quote(table, dialect)}\n"
                    f"GROUP BY {self._quote(group_col, dialect)}\n"
                    f"HAVING SUM({self._quote(amount_col, dialect)}) > 0"
                ),
            )
            final = (
                f"SELECT *\n"
                f"FROM ranked_agg\n"
                f"WHERE rnk <= {slots.top_n or 10}\n"
                f"ORDER BY rnk"
            )
            sql = self._adv.build_cte([cte1], final, dialect)
            exp = f"CTE + RANK() sur '{table}' — top {slots.top_n or 10} par total"

        return {
            "sql":         sql,
            "explanation": exp,
            "params":      {},
            "warnings":    [],
        }

    # ── Templates standards (inchangés + MSSQL TOP corrigé) ──────────

    def _generate_select(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        table = slots.table_names[0] if slots.table_names else _first_business_table(schema)
        fields = schema.get(table, [])

        select_cols = "*"
        if fields and len(fields) <= 20:
            select_cols = ", ".join(fields[:15])

        limit = slots.top_n or 100
        if dialect in ("mssql", "sql_server"):
            sql = f"SELECT TOP {limit} {select_cols}\nFROM {self._quote(table, dialect)}"
        else:
            sql = f"SELECT {select_cols}\nFROM {self._quote(table, dialect)}"

        where_clauses, params = self._build_where(slots, table, fields, dialect)
        if where_clauses:
            sql += f"\nWHERE {' AND '.join(where_clauses)}"

        if dialect not in ("mssql", "sql_server"):
            sql += f"\nLIMIT {min(limit, self.MAX_ROWS)}"

        return {
            "sql":         sql,
            "explanation": f"Sélection des données de '{table}'" + (f" — top {limit}" if slots.top_n else ""),
            "params":      params,
            "warnings":    self._check_warnings(slots, table),
        }

    def _generate_aggregate(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        table = slots.table_names[0] if slots.table_names else _first_business_table(schema)
        fields = schema.get(table, [])
        metric = slots.metric or "COUNT"
        group_by = slots.group_by

        numeric_field = self._find_numeric_field(fields, slots) if metric != "COUNT" else None
        agg_field = f"{metric}({numeric_field})" if numeric_field and metric != "COUNT" else f"{metric}(*)"

        top_n = slots.top_n or 20
        top_clause = f"TOP {top_n} " if dialect in ("mssql", "sql_server") else ""

        if group_by:
            group_field = self._match_field(group_by, fields) or group_by
            sql = (
                f"SELECT {top_clause}{self._quote(group_field, dialect)},\n"
                f"       {agg_field} AS {metric.lower()}_total\n"
                f"FROM {self._quote(table, dialect)}"
            )
            where_clauses, params = self._build_where(slots, table, fields, dialect)
            if where_clauses:
                sql += f"\nWHERE {' AND '.join(where_clauses)}"
            sql += f"\nGROUP BY {self._quote(group_field, dialect)}"
            sql += f"\nORDER BY {metric.lower()}_total DESC"
            if dialect not in ("mssql", "sql_server"):
                sql += f"\nLIMIT {top_n}"
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
        tables = slots.table_names
        if len(tables) < 2:
            return self._generate_select(slots, schema, dialect)

        t1, t2 = tables[0], tables[1]
        f1 = schema.get(t1, [])
        f2 = schema.get(t2, [])
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
        return self._generate_select(slots, schema, dialect)

    def _generate_count(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        table = slots.table_names[0] if slots.table_names else _first_business_table(schema)
        sql = f"SELECT COUNT(*) AS total\nFROM {self._quote(table, dialect)}"
        return {
            "sql": sql, "explanation": f"Nombre de lignes dans '{table}'",
            "params": {}, "warnings": [],
        }

    def _generate_profile(self, slots: QuerySlots, schema: Dict, dialect: str) -> Dict:
        table = slots.table_names[0] if slots.table_names else _first_business_table(schema)
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

    # ── Helpers (inchangés) ───────────────────────────────────────────

    def _build_where(self, slots: QuerySlots, table: str, fields: List[str], dialect: str) -> Tuple[List[str], Dict]:
        clauses: List[str] = []
        params:  Dict      = {}

        if slots.date_filter:
            df = slots.date_filter
            date_field = self._find_date_field(fields)
            if date_field:
                if df.get("type") == "range":
                    clauses.append(f"{self._quote(date_field, dialect)} BETWEEN '{df['from']}' AND '{df['to']}'")
                elif df.get("type") == "date":
                    clauses.append(f"{self._quote(date_field, dialect)} = '{df['from']}'")

        if slots.amount_filter:
            af = slots.amount_filter
            # Ne pas générer WHERE si op=eq et value correspond à top_n
            try:
                skip = (af.get("op") in ("eq","=") and slots.top_n and
                        abs(float(af.get("value", -1)) - float(slots.top_n)) < 0.001)
            except Exception:
                skip = False
            if not skip:
                amount_field = self._find_amount_field(fields)
                op_map = {"gt": ">", "lt": "<", "eq": "=", "gte": ">=", "lte": "<=", "between": "BETWEEN"}
                op = op_map.get(af.get("op", "eq"), "=")
                if amount_field:
                    clauses.append(f"{self._quote(amount_field, dialect)} {op} {af['value']}")

        return clauses, params

    def _find_date_field(self, fields: List[str]) -> Optional[str]:
        for kw in ["date", "at", "time", "created", "modified", "updated", "when"]:
            for f in fields:
                if kw in f.lower():
                    return f
        return None

    def _find_amount_field(self, fields: List[str]) -> Optional[str]:
        for kw in ["amount", "total", "price", "cost", "value", "montant", "prix", "sum"]:
            for f in fields:
                if kw in f.lower():
                    return f
        return None

    def _find_numeric_field(self, fields: List[str], slots: QuerySlots) -> Optional[str]:
        if slots.field_names:
            m = self._match_field(slots.field_names[0], fields)
            if m:
                return m
        if hasattr(slots, "raw_entities") and slots.raw_entities:
            for ent in slots.raw_entities:
                if ent.entity_type == "table":
                    m = self._match_field(ent.value, fields)
                    if m and (not slots.table_names or m != slots.table_names[0]):
                        return m
        numeric_kw = [
            "freight", "amount", "total", "price", "qty", "quantity", "value",
            "montant", "prix", "qte", "sum", "cost", "revenue", "shipping", "tax",
            "discount", "salary", "wage", "budget", "invoice", "payment", "fee",
            "charge", "rate", "weight", "poids", "unitprice", "extended",
        ]
        for kw in numeric_kw:
            for f in fields:
                if kw in f.lower():
                    return f
        id_kw = ["id", "code", "name", "country", "city", "address", "phone", "email", "desc"]
        for f in fields:
            if not any(kw in f.lower() for kw in id_kw):
                return f
        return fields[0] if fields else None

    def _find_join_key(self, t1: str, t2: str, f1: List[str], f2: List[str]) -> Optional[Tuple[str, str]]:
        for f in f2:
            if t1.lower().rstrip("s") in f.lower() and "id" in f.lower():
                for pk in f1:
                    if pk.lower() in ("id", f"{t1.lower()}_id", "pk"):
                        return (pk, f)
        for f in f1:
            if t2.lower().rstrip("s") in f.lower() and "id" in f.lower():
                for pk in f2:
                    if pk.lower() in ("id", f"{t2.lower()}_id", "pk"):
                        return (f, pk)
        common = set(fn.lower() for fn in f1) & set(fn.lower() for fn in f2)
        id_common = [c for c in common if "id" in c]
        if id_common:
            return (id_common[0], id_common[0])
        return None

    # Synonymes FR/EN pour les champs communs
    _FIELD_SYNONYMS = {
        "pays": ["country","shipcountry","countrycode","pays","nation"],
        "country": ["country","shipcountry","countrycode"],
        "region": ["region","shipregion","zone","area"],
        "ville": ["city","shipcity","ville","town"],
        "city": ["city","shipcity"],
        "date": ["date","orderdate","trndate","creationdate","shippeddate","invoicedate"],
        "montant": ["amount","freight","amounti","total","price","unitprice","montant"],
        "client": ["customerid","companyname","customername","customer"],
        "employe": ["employeeid","lastname","firstname","employee"],
        "produit": ["productid","productname","product"],
        "categorie": ["categoryid","categoryname","category"],
        "fournisseur": ["supplierid","companyname","supplier"],
        "banque": ["banque","bank","bankname","acc_id"],
        "societe": ["societe","company","companyname"],
        "code": ["code","ref","reference"],
        "mois": ["month","mois","orderdate","trndate"],
        "annee": ["year","annee","orderdate","trndate"],
        "semaine": ["week","semaine","orderdate"],
        # ES — Español
        "cliente":   ["customerid","companyname","customer"],
        "empleado":  ["employeeid","lastname","employee"],
        "producto":  ["productid","productname","product"],
        "categoria": ["categoryid","categoryname","category"],
        "pais":      ["shipcountry","country","countrycode"],
        "ciudad":    ["shipcity","city"],
        "fecha":     ["orderdate","trndate","date"],
        "monto":     ["amounti","freight","amount","unitprice"],
        "importe":   ["amounti","freight","amount"],
        "mes":       ["month","orderdate","trndate"],
        "anio":      ["year","orderdate","trndate"],
        # DE — Deutsch
        "kunde":        ["customerid","companyname","customer"],
        "mitarbeiter":  ["employeeid","lastname","employee"],
        "produkt":      ["productid","productname","product"],
        "kategorie":    ["categoryid","categoryname","category"],
        "land":         ["shipcountry","country","countrycode"],
        "stadt":        ["shipcity","city"],
        "datum":        ["orderdate","trndate","date"],
        "betrag":       ["amounti","freight","amount","unitprice"],
        "monat":        ["month","orderdate","trndate"],
        "jahr":         ["year","orderdate","trndate"],
    }

    def _match_field(self, name: str, fields: List[str]) -> Optional[str]:
        if not name:
            return None
        name_lower = name.lower()
        # Variantes à tester : singulier + sans accent
        import unicodedata
        def _norm(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s)
                          if unicodedata.category(c) != 'Mn')
        variants = {name_lower}
        # Pluriels ES: -es → , -s →
        if name_lower.endswith('es'): variants.add(name_lower[:-2])
        if name_lower.endswith('s'):  variants.add(name_lower[:-1])
        # Pluriels DE: -en → , -er →
        if name_lower.endswith('en'): variants.add(name_lower[:-2])
        if name_lower.endswith('er'): variants.add(name_lower[:-2])
        # Sans accents
        variants.add(_norm(name_lower))
        for variant in variants:
            # 1. Correspondance exacte
            for f in fields:
                if variant == f.lower():
                    return f
            # 2. Correspondance partielle
            for f in fields:
                if variant in f.lower() or f.lower() in variant:
                    return f
            # 3. Synonymes FR/EN/ES/DE
            synonyms = self._FIELD_SYNONYMS.get(variant, [])
            for syn in synonyms:
                for f in fields:
                    if syn == f.lower() or syn in f.lower():
                        return f
        return None

    def _quote(self, name: str, dialect: str) -> str:
        if dialect in ("mssql", "sql_server"):
            return f"[{name}]"
        if dialect == "mysql":
            return f"`{name}`"
        return f'"{name}"'

    def _limit(self, n: int, dialect: str) -> str:
        if dialect in ("mssql", "sql_server"):
            return ""
        return f"LIMIT {min(n, self.MAX_ROWS)}"

    def _fallback_sql(self, tables: List[str], dialect: str) -> str:
        table = tables[0] if tables else "your_table"
        if dialect in ("mssql", "sql_server"):
            return f"SELECT TOP 100 *\nFROM {self._quote(table, dialect)}"
        return f"SELECT *\nFROM {self._quote(table, dialect)}\nLIMIT 100"

    def _check_warnings(self, slots: QuerySlots, table: str) -> List[str]:
        warnings = []
        if slots.ambiguities:
            warnings.extend(slots.ambiguities)
        return warnings


# ══════════════════════════════════════════════════════════════════════
# §2.3.3.C  UNIVERSAL QUERY PLANNER  (inchangé + intégration validator)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QueryStep:
    step_id:     int
    source_id:   str
    source_type: str   # database, webservice, file
    action:      str   # sql, api_call, merge, filter, sort
    query:       str
    depends_on:  List[int] = field(default_factory=list)
    parallel:    bool = False
    cache_ttl:   int  = 300


@dataclass
class QueryPlan:
    question:       str
    steps:          List[QueryStep]
    merge_strategy: str   # join, union, enrich, none
    explanation:    str
    estimated_ms:   int = 0


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
        sql_gen = SQLGenerator()
        api_bld = APIQueryBuilder()
        steps:   List[QueryStep] = []
        step_id  = 0

        # ── Cas 1 : Source unique ─────────────────────────────────────
        if len(slots.table_names) <= 2 and len(sources) == 1:
            src     = sources[0]
            schema  = schemas.get(str(src["id"]), {})
            dialect = self._get_dialect(src["connector_type"])

            if src.get("category") == "webservice":
                fields = list(schema.values())[0] if schema else []
                api_r  = api_bld.build(slots, fields, src.get("base_url", ""))
                steps.append(QueryStep(
                    step_id=step_id, source_id=str(src["id"]),
                    source_type="webservice", action="api_call",
                    query=json.dumps(api_r), parallel=False, cache_ttl=120,
                ))
                explanation = api_r["explanation"]
            else:
                result = sql_gen.generate(slots, schema, dialect)
                steps.append(QueryStep(
                    step_id=step_id, source_id=str(src["id"]),
                    source_type=src.get("category", "database"), action="sql",
                    query=result["sql"], parallel=False, cache_ttl=60,
                ))
                explanation = result["explanation"]

            return QueryPlan(
                question="", steps=steps,
                merge_strategy="none", explanation=explanation,
            )

        # ── Cas 2 : Cross-source ──────────────────────────────────────
        db_sources  = [s for s in sources if s.get("category") == "database"]
        api_sources = [s for s in sources if s.get("category") == "webservice"]

        for src in db_sources[:2]:
            schema  = schemas.get(str(src["id"]), {})
            dialect = self._get_dialect(src["connector_type"])
            result  = sql_gen.generate(slots, schema, dialect)
            steps.append(QueryStep(
                step_id=step_id, source_id=str(src["id"]),
                source_type="database", action="sql",
                query=result["sql"], parallel=True, cache_ttl=300,
            ))
            step_id += 1

        for src in api_sources[:1]:
            fields = list(schemas.get(str(src["id"]), {}).values())
            fields = fields[0] if fields else []
            api_r  = api_bld.build(slots, fields, src.get("base_url", ""))
            steps.append(QueryStep(
                step_id=step_id, source_id=str(src["id"]),
                source_type="webservice", action="api_call",
                query=json.dumps(api_r),
                depends_on=[s.step_id for s in steps if s.source_type == "database"],
                parallel=False, cache_ttl=120,
            ))
            step_id += 1

        if len(steps) > 1:
            steps.append(QueryStep(
                step_id=step_id, source_id="merger",
                source_type="internal", action="merge",
                query=json.dumps({
                    "strategy": "join",
                    "on":       self._find_merge_key(slots, schemas),
                }),
                depends_on=[s.step_id for s in steps],
                parallel=False, cache_ttl=0,
            ))

        return QueryPlan(
            question="", steps=steps,
            merge_strategy="join" if len(steps) > 1 else "none",
            explanation=f"Plan cross-source : {len(steps)} étapes",
            estimated_ms=len(steps) * 200,
        )

    def _get_dialect(self, connector_type: str) -> str:
        if "mssql" in connector_type or "sage_100" in connector_type:
            return "mssql"
        if "mysql" in connector_type:
            return "mysql"
        return "postgresql"

    def _build_api_query(self, slots: QuerySlots, source: Dict) -> str:
        base = source.get("base_url", "")
        if slots.table_names:
            return f"{base}/{slots.table_names[0].lower()}"
        return base

    def _find_merge_key(self, slots: QuerySlots, schemas: Dict) -> str:
        common_keys = ["id", "customer_id", "order_id", "product_id"]
        for key in common_keys:
            for schema in schemas.values():
                for fields in schema.values():
                    if key in [f.lower() for f in fields]:
                        return key
        return "id"