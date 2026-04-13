"""
OnePilot — Semantic Enricher v1.0
§2.2.3 Indexation sémantique avancée

Fonctionnalités :
✅ A. Classification domaines métier (Finance, RH, Ventes, Logistique, Production)
✅ A. Extraction concepts métier (Customer, Order, Invoice, Product, Employee)
✅ A. Classification référentiel vs transactionnel
✅ A. Synonymes FR/EN/ES/DE
✅ A. Stemming / lemmatisation (nltk)
✅ B. Détection dimensions analytiques (temps, géo, produit, montant)
✅ C. Indexation MeiliSearch (full-text search)
✅ C. Embeddings TF-IDF + pgvector (recherche sémantique)
✅ C. Hybrid search (keyword + semantic)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# A. DICTIONNAIRES MÉTIER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Synonymes multi-langue ────────────────────────────────────────────────────
SYNONYMS: Dict[str, List[str]] = {
    # Clients / Partenaires
    "customer":   ["client", "partner", "debtor", "buyer", "customer_account",
                   "kunden", "cliente", "cli", "cust", "bpcustomer"],
    "supplier":   ["vendor", "fournisseur", "creditor", "seller", "proveedor",
                   "lieferant", "vendor_account", "bpvendor"],
    "employee":   ["employe", "staff", "worker", "personnel", "collaborateur",
                   "mitarbeiter", "empleado", "emp", "usr", "user"],
    "contact":    ["person", "personne", "individu", "interlocuteur"],

    # Commandes / Ventes
    "order":      ["commande", "sales_order", "purchase_order", "so", "po",
                   "sorder", "command", "ordre", "bestellung", "pedido",
                   "oohead", "vbak", "salesorder"],
    "order_line": ["order_detail", "order_item", "ligne_commande", "sorderline",
                   "vbap", "oolines", "sales_line"],
    "quote":      ["devis", "quotation", "offre", "angebot", "cotizacion"],
    "contract":   ["contrat", "agreement", "vertrag", "contrato"],

    # Facturation / Finance
    "invoice":    ["facture", "bill", "ar_invoice", "sales_invoice", "billing",
                   "rechnung", "factura", "fatura", "inv", "soinvoice",
                   "ra_customer"],
    "payment":    ["paiement", "reglement", "remittance", "zahlung", "pago"],
    "amount":     ["montant", "total", "sum", "value", "prix", "price",
                   "betrag", "importe", "amt", "ttc", "ht"],
    "account":    ["compte", "gl_account", "ledger", "konto", "cuenta",
                   "gl_je_lines", "gaccentry"],
    "budget":     ["budget", "forecast", "prevision", "plan"],
    "transaction":["transaction", "mouvement", "transactionline"],

    # Produits / Stock
    "product":    ["article", "item", "produit", "goods", "ware", "produkt",
                   "producto", "itmmaster", "mitmas", "mara"],
    "category":   ["categorie", "famille", "family", "group", "warengruppe",
                   "categoria", "cat"],
    "stock":      ["inventory", "inventaire", "bestand", "existencias",
                   "warehouse", "entrepot"],
    "price":      ["tarif", "pricelist", "liste_prix", "preisliste"],

    # RH
    "department": ["departement", "service", "division", "abteilung",
                   "departamento", "dept"],
    "position":   ["poste", "job", "fonction", "stelle", "puesto"],
    "salary":     ["salaire", "wage", "remuneration", "gehalt", "salario"],
    "leave":      ["conge", "absence", "vacation", "urlaub", "vacacion"],

    # Logistique
    "shipment":   ["expedition", "livraison", "delivery", "shipping",
                   "versand", "envio", "dispatch"],
    "warehouse":  ["entrepot", "depot", "lager", "almacen", "wh"],
    "address":    ["adresse", "location", "adresa", "direccion", "adr"],

    # Technique
    "log":        ["audit", "history", "trace", "journal", "protokoll",
                   "historial", "revdat"],
    "config":     ["configuration", "setting", "parametre", "einstellung",
                   "parametro", "setup"],
    "status":     ["statut", "etat", "state", "zustand", "estado", "sts"],
    "code":       ["code", "ref", "reference", "nummer", "numero", "no", "num"],
}

# Index inverse : mot → concept
_SYNONYM_INDEX: Dict[str, str] = {}
for _concept, _words in SYNONYMS.items():
    _SYNONYM_INDEX[_concept] = _concept
    for _w in _words:
        _SYNONYM_INDEX[_w.lower()] = _concept


# ── Classification domaines métier ────────────────────────────────────────────
DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "Finance": [
        "invoice", "facture", "payment", "paiement", "account", "compte",
        "budget", "gl", "ledger", "tax", "tva", "fiscal", "bank", "banque",
        "bnk", "cash", "tresorerie", "credit", "debit", "ar", "ap",
        "revenue", "chiffre", "profit", "loss", "balance", "transaction",
        "gaccentry", "gl_je", "ra_customer", "ap_invoice",
    ],
    "Ventes": [
        "order", "commande", "sales", "vente", "customer", "client",
        "invoice", "quote", "devis", "contract", "contrat", "so", "crm",
        "opportunity", "lead", "pipeline", "revenue", "discount", "remise",
        "vbak", "vbap", "sorder", "oohead",
    ],
    "Achats": [
        "purchase", "achat", "supplier", "fournisseur", "vendor", "po",
        "procurement", "rfq", "appel_offre", "bon_commande",
        "ap_invoice", "bpvendor",
    ],
    "Logistique": [
        "shipment", "expedition", "delivery", "livraison", "warehouse",
        "entrepot", "stock", "inventory", "transport", "carrier",
        "tracking", "route", "depot", "wh", "dispatch",
    ],
    "Production": [
        "production", "manufacturing", "bom", "nomenclature", "routing",
        "gamme", "work_order", "ordre_fabrication", "machine", "capacity",
        "workstation", "poste_charge", "assembly",
    ],
    "RH": [
        "employee", "employe", "staff", "personnel", "payroll", "paie",
        "salary", "salaire", "leave", "conge", "department", "position",
        "poste", "contract_rh", "attendance", "pointage", "hr",
    ],
    "Referentiel": [
        "config", "configuration", "setting", "parametre", "reference",
        "code_table", "lookup", "master", "referentiel", "setup",
        "country", "currency", "unit", "category",
    ],
    "Technique": [
        "log", "audit", "history", "trace", "session", "token", "auth",
        "migration", "backup", "monitor", "alert", "notification",
        "revdat", "btchtsklog", "usrnotif",
    ],
}

# ── Concepts métier principaux ────────────────────────────────────────────────
CONCEPT_PATTERNS: Dict[str, List[str]] = {
    "Customer":    ["customer", "client", "partner", "debtor", "bpcustomer"],
    "Supplier":    ["supplier", "vendor", "fournisseur", "creditor", "bpvendor"],
    "Employee":    ["employee", "employe", "staff", "worker", "usr"],
    "Order":       ["order", "commande", "sorder", "oohead", "vbak", "salesorder"],
    "OrderLine":   ["order_line", "orderline", "sorderline", "vbap", "oolines"],
    "Invoice":     ["invoice", "facture", "bill", "soinvoice", "ra_customer"],
    "Product":     ["product", "article", "item", "itmmaster", "mitmas", "mara"],
    "Category":    ["category", "categorie", "famille", "family"],
    "Stock":       ["stock", "inventory", "inventaire", "warehouse"],
    "Payment":     ["payment", "paiement", "reglement"],
    "Account":     ["account", "compte", "ledger", "gl"],
    "Department":  ["department", "departement", "service", "division"],
    "Address":     ["address", "adresse", "location"],
    "Contact":     ["contact", "person", "personne"],
    "Log":         ["log", "audit", "history", "trace", "revdat"],
    "Config":      ["config", "configuration", "setting", "parametre"],
}

# ── Classification référentiel vs transactionnel ──────────────────────────────
TRANSACTIONAL_PATTERNS = [
    "order", "commande", "invoice", "facture", "payment", "paiement",
    "shipment", "transaction", "movement", "mouvement", "log", "audit",
    "history", "journal", "entry", "ecriture", "notif", "btch",
]
REFERENCE_PATTERNS = [
    "config", "setting", "parametre", "country", "pays", "currency",
    "devise", "unit", "unite", "category", "categorie", "type", "status",
    "code", "reference", "master", "lookup", "table_ref",
]


# ═══════════════════════════════════════════════════════════════════════════════
# B. DIMENSIONS ANALYTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSION_PATTERNS: Dict[str, Dict] = {
    "time": {
        "patterns": [
            r"\bdate\b", r"\btime\b", r"\btimestamp\b", r"\bat\b$",
            r"_date$", r"_time$", r"_at$", r"_dt$", r"_dts$",
            r"^date_", r"^time_", r"created", r"updated", r"modified",
            r"posted", r"ordered", r"shipped", r"invoiced", r"due",
            r"start", r"end", r"begin", r"expire", r"fiscal",
        ],
        "hierarchy": ["year", "quarter", "month", "week", "day"],
        "label": "Dimension Temps",
    },
    "geo": {
        "patterns": [
            r"\bcountry\b", r"\bpays\b", r"\bregion\b", r"\bcity\b",
            r"\bville\b", r"\bstate\b", r"\bpostal\b", r"\bzip\b",
            r"\baddress\b", r"\badresse\b", r"\blatitude\b", r"\blongitude\b",
            r"_country$", r"_city$", r"_region$", r"_state$",
            r"^country_", r"^city_", r"iso_", r"\bgeo\b",
        ],
        "hierarchy": ["country", "region", "city"],
        "label": "Dimension Géographique",
    },
    "product": {
        "patterns": [
            r"\bproduct\b", r"\barticle\b", r"\bitem\b", r"\bsku\b",
            r"\bbrand\b", r"\bmarque\b", r"\bcategory\b", r"\bcategorie\b",
            r"\bfamily\b", r"\bfamille\b", r"product_id$", r"item_id$",
            r"article_id$", r"_product$", r"_article$",
        ],
        "hierarchy": ["brand", "category", "subcategory", "product"],
        "label": "Dimension Produit",
    },
    "amount": {
        "patterns": [
            r"\bamount\b", r"\bmontant\b", r"\btotal\b", r"\bprice\b",
            r"\bprix\b", r"\bcost\b", r"\bcout\b", r"\bvalue\b",
            r"\bvaleur\b", r"\bsum\b", r"\bsomme\b", r"\bttc\b",
            r"\bht\b", r"\btax\b", r"\btva\b", r"_amount$", r"_price$",
            r"_cost$", r"_total$", r"^amt_", r"^mnt_",
        ],
        "hierarchy": None,
        "label": "Mesure Montant",
    },
    "quantity": {
        "patterns": [
            r"\bqty\b", r"\bquantity\b", r"\bquantite\b", r"\bcount\b",
            r"\bnb\b", r"\bnombre\b", r"\bvolume\b", r"_qty$", r"_count$",
            r"_nb$", r"^qty_", r"^nb_",
        ],
        "hierarchy": None,
        "label": "Mesure Quantité",
    },
    "identifier": {
        "patterns": [
            r"_id$", r"^id_", r"^id$", r"_code$", r"_key$", r"_ref$",
            r"_no$", r"_num$", r"_number$", r"_uuid$",
        ],
        "hierarchy": None,
        "label": "Identifiant",
    },
    "status": {
        "patterns": [
            r"\bstatus\b", r"\bstatut\b", r"\betat\b", r"\bstate\b",
            r"\bflag\b", r"\bactive\b", r"\benabled\b", r"_status$",
            r"_state$", r"_flag$", r"^is_", r"^has_",
        ],
        "hierarchy": None,
        "label": "Statut / Flag",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEMMING / LEMMATISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> List[str]:
    """
    Normalise un nom de table/colonne en tokens :
    - Supprime les préfixes ERP (AA_, TH_, GS_, CS_, etc.)
    - Split sur _ et CamelCase
    - Stemming léger (suppression suffixes courants)
    - Retourne la liste de tokens normalisés
    """
    # Nettoyage initial
    name = name.lower().strip()

    # Suppression préfixes ERP connus (AA_, TH_, GS_, CS_, BT_, etc.)
    name = re.sub(r'^[a-z]{1,3}_', '', name)

    # Split CamelCase
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name).lower()

    # Split sur séparateurs
    tokens = re.split(r'[_\-\s\.]+', name)
    tokens = [t for t in tokens if len(t) > 1]

    # Stemming léger
    stemmed = []
    for token in tokens:
        t = token
        # Pluriels anglais
        if t.endswith('ies') and len(t) > 4:
            t = t[:-3] + 'y'
        elif t.endswith('ses') and len(t) > 4:
            t = t[:-2]
        elif t.endswith('s') and len(t) > 3 and not t.endswith('ss'):
            t = t[:-1]
        # Suffixes FR
        if t.endswith('tion'):
            t = t[:-4]
        elif t.endswith('ment') and len(t) > 6:
            t = t[:-4]
        stemmed.append(t)

    return stemmed


def _resolve_synonyms(tokens: List[str]) -> List[str]:
    """Résout les synonymes FR/EN/ES/DE → concept canonique."""
    resolved = []
    for token in tokens:
        canonical = _SYNONYM_INDEX.get(token, token)
        resolved.append(canonical)
        if canonical != token:
            resolved.append(token)  # garder l'original aussi
    return list(dict.fromkeys(resolved))  # dédupliquer en gardant l'ordre


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_domain(entity_name: str, field_names: List[str] = None) -> str:
    """Classifie une entité dans un domaine métier (patterns statiques uniquement)."""
    tokens = _normalize_name(entity_name)
    resolved = _resolve_synonyms(tokens)
    all_tokens = set(resolved + tokens)

    # Ajouter tokens des champs
    if field_names:
        for fname in field_names[:20]:  # cap à 20 champs
            all_tokens.update(_normalize_name(fname))

    scores: Dict[str, int] = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0
        for token in all_tokens:
            for pattern in patterns:
                if token == pattern or token.startswith(pattern) or pattern in token:
                    score += 1
        if score > 0:
            scores[domain] = score

    if not scores:
        return "Autre"
    return max(scores, key=scores.get)


def classify_domain_with_custom(
    entity_name: str,
    field_names: List[str] = None,
    custom_domains: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Classifie une entité — combine domaines statiques + domaines personnalisés.
    Les domaines personnalisés ont la priorité sur les domaines statiques.
    custom_domains : {domain_name: [pattern1, pattern2, ...]}
    """
    tokens = _normalize_name(entity_name)
    resolved = _resolve_synonyms(tokens)
    all_tokens = set(resolved + tokens)

    if field_names:
        for fname in field_names[:20]:
            all_tokens.update(_normalize_name(fname))

    name_lower = entity_name.lower()

    # ── Domaines personnalisés en priorité ────────────────────
    if custom_domains:
        custom_scores: Dict[str, int] = {}
        for domain_name, patterns in custom_domains.items():
            score = 0
            for token in all_tokens:
                for pattern in patterns:
                    p = pattern.lower()
                    if token == p or p in token or p in name_lower:
                        score += 2  # boost priorité
            if score > 0:
                custom_scores[domain_name] = score
        if custom_scores:
            return max(custom_scores, key=custom_scores.get)

    # ── Domaines statiques fallback ───────────────────────────
    scores: Dict[str, int] = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0
        for token in all_tokens:
            for pattern in patterns:
                if token == pattern or token.startswith(pattern) or pattern in token:
                    score += 1
        if score > 0:
            scores[domain] = score

    if not scores:
        return "Autre"
    return max(scores, key=scores.get)


def classify_concept(entity_name: str) -> Optional[str]:
    """Identifie le concept métier principal d'une entité."""
    tokens = set(_normalize_name(entity_name))
    resolved = set(_resolve_synonyms(list(tokens)))
    all_tokens = tokens | resolved

    scores: Dict[str, int] = {}
    for concept, patterns in CONCEPT_PATTERNS.items():
        score = sum(
            1 for token in all_tokens
            for pattern in patterns
            if token == pattern or pattern in token
        )
        if score > 0:
            scores[concept] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def classify_entity_class(
    entity_name: str,
    row_count: Optional[int] = None,
) -> str:
    """
    Classifie une entité :
    - transactional : commandes, factures, mouvements (gros volume)
    - reference     : pays, devises, catégories (faible volume)
    - log           : historique, audit, trace
    - config        : paramètres, configuration
    """
    tokens = _normalize_name(entity_name)
    name_lower = entity_name.lower()

    # Log en priorité
    if any(p in name_lower for p in ["log", "audit", "history", "trace", "revdat", "btch"]):
        return "log"

    # Config
    if any(p in name_lower for p in REFERENCE_PATTERNS[:8]):
        return "config"

    # Transactionnel
    if any(p in name_lower for p in TRANSACTIONAL_PATTERNS):
        return "transactional"

    # Référentiel par volume
    if row_count is not None:
        if row_count < 10_000:
            return "reference"
        elif row_count > 100_000:
            return "transactional"

    # Référentiel par nom
    if any(p in name_lower for p in REFERENCE_PATTERNS):
        return "reference"

    return "transactional"


def detect_dimensions(field_names: List[str]) -> Dict[str, List[str]]:
    """
    Détecte les dimensions analytiques dans les colonnes d'une entité.
    Retourne un dict {dimension_type: [col1, col2, ...]}
    """
    result: Dict[str, List[str]] = {}

    for field_name in field_names:
        fname_lower = field_name.lower()
        for dim_type, dim_config in DIMENSION_PATTERNS.items():
            for pattern in dim_config["patterns"]:
                if re.search(pattern, fname_lower):
                    result.setdefault(dim_type, [])
                    if field_name not in result[dim_type]:
                        result[dim_type].append(field_name)
                    break

    return result


def detect_field_dimension(field_name: str, data_type: str = "") -> Optional[str]:
    """Détecte le type de dimension d'un champ individuel."""
    fname_lower = field_name.lower()
    dtype_lower = data_type.lower()

    # Booléen par type
    if dtype_lower in ("boolean", "bool", "bit"):
        return "status"

    for dim_type, dim_config in DIMENSION_PATTERNS.items():
        for pattern in dim_config["patterns"]:
            if re.search(pattern, fname_lower):
                return dim_type

    return None


def build_field_hierarchy(
    field_name: str,
    dim_type: str,
    data_type: str = "",
    dialect: str = "mssql",
    fiscal_year_start: int = 1,  # 1=Jan (standard), 7=Juil, 10=Oct...
) -> Optional[Dict[str, Any]]:
    """
    Génère les expressions SQL de hiérarchie pour un champ dimension.
    Supporte les calendriers fiscaux non standard (fiscal_year_start != 1).

    Retourne un dict prêt à stocker en JSONB :
    {
      "type": "time",
      "label": "Dimension Temps",
      "levels": [
        {"level": "year",    "expr": "YEAR({field})",              "label": "Année"},
        {"level": "quarter", "expr": "DATEPART(quarter, {field})", "label": "Trimestre"},
        ...
      ]
    }
    """
    if dim_type == "time":
        if dialect in ("mssql", "sqlserver"):
            levels = [
                {"level": "year",    "expr": f"YEAR({field_name})",                      "label": "Année"},
                {"level": "quarter", "expr": f"DATEPART(quarter, {field_name})",          "label": "Trimestre"},
                {"level": "month",   "expr": f"MONTH({field_name})",                      "label": "Mois"},
                {"level": "week",    "expr": f"DATEPART(week, {field_name})",             "label": "Semaine"},
                {"level": "day",     "expr": f"DAY({field_name})",                        "label": "Jour"},
                {"level": "label_month", "expr": f"FORMAT({field_name}, 'yyyy-MM')",      "label": "Mois (texte)"},
                {"level": "label_quarter", "expr": f"CONCAT('Q', DATEPART(quarter, {field_name}), '-', YEAR({field_name}))", "label": "Trimestre (texte)"},
            ]
            if fiscal_year_start != 1:
                levels += [
                    {"level": "fiscal_year", "expr": f"CASE WHEN MONTH({field_name}) >= {fiscal_year_start} THEN YEAR({field_name}) + 1 ELSE YEAR({field_name}) END", "label": f"Année fiscale (mois {fiscal_year_start})"},
                    {"level": "fiscal_quarter", "expr": f"((MONTH({field_name}) - {fiscal_year_start} + 12) % 12) / 3 + 1", "label": "Trimestre fiscal"},
                    {"level": "fiscal_month", "expr": f"((MONTH({field_name}) - {fiscal_year_start} + 12) % 12) + 1", "label": "Mois fiscal"},
                ]
        else:  # PostgreSQL
            levels = [
                {"level": "year",    "expr": f"EXTRACT(year FROM {field_name})::int",     "label": "Année"},
                {"level": "quarter", "expr": f"EXTRACT(quarter FROM {field_name})::int",  "label": "Trimestre"},
                {"level": "month",   "expr": f"EXTRACT(month FROM {field_name})::int",    "label": "Mois"},
                {"level": "week",    "expr": f"EXTRACT(week FROM {field_name})::int",     "label": "Semaine"},
                {"level": "day",     "expr": f"EXTRACT(day FROM {field_name})::int",      "label": "Jour"},
                {"level": "label_month",   "expr": f"TO_CHAR({field_name}, 'YYYY-MM')",   "label": "Mois (texte)"},
                {"level": "label_quarter", "expr": f"CONCAT('Q', EXTRACT(quarter FROM {field_name})::int, '-', EXTRACT(year FROM {field_name})::int)", "label": "Trimestre (texte)"},
            ]
            if fiscal_year_start != 1:
                levels += [
                    {"level": "fiscal_year", "expr": f"CASE WHEN EXTRACT(month FROM {field_name})::int >= {fiscal_year_start} THEN EXTRACT(year FROM {field_name})::int + 1 ELSE EXTRACT(year FROM {field_name})::int END", "label": f"Année fiscale (mois {fiscal_year_start})"},
                    {"level": "fiscal_quarter", "expr": f"((EXTRACT(month FROM {field_name})::int - {fiscal_year_start} + 12) % 12) / 3 + 1", "label": "Trimestre fiscal"},
                    {"level": "fiscal_month", "expr": f"((EXTRACT(month FROM {field_name})::int - {fiscal_year_start} + 12) % 12) + 1", "label": "Mois fiscal"},
                ]
        return {
            "type":              "time",
            "label":             "Dimension Temps",
            "field":             field_name,
            "fiscal_year_start": fiscal_year_start,
            "levels":            levels,
        }

    elif dim_type == "geo":
        # Pour les dimensions géo on retourne juste le champ lui-même
        # (les jointures entre tables géo sont gérées par le graph de relations)
        level_label = "Pays"
        if any(k in field_name.lower() for k in ["region", "state", "province"]):
            level_label = "Région"
        elif any(k in field_name.lower() for k in ["city", "ville", "town"]):
            level_label = "Ville"
        elif any(k in field_name.lower() for k in ["zip", "postal", "code_postal"]):
            level_label = "Code postal"

        return {
            "type":  "geo",
            "label": "Dimension Géographique",
            "field": field_name,
            "levels": [
                {"level": "value", "expr": field_name, "label": level_label},
            ],
        }

    elif dim_type == "product":
        return {
            "type":  "product",
            "label": "Dimension Produit",
            "field": field_name,
            "levels": [
                {"level": "value", "expr": field_name, "label": "Produit"},
            ],
        }

    elif dim_type == "amount":
        if dialect in ("mssql", "sqlserver"):
            return {
                "type":  "amount",
                "label": "Mesure Montant",
                "field": field_name,
                "levels": [
                    {"level": "sum",   "expr": f"SUM({field_name})",   "label": "Total"},
                    {"level": "avg",   "expr": f"AVG({field_name})",   "label": "Moyenne"},
                    {"level": "min",   "expr": f"MIN({field_name})",   "label": "Minimum"},
                    {"level": "max",   "expr": f"MAX({field_name})",   "label": "Maximum"},
                    {"level": "count", "expr": f"COUNT({field_name})", "label": "Nombre"},
                ],
            }
        else:
            return {
                "type":  "amount",
                "label": "Mesure Montant",
                "field": field_name,
                "levels": [
                    {"level": "sum",   "expr": f"SUM({field_name})",   "label": "Total"},
                    {"level": "avg",   "expr": f"AVG({field_name})",   "label": "Moyenne"},
                    {"level": "min",   "expr": f"MIN({field_name})",   "label": "Minimum"},
                    {"level": "max",   "expr": f"MAX({field_name})",   "label": "Maximum"},
                    {"level": "count", "expr": f"COUNT({field_name})", "label": "Nombre"},
                ],
            }

    elif dim_type == "quantity":
        return {
            "type":  "quantity",
            "label": "Mesure Quantité",
            "field": field_name,
            "levels": [
                {"level": "sum",   "expr": f"SUM({field_name})",   "label": "Total"},
                {"level": "avg",   "expr": f"AVG({field_name})",   "label": "Moyenne"},
                {"level": "count", "expr": f"COUNT({field_name})", "label": "Nombre"},
            ],
        }

    return None


def extract_semantic_tags(
    entity_name: str,
    domain: str,
    concept: Optional[str],
    entity_class: str,
    dimensions: Dict[str, List[str]],
) -> List[str]:
    """Génère les tags sémantiques d'une entité."""
    tags = []

    # Tags domaine
    tags.append(f"domain:{domain.lower()}")

    # Tags concept
    if concept:
        tags.append(f"concept:{concept.lower()}")

    # Tags classe
    tags.append(f"class:{entity_class}")

    # Tags dimensions
    for dim_type in dimensions:
        tags.append(f"dim:{dim_type}")

    # Tags synonymes résolus
    tokens = _normalize_name(entity_name)
    for token in _resolve_synonyms(tokens)[:5]:
        if len(token) > 2:
            tags.append(f"term:{token}")

    return list(dict.fromkeys(tags))  # dédupliquer


# ═══════════════════════════════════════════════════════════════════════════════
# C. EMBEDDINGS TF-IDF (version légère sans BERT)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_entity_text(
    entity_name: str,
    field_names: List[str],
    description: Optional[str],
    domain: str,
    concept: Optional[str],
) -> str:
    """Construit le texte représentatif d'une entité pour l'embedding."""
    parts = []

    # Nom normalisé + synonymes
    tokens = _normalize_name(entity_name)
    resolved = _resolve_synonyms(tokens)
    parts.extend(resolved)
    parts.append(entity_name.lower())

    # Description
    if description:
        parts.append(description.lower())

    # Domaine et concept
    parts.append(domain.lower())
    if concept:
        parts.append(concept.lower())
        # Synonymes du concept
        synonyms = SYNONYMS.get(concept.lower(), [])
        parts.extend(synonyms[:5])

    # Noms des champs importants
    for fname in field_names[:30]:
        parts.extend(_normalize_name(fname))

    return " ".join(parts)



# ═══════════════════════════════════════════════════════════════════════════════
# BERT MODEL — Singleton chargé une seule fois au démarrage
# ═══════════════════════════════════════════════════════════════════════════════

_BERT_MODEL = None
_BERT_MODEL_NAME = "all-MiniLM-L6-v2"
_BERT_DIM = 384  # dimension fixe du modèle all-MiniLM-L6-v2


def _get_bert_model():
    """
    Charge le modèle BERT une seule fois (singleton).
    Retourne None si sentence-transformers n'est pas installé.
    """
    global _BERT_MODEL
    if _BERT_MODEL is not None:
        return _BERT_MODEL

    try:
        import os
        cache_dir = "/tmp/bert_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

        from sentence_transformers import SentenceTransformer
        logger.info(f"[SemanticEnricher] Chargement BERT {_BERT_MODEL_NAME} (cache={cache_dir})...")
        _BERT_MODEL = SentenceTransformer(_BERT_MODEL_NAME, cache_folder=cache_dir)
        logger.info(f"[SemanticEnricher] ✅ BERT chargé — dim={_BERT_DIM}")
        return _BERT_MODEL
    except ImportError:
        logger.warning("[SemanticEnricher] sentence-transformers non installé — fallback TF-IDF")
        return None
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Erreur chargement BERT: {e} — fallback TF-IDF")
        return None


def compute_bert_embedding(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Calcule des embeddings BERT (all-MiniLM-L6-v2).
    Dimension fixe : 384.
    Traitement par batch pour éviter les OOM.
    """
    if not texts:
        return []

    model = _get_bert_model()
    if model is None:
        return compute_tfidf_embedding(texts)  # fallback

    try:
        import numpy as np
        all_embeddings = []

        # Traitement par batch
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,  # normalisation L2 intégrée
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())

        logger.info(f"[SemanticEnricher] BERT embeddings: {len(all_embeddings)} vecteurs dim={_BERT_DIM}")
        return all_embeddings

    except Exception as e:
        logger.warning(f"[SemanticEnricher] Erreur BERT embedding: {e} — fallback TF-IDF")
        return compute_tfidf_embedding(texts)


def compute_tfidf_embedding(texts: List[str], dim: int = 384) -> List[List[float]]:
    """
    Calcule des embeddings TF-IDF normalisés.
    Utilisé comme fallback si BERT n'est pas disponible.
    Retourne une liste de vecteurs de dimension `dim`.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        import numpy as np

        if not texts:
            return []

        # TF-IDF avec n-grams
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=min(5000, len(texts) * 100),
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Réduction dimensionnelle vers `dim` dimensions
        actual_dim = min(dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if actual_dim < 2:
            return [list(tfidf_matrix[i].toarray()[0][:dim]) + [0.0] * max(0, dim - tfidf_matrix.shape[1])
                    for i in range(tfidf_matrix.shape[0])]

        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)

        # Normalisation L2
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = reduced / norms

        # Padding si nécessaire
        if actual_dim < dim:
            padding = np.zeros((normalized.shape[0], dim - actual_dim))
            normalized = np.hstack([normalized, padding])

        return normalized.tolist()

    except ImportError:
        logger.warning("[SemanticEnricher] sklearn non disponible — embeddings désactivés")
        return [[] for _ in texts]
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Erreur TF-IDF embedding: {e}")
        return [[] for _ in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# C. MEILISEARCH — Indexation
# ═══════════════════════════════════════════════════════════════════════════════

MEILI_HOST = os.environ.get("MEILI_HOST", "http://onepilot_meilisearch:7700")
MEILI_KEY  = os.environ.get("MEILI_MASTER_KEY", "onepilot_meili_key")
MEILI_INDEX = "onepilot_entities"


def _get_meili_client():
    """Retourne le client MeiliSearch."""
    try:
        import meilisearch
        return meilisearch.Client(MEILI_HOST, MEILI_KEY)
    except ImportError:
        logger.warning("[SemanticEnricher] meilisearch-python non installé")
        return None
    except Exception as e:
        logger.warning(f"[SemanticEnricher] MeiliSearch non disponible: {e}")
        return None


def _ensure_meili_index(client) -> bool:
    """Crée et configure l'index MeiliSearch si nécessaire."""
    try:
        try:
            client.get_index(MEILI_INDEX)
        except Exception:
            task = client.create_index(MEILI_INDEX, {"primaryKey": "id"})
            client.wait_for_task(task.task_uid)

        # Configuration de l'index
        idx = client.index(MEILI_INDEX)
        idx.update_searchable_attributes([
            "name", "display_name", "description", "domain",
            "concept", "tags", "field_names", "synonyms_text",
            "sample_values",
        ])
        idx.update_filterable_attributes([
            "source_id", "domain", "concept", "entity_class",
            "source_type",
        ])
        idx.update_ranking_rules([
            "words", "typo", "proximity", "attribute",
            "sort", "exactness",
        ])
        idx.update_typo_tolerance({
            "enabled": True,
            "minWordSizeForTypos": {"oneTypo": 4, "twoTypos": 8},
        })

        # ── Synonymes multi-langue persistants ───────────────────────
        idx.update_synonyms({
            # FR → EN
            "facture":      ["invoice", "bill", "finance", "soinvoice", "ar"],
            "client":       ["customer", "finance", "account", "debtor"],
            "commande":     ["order", "sorder", "achat", "oohead", "vbak"],
            "fournisseur":  ["supplier", "vendor", "achat", "creditor"],
            "employe":      ["employee", "rh", "staff", "personnel"],
            "salaire":      ["salary", "payroll", "rh", "paie"],
            "produit":      ["product", "article", "item", "itmmaster"],
            "stock":        ["inventory", "warehouse", "entrepot"],
            "paiement":     ["payment", "reglement", "finance"],
            "compte":       ["account", "gl", "ledger", "finance"],
            "budget":       ["budget", "finance", "forecast"],
            "tresorerie":   ["cash", "finance", "bank", "bnk"],
            "banque":       ["bank", "finance", "bnk"],
            "vente":        ["sales", "order", "vbak"],
            "achat":        ["purchase", "order", "po", "fournisseur"],
            # EN → FR
            "invoice":      ["facture", "bill", "finance"],
            "customer":     ["client", "partner", "finance"],
            "order":        ["commande", "sorder", "achat"],
            "supplier":     ["fournisseur", "vendor", "achat"],
            "employee":     ["employe", "rh", "staff"],
            "salary":       ["salaire", "payroll", "rh"],
            "product":      ["produit", "article", "item"],
            "payment":      ["paiement", "reglement"],
            "account":      ["compte", "gl", "finance"],
        })
        return True
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Config index MeiliSearch: {e}")
        return False


def index_entity_meili(
    client,
    entity_id: str,
    source_id: str,
    source_name: str,
    source_type: str,
    entity_name: str,
    entity_type: str,
    description: Optional[str],
    domain: str,
    concept: Optional[str],
    entity_class: str,
    tags: List[str],
    field_names: List[str],
    sample_values: Optional[List[str]] = None,
) -> bool:
    """Indexe une entité dans MeiliSearch avec sample data."""
    try:
        # Construire le texte des synonymes
        tokens = _normalize_name(entity_name)
        resolved = _resolve_synonyms(tokens)
        synonyms_text = " ".join(resolved + tokens)

        doc = {
            "id":            str(entity_id),
            "source_id":     str(source_id),
            "source_name":   source_name,
            "source_type":   source_type,
            "name":          entity_name,
            "display_name":  entity_name.replace("_", " ").title(),
            "description":   description or "",
            "entity_type":   entity_type,
            "domain":        domain,
            "concept":       concept or "",
            "entity_class":  entity_class,
            "tags":          " ".join(tags),
            "field_names":   " ".join(field_names[:50]),
            "synonyms_text": synonyms_text,
            "sample_values": " ".join(sample_values[:30]) if sample_values else "",
        }

        idx = client.index(MEILI_INDEX)
        idx.add_documents([doc])
        return True
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Index MeiliSearch {entity_name}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR PRINCIPAL — enrich_source()
# ═══════════════════════════════════════════════════════════════════════════════

async def enrich_source(source_id: UUID, source_name: str, source_type: str) -> Dict[str, Any]:
    """
    Enrichit sémantiquement toutes les entités d'une source :
    1. Classification domaine + concept + classe
    2. Détection dimensions analytiques
    3. Tags sémantiques
    4. Embeddings BERT
    5. Indexation MeiliSearch
    6. Mise à jour PostgreSQL
    """
    from .database import get_pg_pool

    pool = await get_pg_pool()
    meili_client = _get_meili_client()

    if meili_client:
        _ensure_meili_index(meili_client)

    # ── Charger le calendrier fiscal de la source ─────────────────────────────
    fiscal_year_start = 1  # défaut : janvier
    try:
        async with pool.acquire() as conn:
            fiscal_row = await conn.fetchrow(
                "SELECT fiscal_year_start FROM source_fiscal_calendars WHERE source_id=$1",
                source_id
            )
            if fiscal_row:
                fiscal_year_start = fiscal_row["fiscal_year_start"]
                logger.info(f"[SemanticEnricher] Calendrier fiscal: mois {fiscal_year_start}")
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Calendrier fiscal non disponible: {e}")

    # ── Charger les domaines personnalisés ────────────────────────────────────
    custom_domains: Optional[Dict[str, List[str]]] = None
    try:
        async with pool.acquire() as conn:
            domain_rows = await conn.fetch(
                "SELECT domain_name, patterns FROM source_domains WHERE source_id=$1 ORDER BY priority DESC",
                source_id
            )
            if domain_rows:
                custom_domains = {r["domain_name"]: list(r["patterns"]) for r in domain_rows}
                logger.info(f"[SemanticEnricher] {len(custom_domains)} domaines personnalisés chargés")
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Domaines personnalisés non disponibles: {e}")

    # ── Charger les top_values depuis entity_profiles ─────────────────────────
    sample_values_by_entity: Dict[str, List[str]] = {}
    try:
        async with pool.acquire() as conn:
            profile_rows = await conn.fetch("""
                SELECT entity_name, profile_data
                FROM entity_profiles
                WHERE source_id = $1
            """, source_id)
        for pr in profile_rows:
            ename = pr["entity_name"]
            try:
                pdata = json.loads(pr["profile_data"]) if isinstance(pr["profile_data"], str) else dict(pr["profile_data"])
                sample_vals = []
                columns = pdata.get("columns", [])
                # columns peut être une liste ou un dict
                if isinstance(columns, dict):
                    col_list = columns.values()
                else:
                    col_list = columns  # c'est une liste
                for col_profile in col_list:
                    for tv in col_profile.get("top_values", [])[:3]:
                        val = str(tv.get("value", "")).strip()
                        if val and len(val) < 50:
                            sample_vals.append(val)
                sample_values_by_entity[ename] = sample_vals[:30]
            except Exception:
                pass
        logger.info(f"[SemanticEnricher] Sample values chargés pour {len(sample_values_by_entity)} entités")
    except Exception as e:
        logger.warning(f"[SemanticEnricher] Sample values non disponibles: {e}")

    # Charger toutes les entités de la source
    async with pool.acquire() as conn:
        entity_rows = await conn.fetch("""
            SELECT se.id, se.name, se.entity_type, se.description, se.row_count,
                   COALESCE(
                       json_agg(ef.name ORDER BY ef.position) FILTER (WHERE ef.name IS NOT NULL),
                       '[]'
                   ) AS field_names,
                   COALESCE(
                       json_agg(ef.data_type ORDER BY ef.position) FILTER (WHERE ef.name IS NOT NULL),
                       '[]'
                   ) AS field_types
            FROM source_entities se
            LEFT JOIN entity_fields ef ON ef.entity_id = se.id
            WHERE se.source_id = $1
              AND se.entity_type IN ('table', 'view', 'materialized_view', 'odata_entity',
                                     'rest_resource', 'graphql_type', 'excel_sheet',
                                     'csv_file', 'parquet_file')
            GROUP BY se.id, se.name, se.entity_type, se.description, se.row_count
            ORDER BY se.name
        """, source_id)

    if not entity_rows:
        return {"success": False, "message": "Aucune entité à enrichir", "enriched": 0}

    logger.info(f"[SemanticEnricher] {len(entity_rows)} entités à enrichir pour {source_name}")

    # ── Préparer les données pour embeddings ──────────────────────────────────
    enriched_entities = []
    texts_for_embedding = []

    for row in entity_rows:
        entity_id   = row["id"]
        entity_name = row["name"]
        entity_type = row["entity_type"]
        description = row["description"]
        row_count   = row["row_count"]
        field_names = json.loads(row["field_names"]) if isinstance(row["field_names"], str) else list(row["field_names"] or [])
        field_types = json.loads(row["field_types"]) if isinstance(row["field_types"], str) else list(row["field_types"] or [])

        # Classification
        domain       = classify_domain_with_custom(entity_name, field_names, custom_domains)
        concept      = classify_concept(entity_name)
        entity_class = classify_entity_class(entity_name, row_count)
        dimensions   = detect_dimensions(field_names)
        tags         = extract_semantic_tags(entity_name, domain, concept, entity_class, dimensions)

        # Texte pour embedding
        text = _build_entity_text(entity_name, field_names, description, domain, concept)
        texts_for_embedding.append(text)

        enriched_entities.append({
            "id":           entity_id,
            "name":         entity_name,
            "entity_type":  entity_type,
            "description":  description,
            "domain":       domain,
            "concept":      concept,
            "entity_class": entity_class,
            "dimensions":   dimensions,
            "tags":         tags,
            "field_names":  field_names,
            "field_types":  field_types,
        })

    # ── Calcul des embeddings (batch) ─────────────────────────────────────────
    logger.info(f"[SemanticEnricher] Calcul embeddings BERT pour {len(texts_for_embedding)} entités...")
    embeddings = await asyncio.to_thread(compute_bert_embedding, texts_for_embedding)

    # ── Mise à jour PostgreSQL + MeiliSearch ──────────────────────────────────
    enriched_count = 0
    meili_count    = 0

    async with pool.acquire() as conn:
        for i, entity in enumerate(enriched_entities):
            embedding_vector = embeddings[i] if i < len(embeddings) and embeddings[i] else None

            # Mise à jour PostgreSQL
            try:
                if embedding_vector:
                    await conn.execute("""
                        UPDATE source_entities
                        SET business_domain  = $1,
                            business_concept = $2,
                            entity_class     = $3,
                            semantic_tags    = $4::jsonb,
                            dimensions       = $5::jsonb,
                            embedding        = $6::vector
                        WHERE id = $7
                    """,
                        entity["domain"],
                        entity["concept"],
                        entity["entity_class"],
                        json.dumps(entity["tags"]),
                        json.dumps(entity["dimensions"]),
                        str(embedding_vector),
                        entity["id"],
                    )
                else:
                    await conn.execute("""
                        UPDATE source_entities
                        SET business_domain  = $1,
                            business_concept = $2,
                            entity_class     = $3,
                            semantic_tags    = $4::jsonb,
                            dimensions       = $5::jsonb
                        WHERE id = $6
                    """,
                        entity["domain"],
                        entity["concept"],
                        entity["entity_class"],
                        json.dumps(entity["tags"]),
                        json.dumps(entity["dimensions"]),
                        entity["id"],
                    )

                # Mise à jour dimension_type + hierarchy sur les champs
                for j, fname in enumerate(entity["field_names"]):
                    dtype = entity["field_types"][j] if j < len(entity["field_types"]) else ""
                    dim_type = detect_field_dimension(fname, dtype)
                    if dim_type:
                        # Détecter le dialecte depuis le type de source
                        dialect = "postgresql" if "postgres" in source_type.lower() else "mssql"
                        hierarchy = build_field_hierarchy(fname, dim_type, dtype, dialect, fiscal_year_start)
                        await conn.execute("""
                            UPDATE entity_fields
                            SET dimension_type      = $1,
                                dimension_hierarchy = $2::jsonb
                            WHERE entity_id = $3 AND name = $4
                        """, dim_type, json.dumps(hierarchy) if hierarchy else None,
                            entity["id"], fname)

                enriched_count += 1

            except Exception as e:
                logger.warning(f"[SemanticEnricher] PG update {entity['name']}: {e}")

            # Indexation MeiliSearch
            if meili_client:
                ok = index_entity_meili(
                    meili_client,
                    str(entity["id"]),
                    str(source_id),
                    source_name,
                    source_type,
                    entity["name"],
                    entity["entity_type"],
                    entity["description"],
                    entity["domain"],
                    entity["concept"],
                    entity["entity_class"],
                    entity["tags"],
                    entity["field_names"],
                    sample_values=sample_values_by_entity.get(entity["name"], []),
                )
                if ok:
                    meili_count += 1

    # Statistiques par domaine
    domain_stats: Dict[str, int] = {}
    for ent in enriched_entities:
        domain_stats[ent["domain"]] = domain_stats.get(ent["domain"], 0) + 1

    logger.info(f"[SemanticEnricher] ✅ {enriched_count} entités enrichies | MeiliSearch: {meili_count}")

    return {
        "success":        True,
        "enriched":       enriched_count,
        "meili_indexed":  meili_count,
        "domain_stats":   domain_stats,
        "message":        f"{enriched_count} entités enrichies sémantiquement",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# C. RECHERCHE SÉMANTIQUE HYBRIDE
# ═══════════════════════════════════════════════════════════════════════════════

async def semantic_search(
    query: str,
    source_ids: Optional[List[str]] = None,
    limit: int = 10,
    use_vector: bool = True,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Recherche sémantique hybride :
    1. MeiliSearch (full-text + fuzzy)
    2. pgvector (similarité vectorielle)
    3. Boost par historique des recherches passées
    4. Fusion et re-ranking des résultats
    """
    results_meili = []
    results_vector = []

    # ── 1. Recherche MeiliSearch ──────────────────────────────────────────────
    meili_client = _get_meili_client()
    if meili_client:
        try:
            idx = meili_client.index(MEILI_INDEX)
            filter_str = None
            if source_ids:
                filter_str = " OR ".join([f'source_id = "{sid}"' for sid in source_ids])

            search_params = {"limit": limit * 2}
            if filter_str:
                search_params["filter"] = filter_str

            meili_results = idx.search(query, search_params)
            for hit in meili_results.get("hits", []):
                results_meili.append({
                    "entity_id":    hit.get("id"),
                    "source_id":    hit.get("source_id"),
                    "source_name":  hit.get("source_name"),
                    "name":         hit.get("name"),
                    "domain":       hit.get("domain"),
                    "concept":      hit.get("concept"),
                    "entity_class": hit.get("entity_class"),
                    "description":  hit.get("description"),
                    "score":        hit.get("_rankingScore", 0.5),
                    "method":       "fulltext",
                })
        except Exception as e:
            logger.warning(f"[SemanticSearch] MeiliSearch error: {e}")

    # ── 2. Recherche vectorielle pgvector ─────────────────────────────────────
    if use_vector:
        try:
            from .database import get_pg_pool
            pool = await get_pg_pool()

            # Embedding de la requête avec BERT
            query_embeddings = await asyncio.to_thread(
                compute_bert_embedding, [query]
            )
            if query_embeddings and query_embeddings[0]:
                query_vector = str(query_embeddings[0])
                async with pool.acquire() as conn:
                    filter_clause = ""
                    params: list = [query_vector, limit * 2]
                    if source_ids:
                        filter_clause = f"AND source_id = ANY($3::uuid[])"
                        params.append(source_ids)

                    vector_rows = await conn.fetch(f"""
                        SELECT id, name, business_domain, business_concept,
                               entity_class, description, source_id,
                               1 - (embedding <=> $1::vector) AS similarity
                        FROM source_entities
                        WHERE embedding IS NOT NULL {filter_clause}
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """, *params)

                    for row in vector_rows:
                        results_vector.append({
                            "entity_id":    str(row["id"]),
                            "source_id":    str(row["source_id"]),
                            "name":         row["name"],
                            "domain":       row["business_domain"],
                            "concept":      row["business_concept"],
                            "entity_class": row["entity_class"],
                            "description":  row["description"],
                            "score":        float(row["similarity"]),
                            "method":       "vector",
                        })
        except Exception as e:
            logger.warning(f"[SemanticSearch] pgvector error: {e}")

    # ── 3. Fusion hybride ─────────────────────────────────────────────────────
    merged: Dict[str, Dict] = {}

    for r in results_meili:
        key = r["entity_id"]
        merged[key] = {**r, "final_score": r["score"] * 0.6}

    for r in results_vector:
        key = r["entity_id"]
        if key in merged:
            merged[key]["final_score"] += r["score"] * 0.4
            merged[key]["method"] = "hybrid"
        else:
            merged[key] = {**r, "final_score": r["score"] * 0.4}

    # ── 3. Boost par historique des recherches ────────────────────────────────
    history_boost: Dict[str, float] = {}
    try:
        from .database import get_pg_pool
        pool = await get_pg_pool()
        query_norm = query.lower().strip()
        async with pool.acquire() as conn:
            history_rows = await conn.fetch("""
                SELECT clicked_id, COUNT(*) AS click_count
                FROM search_history
                WHERE query_norm ILIKE $1
                  AND clicked_id IS NOT NULL
                GROUP BY clicked_id
                ORDER BY click_count DESC
                LIMIT 20
            """, f"%{query_norm}%")
            for hr in history_rows:
                # Boost max 20% basé sur le nombre de clics
                boost = min(float(hr["click_count"]) * 0.02, 0.20)
                history_boost[str(hr["clicked_id"])] = boost
    except Exception as e:
        logger.warning(f"[SemanticSearch] History boost error: {e}")

    # Appliquer le boost historique
    for key in merged:
        if key in history_boost:
            merged[key]["final_score"] += history_boost[key]
            merged[key]["history_boosted"] = True

    # Trier par score final
    final_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)

    # ── 4. Construire résultats finaux ────────────────────────────────────────
    output = [
        {
            "rank":             i + 1,
            "entity_id":        r["entity_id"],
            "source_id":        r.get("source_id"),
            "source_name":      r.get("source_name"),
            "name":             r["name"],
            "domain":           r.get("domain"),
            "concept":          r.get("concept"),
            "entity_class":     r.get("entity_class"),
            "description":      r.get("description"),
            "relevance":        round(r["final_score"], 4),
            "method":           r.get("method", "fulltext"),
            "history_boosted":  r.get("history_boosted", False),
        }
        for i, r in enumerate(final_results[:limit])
    ]

    # ── 5. Sauvegarder dans l'historique ─────────────────────────────────────
    try:
        from .database import get_pg_pool
        pool = await get_pg_pool()
        query_norm = query.lower().strip()
        top5 = [{"entity_id": r["entity_id"], "name": r["name"], "relevance": r["relevance"]} for r in output[:5]]
        source_id_val = UUID(source_ids[0]) if source_ids else None
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO search_history
                    (source_id, query, query_norm, results, result_count, user_id)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
            """, source_id_val, query, query_norm,
                json.dumps(top5), len(output), user_id)
    except Exception as e:
        logger.warning(f"[SemanticSearch] History save error: {e}")

    return output