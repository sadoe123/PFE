"""
OnePilot – Dashboard Engine §2.4.3
Génération automatique de dashboards interactifs depuis une question NL.

Pipeline :
  Question NL
    → DashboardIntent (métriques, dimensions, filtres)
    → DataRetriever (SQL execution)
    → VisualizationSelector (type de chart optimal)
    → InsightDetector (anomalies, tendances)
    → DashboardSpec (JSON spec pour Chart.js)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# TYPES DE VISUALISATION
# ══════════════════════════════════════════════════════════════

class ChartType:
    BAR         = "bar"
    BAR_H       = "bar_horizontal"
    LINE        = "line"
    PIE         = "pie"
    DOUGHNUT    = "doughnut"
    SCATTER     = "scatter"
    AREA        = "area"
    GAUGE       = "gauge"
    KPI_CARD    = "kpi_card"
    TABLE       = "table"
    HEATMAP     = "heatmap"
    WATERFALL   = "waterfall"
    FUNNEL      = "funnel"
    TREEMAP     = "treemap"
    BUBBLE      = "bubble"
    SPARKLINE   = "sparkline"
    PIVOT       = "pivot"
    KPI_DELTA   = "kpi_delta"
    # ── Nouveaux types Sprint 1 ───────────────────────────────
    CHOROPLETH  = "choropleth"    # Carte géographique colorée (Plotly)
    BUBBLE_MAP  = "bubble_map"    # Carte avec bulles proportionnelles (Plotly)
    SANKEY      = "sankey"        # Diagramme de flux (D3 / Plotly)


# ══════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════

@dataclass
class DashboardWidget:
    """Un widget individuel dans le dashboard."""
    widget_id:   str
    chart_type:  str
    title:       str
    data:        Dict[str, Any]       # {labels, datasets, ...}
    options:     Dict[str, Any] = field(default_factory=dict)
    width:       int = 12             # colonnes (sur 12)
    height:      str = "300px"
    insights:    List[str] = field(default_factory=list)
    sql:         Optional[str] = None
    filters:     List[Dict] = field(default_factory=list)


@dataclass
class DashboardSpec:
    """Spécification complète d'un dashboard."""
    dashboard_id: str
    title:        str
    question:     str
    source_id:    str
    widgets:      List[DashboardWidget]
    filters:      List[Dict] = field(default_factory=list)
    insights:     List[str] = field(default_factory=list)
    generated_at: str = ""
    duration_ms:  int = 0
    options:      Dict[str, Any] = field(default_factory=dict)   # Sprint 3: recommendations

    def to_dict(self) -> Dict:
        return {
            "dashboard_id": self.dashboard_id,
            "title":        self.title,
            "question":     self.question,
            "source_id":    self.source_id,
            "generated_at": self.generated_at,
            "duration_ms":  self.duration_ms,
            "filters":      self.filters,
            "insights":     self.insights,
            "options":      self.options,          # Sprint 3 : recommendations
            "widgets": [
                {
                    "widget_id":  w.widget_id,
                    "chart_type": w.chart_type,
                    "title":      w.title,
                    "data":       w.data,
                    "options":    w.options,
                    "width":      w.width,
                    "height":     w.height,
                    "insights":   w.insights,
                    "sql":        w.sql,
                    "filters":    w.filters,
                }
                for w in self.widgets
            ],
        }


# ══════════════════════════════════════════════════════════════
# INTENT EXTRACTION — métriques, dimensions, filtres
# ══════════════════════════════════════════════════════════════

@dataclass
class DashboardIntent:
    metrics:    List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    time_field: Optional[str] = None
    geo_field:  Optional[str] = None
    filters:    List[Dict] = field(default_factory=list)
    top_n:      Optional[int] = None
    chart_hint: Optional[str] = None   # hint explicite de l'utilisateur
    question:   str = ""                # question originale pour sélection visuelle
    is_trend:       bool = False
    is_comparison:  bool = False
    is_composition: bool = False
    is_correlation: bool = False
    is_top_n:       bool = False
    is_kpi:         bool = False
    is_geo:         bool = False        # ← Sprint 1 : question géographique
    is_flow:        bool = False        # ← Sprint 1 : question de flux/parcours


class DashboardIntentExtractor:
    """
    Extrait les métriques, dimensions et paramètres de visualisation
    depuis une question NL et les slots NLU.
    """

    # Mots-clés → métriques connues
    _METRIC_HINTS = {
        "chiffre": "revenue", "affaires": "revenue", "ca": "revenue",
        "vente": "revenue", "ventes": "revenue", "revenu": "revenue",
        "montant": "amount", "total": "total", "somme": "sum",
        "commande": "orders", "commandes": "orders",
        "client": "customers", "clients": "customers",
        "produit": "products", "produits": "products",
        "marge": "margin", "bénéfice": "profit", "profit": "profit",
        "stock": "stock", "quantité": "quantity",
        "salaire": "salary", "masse": "payroll",
    }

    # Mots-clés → dimensions temporelles
    _TIME_HINTS = [
        "mois", "month", "année", "year", "trimestre", "quarter",
        "semaine", "week", "jour", "day", "date", "période",
        "trndate", "orderdate", "created_at", "updated_at",
    ]

    # Mots-clés → types de chart
    _CHART_HINTS = {
        # ── Courbe / Line ─────────────────────────────────────
        "en courbe": "line", "en ligne": "line", "courbe": "line",
        "évolution": "line", "tendance": "line", "trend": "line",
        "progression": "line", "graphique linéaire": "line",
        # ── Camembert / Pie ───────────────────────────────────
        "en camembert": "pie", "camembert": "pie",
        "en cercle": "pie", "en secteurs": "pie",
        "graphique circulaire": "pie", "circulaire": "pie",
        # ── Donut / Anneau ────────────────────────────────────
        "en donut": "doughnut", "donut": "doughnut",
        "en anneau": "doughnut", "anneau": "doughnut",
        # ── Barres verticales ─────────────────────────────────
        "en barres": "bar", "en colonnes": "bar",
        "histogramme": "bar", "en histogramme": "bar",
        "comparaison": "bar", "comparer": "bar",
        # ── Barres horizontales ───────────────────────────────
        "en barres horizontales": "bar_horizontal",
        "barres horizontales": "bar_horizontal",
        "classement": "bar_horizontal", "ranking": "bar_horizontal",
        "top": "bar_horizontal",
        # ── Aire / Area ───────────────────────────────────────
        "en aire": "area", "surface": "area",
        "cumulatif": "area", "cumul": "area", "running": "area",
        # ── Treemap ───────────────────────────────────────────
        "treemap": "treemap", "en arbre": "treemap",
        "hiérarchie": "treemap", "arborescence": "treemap",
        # ── Scatter ───────────────────────────────────────────
        "corrélation": "scatter", "nuage de points": "scatter",
        "scatter": "scatter", "relation": "scatter",
        # ── Gauge ─────────────────────────────────────────────
        "en jauge": "gauge", "jauge": "gauge",
        "gauge": "gauge", "objectif": "gauge",
        # ── KPI ───────────────────────────────────────────────
        "kpi": "kpi_card", "indicateur": "kpi_card",
        "carte kpi": "kpi_card",
        # ── Tableau ───────────────────────────────────────────
        "en tableau": "table", "tableau": "table", "liste": "table",
        # ── Waterfall ─────────────────────────────────────────
        "cascade": "waterfall", "waterfall": "waterfall",
        # ── Funnel ────────────────────────────────────────────
        "entonnoir": "funnel", "funnel": "funnel",
        # ── Répartition auto (pie/donut) ──────────────────────
        "répartition": "pie", "distribution": "pie",
        "part": "pie", "proportion": "pie", "pourcentage": "pie",
        # ── Carte géographique ────────────────────────────────
        "carte": "choropleth", "par pays": "choropleth",
        "par région": "choropleth", "par ville": "bubble_map",
        "géographique": "choropleth", "choropleth": "choropleth",
        "carte monde": "choropleth", "map": "choropleth",
        # ── Sankey / Flux ─────────────────────────────────────
        "flux": "sankey", "parcours": "sankey", "sankey": "sankey",
        "transition": "sankey", "mouvement": "sankey", "flow": "sankey",
    }

    def extract(
        self,
        question: str,
        slots,                          # QuerySlots from NLU
        schema: Dict[str, List[str]],
    ) -> DashboardIntent:
        q = question.lower()
        intent = DashboardIntent()
        intent.question = question

        # ── Métriques ────────────────────────────────────────
        if slots.metric:
            intent.metrics.append(slots.metric)
        for kw, metric in self._METRIC_HINTS.items():
            if kw in q and metric not in intent.metrics:
                intent.metrics.append(metric)
        if not intent.metrics:
            intent.metrics = ["value"]  # fallback

        # ── Dimensions temporelles ────────────────────────────
        for hint in self._TIME_HINTS:
            if hint in q:
                intent.is_trend = True
                intent.time_field = hint
                break
        # Cherche un champ date dans le schéma
        for tbl in (slots.table_names or list(schema.keys())[:3]):
            for fld in schema.get(tbl, []):
                if any(h in fld.lower() for h in ["date", "time", "at", "period"]):
                    intent.time_field = fld
                    intent.is_trend = True
                    break

        # ── Dimensions catégorielles (group_by) ───────────────
        if slots.group_by:
            intent.dimensions.append(slots.group_by)

        # ── Type de chart hint ────────────────────────────────
        for kw, chart in self._CHART_HINTS.items():
            if kw in q:
                intent.chart_hint = chart
                break

        # ── Intent flags ─────────────────────────────────────
        intent.is_trend       = any(k in q for k in ["évolution","tendance","trend","mensuel","annuel","par mois","par année"])
        intent.is_comparison  = any(k in q for k in ["comparer","comparaison","vs","versus","par rapport"])
        intent.is_composition = any(k in q for k in ["répartition","distribution","part","proportion","pourcentage"])
        intent.is_correlation = any(k in q for k in ["corrélation","relation entre","lien entre"])
        intent.is_top_n       = any(k in q for k in ["top","classement","ranking","premier","meilleur"])
        intent.is_kpi         = any(k in q for k in ["kpi","indicateur","dashboard","tableau de bord","bilan"])
        intent.is_geo         = any(k in q for k in ["pays","country","région","region","ville","city","carte","map","géographique","geographic"])
        intent.is_flow        = any(k in q for k in ["flux","parcours","sankey","transition","mouvement client","flow"])

        # Détecte le champ géographique dans le schéma
        if intent.is_geo:
            for tbl in (slots.table_names or list(schema.keys())[:3]):
                for fld in schema.get(tbl, []):
                    if any(h in fld.lower() for h in ["country","pays","countryname","region","city","ville"]):
                        intent.geo_field = fld
                        break

        intent.top_n = slots.top_n

        return intent


# ══════════════════════════════════════════════════════════════
# VISUALIZATION SELECTOR — choisit le type de chart optimal
# ══════════════════════════════════════════════════════════════

class VisualizationSelector:
    """
    Algorithme de décision pour le type de visualisation optimal.
    Basé sur : intent, structure des données, nombre de lignes/colonnes.
    """

    def select(
        self,
        intent: DashboardIntent,
        rows: List[Dict],
        columns: List[str],
    ) -> str:
        """Retourne le ChartType optimal."""

        # Hint explicite de l'utilisateur → priorité absolue
        if intent.chart_hint:
            return intent.chart_hint

        n_rows = len(rows)
        n_num  = self._count_numeric(rows, columns)
        n_cat  = len(columns) - n_num
        q = (getattr(intent, "question", "") or "").lower()

        # ── Géographie → Choropleth ou Bubble map ─────────────
        if intent.is_geo and intent.geo_field:
            # Si la dimension géo est une ville → bubble map
            if any(k in intent.geo_field.lower() for k in ["city","ville","shipcity"]):
                return ChartType.BUBBLE_MAP
            return ChartType.CHOROPLETH

        # ── Géographie sans geo_field détecté : cherche dans les colonnes disponibles ──
        if intent.is_geo:
            city_cols = [c for c in columns if any(k in c.lower() for k in ["city","ville","shipcity"])]
            country_cols = [c for c in columns if any(k in c.lower() for k in ["country","pays","shipcountry","countryname"])]
            if city_cols:
                return ChartType.BUBBLE_MAP
            if country_cols:
                return ChartType.CHOROPLETH
            # Pas de colonne géo dans les données → fallback bar
            # (les données ne sont pas géographiques, le mot "pays" était dans la question mais pas dans les données)

        # ── Flux → Sankey ────────────────────────────────────
        if intent.is_flow:
            return ChartType.SANKEY

        # ── Règles de décision ────────────────────────────────
        if intent.is_trend and intent.time_field:
            return ChartType.LINE

        if intent.is_composition and n_rows <= 8:
            return ChartType.DOUGHNUT

        if intent.is_composition and n_rows > 8:
            return ChartType.BAR

        if intent.is_correlation and n_num >= 3:
            return ChartType.BUBBLE   # 3 métriques → bulles (x, y, taille)
        if intent.is_correlation and n_num >= 2:
            return ChartType.SCATTER

        if intent.is_top_n:
            return ChartType.BAR_H if n_rows > 5 else ChartType.BAR

        if intent.is_kpi and n_rows == 1:
            return ChartType.KPI_CARD

        if n_rows == 1 and n_num >= 1:
            return ChartType.KPI_CARD

        # ── Heatmap : 2 dimensions catégorielles + 1 métrique + volume ──
        if n_cat >= 2 and n_num >= 1 and n_rows > 20:
            return ChartType.HEATMAP

        # ── Funnel : mots-clés conversion/entonnoir ──────────
        funnel_kw = ["entonnoir","funnel","conversion","étape","pipeline","tunnel"]
        if any(k in q for k in funnel_kw):
            return ChartType.FUNNEL

        if n_num >= 1 and n_rows <= 20:
            return ChartType.BAR

        if n_num >= 1 and n_rows > 20:
            return ChartType.LINE

        return ChartType.TABLE  # fallback

    def _count_numeric(self, rows: List[Dict], columns: List[str]) -> int:
        if not rows:
            return 0
        sample = rows[0]
        return sum(
            1 for col in columns
            if isinstance(sample.get(col), (int, float))
        )


# ══════════════════════════════════════════════════════════════
# INSIGHT DETECTOR — anomalies, tendances, seuils
# ══════════════════════════════════════════════════════════════

class InsightDetector:
    """Détecte automatiquement les insights dans les données."""

    def detect(
        self,
        rows: List[Dict],
        columns: List[str],
        chart_type: str,
    ) -> List[str]:
        insights = []
        if not rows or len(rows) < 2:
            return insights

        _ip=["_id","id_","rowid","productid","orderid","customerid","employeeid"]
        _mh=["montant","closingbalanceamount","total_closingbalanceamount","montantavecsigne","amounti","total_amounti","prpcntrvamount","price","amount","total","revenue","montant","freight","montant_total"]
        from decimal import Decimal as _DI
        num_cols=[c for c in columns
                  if isinstance(rows[0].get(c),(int,float,_DI))
                  and not any(p in c.lower() for p in _ip)]
        mn=[c for c in num_cols if any(h in c.lower() for h in _mh)]
        if mn: num_cols=mn+[c for c in num_cols if c not in mn]
        if not num_cols: return insights
        col=num_cols[0]; values=[float(r.get(col,0)) for r in rows if r.get(col) is not None]
        if len(values)<2: return insights
        def _fmt(v):
            if not isinstance(v,(int,float)): return str(v)
            a=abs(v)
            if a>=1e6: return f"{v/1e6:.1f}M"
            if a>=1e3: return f"{v/1e3:.1f}K"
            if isinstance(v,float) and v!=int(v): return f"{v:.2f}"
            return f"{int(v):,}"
        _lh=["name","nom","title","label","code","company","client","customer","product","category","region","city","productname","companyname","customername","categoryname","periode","period","month","year"]
        lc=next((c for c in columns if c not in num_cols and any(h in c.lower() for h in _lh)),next((c for c in columns if c not in num_cols),None))
        labels=[str(r.get(lc,f"#{i+1}")) for i,r in enumerate(rows)] if lc else [_fmt(v) for v in values]
        mx=max(values); mnv=min(values); av=sum(values)/len(values) if values else 1
        mi=values.index(mx); ni=values.index(mnv)
        if av>0 and mx>av*1.5: insights.append(f"[UP] {labels[mi]} : {_fmt(mx)} (+{((mx/av-1)*100):.0f}% vs moy.)")
        if av>0 and mnv<av*0.5: insights.append(f"[DOWN] **{labels[ni]}** : {_fmt(mnv)} (-{((1-mnv/av)*100):.0f}% vs moy.)")
        if chart_type in (ChartType.LINE,ChartType.AREA) and len(values)>=3:
            h1=sum(values[:len(values)//2])/(len(values)//2); h2=sum(values[len(values)//2:])/(len(values)-len(values)//2)
            if h1>0:
                if h2>h1*1.1: insights.append(f"[TREND_UP] Tendance haussière (+{((h2/h1-1)*100):.0f}%) — {_fmt(h1)} → {_fmt(h2)}")
                elif h2<h1*0.9: insights.append(f"[TREND_DOWN] Tendance baissière ({((h2/h1-1)*100):.0f}%) — {_fmt(h1)} → {_fmt(h2)}")
        if len(values)>=5 and sum(values)>0:
            sv=sorted(values,reverse=True); p=sum(sv[:max(1,len(sv)//5)])/sum(values)*100
            if p>60: insights.append(f"[BOLT] Top 20% représentent {p:.0f}% du total")
        return insights[:3]


# ══════════════════════════════════════════════════════════════
# DASHBOARD DESIGNER — compose les widgets
# ══════════════════════════════════════════════════════════════

class DashboardDesigner:
    """
    Construit le DashboardSpec complet depuis les données et l'intent.
    """

    def __init__(self):
        self.selector = VisualizationSelector()
        self.insights = InsightDetector()

    def design(
        self,
        question:  str,
        intent:    DashboardIntent,
        datasets:  List[Dict],     # [{sql, rows, columns, title}, ...]
        source_id: str,
    ) -> DashboardSpec:
        t0 = time.time()
        widgets = []
        all_insights = []

        for i, ds in enumerate(datasets):
            rows    = ds.get("rows", [])
            cols    = ds.get("columns", [])
            sql     = ds.get("sql", "")
            rt=ds.get("title",""); _TBL={"order details":"Détail commandes","orderdetails":"Détail commandes","orders":"Commandes","customers":"Clients","products":"Produits","employees":"Employés","suppliers":"Fournisseurs","categories":"Catégories","shippers":"Transporteurs","invoices":"Factures","sales":"Ventes","dernière integration bancaire":"Soldes bancaires","si_trésorerie":"Trésorerie","si_bancaire":"Mouvements bancaires","vdtssxaaccountdata":"Données comptes","comptes":"Comptes","financement_bi":"Financement","tableaux d'amortissement":"Amortissements"}
            title=_TBL.get(rt.lower(),rt.replace("_"," ").title()) or question[:50]

            if not rows:
                continue

            # Sélection du type de chart
            chart_type = self.selector.select(intent, rows, cols)

            # Construction du widget data selon le type
            try:
                widget_data = self._build_widget_data(chart_type, rows, cols, intent)
            except Exception as _we:
                logger.warning(f"[Dashboard] widget_data error: {_we}")
                widget_data = {"type":"table","headers":cols,"rows":[[str(r.get(c,"")) for c in cols] for r in rows[:20]],"total":len(rows)}

            # Détection insights
            wi = self.insights.detect(rows, cols, chart_type)
            all_insights.extend(wi)

            # Dimensions du widget
            width, height = self._get_dimensions(chart_type, i, len(datasets))

            widget = DashboardWidget(
                widget_id  = f"w_{i}_{chart_type}",
                chart_type = chart_type,
                title      = title,
                data       = widget_data,
                options    = self._build_options(chart_type, intent),
                width      = width,
                height     = height,
                insights   = wi,
                sql        = sql,
                filters    = self._build_filters(cols, rows),
            )
            widgets.append(widget)

            # KPI card supplémentaire si données agrégées
            if chart_type in (ChartType.LINE, ChartType.BAR, ChartType.AREA) and rows:
                kpi = self._build_kpi_widget(rows, cols, i)
                if kpi:
                    # Fallback: if monetary KPI is 0, use count
                    if kpi.data.get("value",0)==0 and kpi.data.get("format")=="currency":
                        cnt_col=next((c for c in cols if c.lower() in ("nb_lignes","count","nb_commandes")),None)
                        if cnt_col:
                            kpi.data["value"]=sum(r.get(cnt_col,0) for r in rows if r.get(cnt_col))
                            kpi.data["label"]="Nb financements"
                            kpi.data["format"]="number"
                            kpi.title="Nb financements"
                    widgets.insert(0, kpi)

        ms = int((time.time() - t0) * 1000)

        return DashboardSpec(
            dashboard_id = f"dash_{int(time.time())}",
            title        = self._generate_title(question),
            question     = question,
            source_id    = source_id,
            widgets      = widgets,
            filters      = self._build_global_filters(datasets),
            insights     = list(dict.fromkeys(all_insights))[:5],
            generated_at = __import__("datetime").datetime.utcnow().isoformat(),
            duration_ms  = ms,
        )

    # ── Builders ──────────────────────────────────────────────

    def _build_widget_data(
        self,
        chart_type: str,
        rows: List[Dict],
        cols: List[str],
        intent: DashboardIntent,
    ) -> Dict:
        """Construit le data object Chart.js."""
        # Safety: ensure values is always defined
        values = []
        from decimal import Decimal as _Dec

        # Filtre les colonnes ID (pas des métriques)
        _id_patterns = [
            "_id","id_","rowid","rev","revtype","seqno","seqnum","_num","_no",
            "productid","orderid","customerid","employeeid","supplierid","categoryid",
            "shipvia","shippedid","regionid","territoryid","reportsto",
        ]
        # Préférer les colonnes avec de vraies métriques métier
        _metric_hints = ["price","amount","total","qty","quantity","stock","units",
                         "salary","cost","revenue","montant","prix","valeur","count"]
        from decimal import Decimal as _Dec
        def _is_num(v): return isinstance(v, (int, float, _Dec))
        num_cols = [
            col for col in cols
            if rows and _is_num(rows[0].get(col))
            and not any(p in col.lower() for p in _id_patterns)
        ]
        # Prioriser les colonnes métriques
        _count_cols = {"nb_commandes","nb_lignes","count_total","count","nb","moyenne","average","avg"}
        metric_cols = [col for col in num_cols
                       if any(h in col.lower() for h in _metric_hints)
                       and col.lower() not in _count_cols]
        if metric_cols:
            num_cols = metric_cols + [c for c in num_cols
                                      if c not in metric_cols
                                      and c.lower() not in _count_cols]
        # Fallback si tous les numériques sont des IDs
        if not num_cols:
            num_cols = [col for col in cols if rows and isinstance(rows[0].get(col), (int, float))]
        cat_cols  = [c for c in cols if c not in num_cols]
        # Préférer les colonnes avec des noms métier comme label
        _label_hints = ["name","nom","title","label","code","description",
                        "company","client","customer","product","category","region","city"]
        label_hints_cols = [c for c in cat_cols if any(h in c.lower() for h in _label_hints)]
        label_col = label_hints_cols[0] if label_hints_cols else (cat_cols[0] if cat_cols else (cols[0] if cols else "index"))
        value_col = num_cols[0] if num_cols else (cols[-1] if cols else "value")

        labels = [str(r.get(label_col, i)) for i, r in enumerate(rows)]
        values = [float(r.get(value_col, 0)) if r.get(value_col) is not None else 0 for r in rows] if value_col else [0]*len(rows)

        if chart_type == ChartType.KPI_CARD:
            total = sum(float(v) for v in values if isinstance(v, (int, float)))
            # Label humain : préférer le nom de colonne le plus lisible
            _col_labels = {
                "total": "Total", "moyenne": "Moyenne", "nb_lignes": "Nb lignes",
                "unitprice": "Prix unitaire", "unitsinstock": "En stock",
                "freight": "Frais de port", "reorderlevel": "Seuil réappro.",
                "amount": "Montant", "revenue": "Chiffre d'affaires",
                "salary": "Salaire", "balance": "Solde", "budget_pct": "Budget (%)",
            }
            human_label = _col_labels.get(value_col.lower(), value_col.replace("_", " ").title())
            return {
                "type":   "kpi",
                "value":  total,
                "label":  human_label,
                "format": "currency" if any(k in value_col.lower() for k in ["price","amount","revenue","salary","freight","balance","total"]) else "number",
            }

        if chart_type in (ChartType.PIE, ChartType.DOUGHNUT):
            _n = min(10, len(labels))

            # Cas 1 : num_cols disponibles → on utilise les valeurs
            _vals = [float(v) for v in values[:_n]] if values else []
            _tot = sum(abs(v) for v in _vals)

            # Cas 2 : valeurs toutes à 0 → cherche colonne count explicite
            if _tot == 0 and rows:
                _cnt_col = next((c for c in rows[0].keys()
                               if c.lower() in ("nb_commandes","nb_lignes","count","count_total","total","nb")), None)
                if _cnt_col:
                    _vals = [float(r.get(_cnt_col, 0) or 0) for r in rows[:_n]]
                    _tot = sum(_vals)

            # Cas 3 : toujours 0 ou num_cols absent → COUNT(*) par label (nb d'occurrences)
            if _tot == 0 and rows:
                _vals = [1.0] * min(_n, len(rows))
                _tot = float(len(_vals))

            # Cas 4 : num_cols absent mais rows ont des numériques quelque part
            if not values and rows:
                for _c in cols:
                    try:
                        _test = [float(rows[i].get(_c, 0) or 0) for i in range(min(_n, len(rows)))]
                        _ts = sum(abs(v) for v in _test)
                        if _ts > 0:
                            _vals = _test
                            _tot = _ts
                            break
                    except (TypeError, ValueError):
                        continue

            _pcts = [round(v / _tot * 100, 1) if _tot else 0 for v in _vals]
            _lbls = [f"{labels[i]} ({_pcts[i]}%)" if i < len(labels) else f"#{i+1} ({_pcts[i]}%)" for i in range(len(_vals))]
            return {
                "labels": _lbls,
                "datasets": [{
                    "data": _vals,
                    "backgroundColor": self._palette(len(_vals)),
                    "borderWidth": 2,
                    "borderColor": "#0d1b2a",
                }],
            }

        if chart_type == ChartType.SCATTER:
            x_col = num_cols[0] if len(num_cols) >= 1 else cols[0]
            y_col = num_cols[1] if len(num_cols) >= 2 else num_cols[0]
            return {
                "datasets": [{
                    "label": f"{x_col} vs {y_col}",
                    "data": [{"x": r.get(x_col, 0), "y": r.get(y_col, 0)} for r in rows[:200]],
                    "backgroundColor": "rgba(0,200,245,0.6)",
                    "pointRadius": 4,
                }],
            }

        if chart_type == ChartType.BUBBLE:
            x_col = num_cols[0] if len(num_cols) >= 1 else cols[0]
            y_col = num_cols[1] if len(num_cols) >= 2 else num_cols[0]
            r_col = num_cols[2] if len(num_cols) >= 3 else num_cols[0]
            # Normaliser la taille des bulles entre 4 et 30
            r_vals = [float(row.get(r_col, 0) or 0) for row in rows[:200]]
            r_max = max(r_vals) if r_vals else 1
            r_min = min(r_vals) if r_vals else 0
            r_range = r_max - r_min or 1
            palette = self._palette(1)
            return {
                "type": "bubble",
                "datasets": [{
                    "label": f"{x_col} / {y_col} / {r_col}",
                    "data": [{
                        "x": float(row.get(x_col, 0) or 0),
                        "y": float(row.get(y_col, 0) or 0),
                        "r": 4 + ((float(row.get(r_col, 0) or 0) - r_min) / r_range) * 26,
                    } for row in rows[:200]],
                    "backgroundColor": palette[0].replace("0.85)", "0.55)"),
                    "borderColor":     palette[0],
                    "borderWidth": 1,
                }],
                "x_label": x_col,
                "y_label": y_col,
                "r_label": r_col,
            }

        if chart_type == ChartType.TABLE:
            return {
                "headers": cols,
                "rows":    [[str(r.get(c, "")) for c in cols] for r in rows[:100]],
                "total":   len(rows),
            }

        if chart_type == ChartType.SPARKLINE:
            return {
                "type":   "sparkline",
                "values": values[:20],
                "labels": labels[:20],
                "label":  value_col,
            }

        if chart_type == ChartType.PIVOT:
            return {
                "type":    "pivot",
                "headers": cols,
                "rows":    [{c: r.get(c, "") for c in cols} for r in rows[:100]],
                "label_col": label_col,
                "value_cols": num_cols,
            }

        # ══════════════════════════════════════════════════════
        # ── HEATMAP ───────────────────────────────────────────
        # Structure : matrice row_label × col_label → valeur
        # ══════════════════════════════════════════════════════
        if chart_type == ChartType.HEATMAP:
            # Besoin de 2 dimensions catégorielles + 1 métrique
            if len(cat_cols) >= 2 and num_cols:
                row_col = cat_cols[0]
                col_col = cat_cols[1]
                val_col = num_cols[0]
                row_labels = sorted(list({str(r.get(row_col, "")) for r in rows}))[:15]
                col_labels = sorted(list({str(r.get(col_col, "")) for r in rows}))[:15]
                # Construire la matrice
                matrix: Dict[str, Dict[str, float]] = {}
                for r in rows:
                    rl = str(r.get(row_col, ""))
                    cl = str(r.get(col_col, ""))
                    v  = float(r.get(val_col, 0) or 0)
                    if rl not in matrix:
                        matrix[rl] = {}
                    matrix[rl][cl] = matrix[rl].get(cl, 0) + v
                return {
                    "type":       "heatmap",
                    "row_labels": row_labels,
                    "col_labels": col_labels,
                    "matrix":     [[matrix.get(rl, {}).get(cl, 0) for cl in col_labels] for rl in row_labels],
                    "value_col":  val_col,
                    "row_col":    row_col,
                    "col_col":    col_col,
                }
            # Fallback si pas assez de dimensions : bar chart data
            return {"labels": labels[:30], "datasets": [{"label": value_col, "data": values[:30],
                    "backgroundColor": self._palette(1)[0]}]}

        # ══════════════════════════════════════════════════════
        # ── FUNNEL ────────────────────────────────────────────
        # Structure : étapes ordonnées avec valeurs décroissantes
        # ══════════════════════════════════════════════════════
        if chart_type == ChartType.FUNNEL:
            funnel_items = []
            total_val = values[0] if values else 1
            for i, (lbl, val) in enumerate(zip(labels[:10], values[:10])):
                pct = (val / total_val * 100) if total_val > 0 else 0
                funnel_items.append({
                    "label":   str(lbl),
                    "value":   val,
                    "pct":     round(pct, 1),
                    "color":   self._palette(10)[i],
                    "drop":    round(100 - pct, 1) if i > 0 else 0,
                })
            return {"type": "funnel", "items": funnel_items, "total": total_val}

        # ══════════════════════════════════════════════════════
        # ── CHOROPLETH ────────────────────────────────────────
        # Données pour carte Plotly choropleth monde
        # ══════════════════════════════════════════════════════
        if chart_type == ChartType.CHOROPLETH:
            # Cherche la colonne pays (ISO ou nom complet)
            geo_col = next(
                (c for c in cat_cols if any(k in c.lower() for k in
                 ["country","countryname","pays","nation","country_code","countrycode","shipcountry"])),
                cat_cols[0] if cat_cols else label_col
            )
            val_col_geo = num_cols[0] if num_cols else value_col
            geo_data = []
            for r in rows[:200]:
                country = str(r.get(geo_col, "") or "")
                val = float(r.get(val_col_geo, 0) or 0)
                if country:
                    geo_data.append({"country": country, "value": val})
            return {
                "type":     "choropleth",
                "data":     geo_data,
                "geo_col":  geo_col,
                "val_col":  val_col_geo,
                "colorscale": "Blues",   # Plotly colorscale
                "title":    value_col.replace("_", " ").title(),
            }

        # ══════════════════════════════════════════════════════
        # ── BUBBLE MAP ───────────────────────────────────────
        # Données pour carte Plotly scattergeo avec bulles
        # ══════════════════════════════════════════════════════
        if chart_type == ChartType.BUBBLE_MAP:
            city_col = next(
                (c for c in cat_cols if any(k in c.lower() for k in ["city","ville","shipcity"])),
                cat_cols[0] if cat_cols else label_col
            )
            country_col = next(
                (c for c in cat_cols if any(k in c.lower() for k in ["country","pays","shipcountry"])),
                None
            )
            val_col_map = num_cols[0] if num_cols else value_col
            map_data = []
            for r in rows[:200]:
                city = str(r.get(city_col, "") or "")
                country = str(r.get(country_col, "") or "") if country_col else ""
                val = float(r.get(val_col_map, 0) or 0)
                if city:
                    map_data.append({"city": city, "country": country, "value": val})
            return {
                "type":        "bubble_map",
                "data":        map_data,
                "city_col":    city_col,
                "country_col": country_col,
                "val_col":     val_col_map,
                "title":       val_col_map.replace("_", " ").title(),
            }

        # ══════════════════════════════════════════════════════
        # ── SANKEY ───────────────────────────────────────────
        # Structure : nœuds source → cible avec poids
        # Nécessite au moins 2 colonnes catégorielles + 1 numérique
        # ══════════════════════════════════════════════════════
        if chart_type == ChartType.SANKEY:
            if len(cat_cols) >= 2 and num_cols:
                src_col  = cat_cols[0]
                tgt_col  = cat_cols[1]
                val_col_sk = num_cols[0]
                # Construire les nœuds et liens
                nodes_set: list = []
                links = []
                for r in rows[:100]:
                    src = str(r.get(src_col, "") or "")
                    tgt = str(r.get(tgt_col, "") or "")
                    val = float(r.get(val_col_sk, 0) or 0)
                    if not src or not tgt or val <= 0:
                        continue
                    if src not in nodes_set:
                        nodes_set.append(src)
                    if tgt not in nodes_set:
                        nodes_set.append(tgt)
                    links.append({
                        "source": nodes_set.index(src),
                        "target": nodes_set.index(tgt),
                        "value":  val,
                        "label":  f"{src} → {tgt}: {val}",
                    })
                colors_sk = self._palette(len(nodes_set))
                return {
                    "type":   "sankey",
                    "nodes":  [{"label": n, "color": colors_sk[i % len(colors_sk)]}
                               for i, n in enumerate(nodes_set)],
                    "links":  links,
                    "src_col":   src_col,
                    "tgt_col":   tgt_col,
                    "val_col":   val_col_sk,
                }
            # Fallback si pas assez de colonnes
            return {"type": "table", "headers": cols, "rows": [[str(r.get(c,"")) for c in cols] for r in rows[:50]], "total": len(rows)}

        # Line, Bar, Bar_H, Area — format standard
        datasets_list = []
        for j, vc in enumerate(num_cols[:3]):
            vals_raw = [float(r.get(vc, 0) or 0) if r.get(vc) is not None else 0.0 for r in rows]
            # If all values are 0, try count column as fallback
            display_lbl = None  # reset par itération
            if all(v==0 for v in vals_raw):
                count_col = next((c for c in cols if c.lower() in ("nb_lignes","count","nb_commandes")), None)
                vals = [float(r.get(count_col,0) or 0) for r in rows] if count_col else vals_raw
                if count_col and vc != count_col:
                    display_lbl = "Nb financements"
            else:
                vals = vals_raw
            color = self._palette(3)[j]
            _d={"total":"Total","montant_total":"Montant total","total_closingbalanceamount":"Solde bancaire (€)","closingbalanceamount":"Solde bancaire (€)","montantavecsigne":"Montant signé (€)","total_amounti":"Montant total (€)","amounti":"Montant (€)","prpcntrvamount":"Montant engagement (€)","total_prpcntrvamount":"Montant total engagements (€)","moyenne":"Moyenne","nb_lignes":"Nb lignes","nb_commandes":"Nb commandes","unitprice":"Prix unitaire","freight":"Frais de port","quantity":"Quantité"}
            dl = display_lbl if display_lbl else _d.get(vc.lower(), vc.replace("_"," ").title())
            ds = {
                "label":            dl,
                "data":             vals,
                "backgroundColor":  color.replace("1)", "0.6)") if "rgba" in color else color,
                "borderColor":      color,
                "borderWidth":      2,
                "fill":             chart_type == ChartType.AREA,
                "tension":          0.4 if chart_type in (ChartType.LINE, ChartType.AREA) else 0,
                "pointRadius":      3 if chart_type == ChartType.LINE else 0,
            }
            if chart_type == ChartType.BAR_H:
                ds["borderRadius"] = 4
            datasets_list.append(ds)

        # tooltip_extra : données contextuelles par point pour tooltips enrichis
        mean_val = sum(values) / len(values) if values else 0
        tooltip_extra = []
        for i, lbl in enumerate(labels[:50]):
            v = values[i] if i < len(values) else 0
            extra = {}
            # % vs N-1 : comparer chaque point au point précédent
            if i > 0:
                prev = values[i-1] if i-1 < len(values) else 0
                if prev != 0:
                    extra["vs_n1"] = round((v - prev) / abs(prev) * 100, 1)
            # % vs moyenne série
            if mean_val != 0:
                extra["vs_mean"] = round((v - mean_val) / abs(mean_val) * 100, 1)
            # nb_commandes si colonne disponible
            if rows and i < len(rows):
                row = rows[i]
                for nb_col in ["nb_commandes", "nb_lignes", "count", "ordercount"]:
                    if nb_col in row:
                        extra["nb_commandes"] = int(row[nb_col] or 0)
                        break
                # panier moyen = valeur / nb_commandes
                if "nb_commandes" in extra and extra["nb_commandes"] > 0 and v > 0:
                    extra["panier_moyen"] = round(v / extra["nb_commandes"], 2)
            tooltip_extra.append(extra)

        result = {
            "labels":        labels[:50],
            "datasets":      datasets_list,
        }
        if any(tooltip_extra):
            result["tooltip_extra"] = tooltip_extra
        return result

    def _build_options(self, chart_type: str, intent: DashboardIntent) -> Dict:
        """Options Chart.js selon le type."""
        base = {
            "responsive":          True,
            "maintainAspectRatio": False,
            "animation":           {"duration": 600},
            "plugins": {
                "legend": {
                    "display":  True,
                    "position": "bottom" if chart_type in (ChartType.PIE, ChartType.DOUGHNUT, ChartType.TREEMAP) else "top",
                    "labels":   {"color": "#7ea4be", "font": {"size": 11}},
                },
                "tooltip": {"mode": "index", "intersect": False},
            },
        }
        if chart_type not in (ChartType.PIE, ChartType.DOUGHNUT, ChartType.SCATTER, ChartType.KPI_CARD, ChartType.TABLE):
            base["scales"] = {
                "x": {
                    "grid":  {"color": "rgba(24,35,54,.5)"},
                    "ticks": {"color": "#7ea4be", "font": {"size": 10}, "maxRotation": 35},
                },
                "y": {
                    "grid":  {"color": "rgba(24,35,54,.5)"},
                    "ticks": {"color": "#7ea4be", "font": {"size": 10}},
                },
            }
        if chart_type == ChartType.BAR_H:
            base["indexAxis"] = "y"
        return base

    def _build_kpi_widget(self, rows, cols, idx):
        """KPI card avec vraie métrique — priorité à montant_total."""
        _ip = ["_id","id_","rowid","productid","orderid","customerid","employeeid"]
        # Priorité absolue : montant_total (résultat JOIN), puis autres métriques
        _priority = ["montant_total","closingbalanceamount","montantavecsigne","amounti","montant","prpcntrvamount","amount","total","revenue","salary","freight","cost","balance","subtotal","net","gross"]
        _metric   = _priority + ["qty","quantity","stock","montant","nb_commandes"]

        # Colonnes numériques non-ID
        num_all = [c for c in cols
                   if rows and isinstance(rows[0].get(c),(int,float,__import__('decimal').Decimal))
                   and not any(p in c.lower() for p in _ip)]
        if not num_all: return None

        # Cherche dans l'ordre de priorité
        col = next((c for c in num_all if c.lower() in _priority), None)
        if not col:
            col = next((c for c in num_all if any(m in c.lower() for m in _metric)), None)
        if not col:
            col = num_all[0]

        from decimal import Decimal as _D
        raw_vals = [float(r.get(col,0) or 0) for r in rows if isinstance(r.get(col),(int,float,_D))]
        total = float(sum(raw_vals))
        avg   = total/len(rows) if rows else 0
        _l = {"montant_total":"Montant total (€)","total_closingbalanceamount":"Solde bancaire (€)","closingbalanceamount":"Solde bancaire (€)","montantavecsigne":"Montant signé (€)","total_amounti":"Montant total (€)","amounti":"Montant (€)","prpcntrvamount":"Montant engagement (€)","total_prpcntrvamount":"Montant total engagements (€)","total":"Total","moyenne":"Moyenne",
              "nb_lignes":"Nb lignes","nb_commandes":"Nb commandes",
              "unitprice":"Prix unitaire","unitsinstock":"En stock",
              "freight":"Frais de port","total_freight":"Frais de port total",
              "amount":"Montant","revenue":"Chiffre d'affaires","salary":"Salaire"}
        h = _l.get(col.lower(), col.replace("_"," ").title())
        c = any(k in col.lower() for k in ["price","amount","revenue","salary","freight","total","balance","montant"])

        # Enrichissement KPI : delta vs moyenne, min, max, tendance
        kpi_data = {"type":"kpi","value":total,"avg":avg,"count":len(rows),"label":h,
                    "format":"currency" if c else "number"}
        if raw_vals:
            mean_val = sum(raw_vals) / len(raw_vals)
            kpi_data["min_val"] = min(raw_vals)
            kpi_data["max_val"] = max(raw_vals)
            # Delta vs moyenne : si la valeur totale dépasse la moyenne * nb_rows de 20%
            if mean_val > 0 and len(raw_vals) > 1:
                # Pour KPI temporel : comparer 1ère moitié vs 2ème moitié
                half = len(raw_vals) // 2
                if half > 0:
                    first_half  = sum(raw_vals[:half]) / half
                    second_half = sum(raw_vals[half:]) / max(1, len(raw_vals) - half)
                    if first_half > 0:
                        delta_pct = round((second_half - first_half) / abs(first_half) * 100, 1)
                        kpi_data["delta"]     = round(second_half - first_half, 2)
                        kpi_data["delta_pct"] = delta_pct
                        kpi_data["previous"]  = round(first_half, 2)
                        kpi_data["trend"]     = "up" if delta_pct > 0 else "down"

        return DashboardWidget(widget_id=f"kpi_{idx}", chart_type=ChartType.KPI_CARD, title=h,
            data=kpi_data, width=3, height="120px")

    def _build_filters(self, cols: List[str], rows: List[Dict]) -> List[Dict]:
        """Génère des filtres dynamiques — uniquement colonnes catégorielles utiles."""
        from decimal import Decimal as _DF
        filters = []
        # Exclure les colonnes numériques (montants, totaux) — pas utiles comme filtres
        _num_kw = ["amount","total","montant","solde","balance","prp","closing","freight","salary","revenue"]
        cat_cols = [
            c for c in cols
            if rows and not isinstance(rows[0].get(c), (int, float, _DF))
            and not any(k in c.lower() for k in _num_kw)
        ]
        for col in cat_cols[:2]:
            unique_vals = sorted(list({str(r.get(col, "")) for r in rows if r.get(col)}))[:20]
            if 2 <= len(unique_vals) <= 15:
                filters.append({
                    "field":   col,
                    "type":    "select",
                    "options": ["Tous"] + sorted(unique_vals),
                })
        return filters

    def _build_global_filters(self, datasets: List[Dict]) -> List[Dict]:
        """Filtres globaux du dashboard."""
        return []

    def _get_dimensions(self, chart_type: str, idx: int, total: int) -> Tuple[int, str]:
        """Retourne width (colonnes sur 12) et height."""
        # Cartes et Sankey prennent toute la largeur
        if chart_type in (ChartType.CHOROPLETH, ChartType.BUBBLE_MAP, ChartType.SANKEY):
            return 12, "450px"
        if chart_type == ChartType.HEATMAP:
            return 12, "380px"
        if chart_type == ChartType.FUNNEL:
            return 6, "380px"
        if chart_type == ChartType.TABLE:
            return 12, "350px"
        if total == 1:
            return 12, "380px"
        if idx == 0:
            return 8, "350px"
        return 6, "300px"

    def _generate_title(self, question: str) -> str:
        q = question.strip()
        # Nettoyer les préfixes de commande
        q = re.sub(r"^(génère|montre|affiche|donne|crée|show|create|generate|fais)\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^(un|une|le|la|les|des|du)\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^(dashboard|tableau\s+de\s+bord)\s+(des?|du|de\s+la|les|l\'|d\')?\s*", "", q, flags=re.IGNORECASE)
        # Supprimer le bruit des recommandations enchaînées
        # Ex: "évolution trimestrielle évolution mensuelle détail anomalies baisses répartition..."
        # → garder seulement la partie significative avant les mots "détail/anomalie/répartition/baisses"
        _noise = r"\s+(détail\s+anomalies?|anomalies?\s+baisses?|répartition\s+dashboard|baisses?\s+répartition|dashboard\s+top|complet\s+top).*$"
        q_clean = re.sub(_noise, "", q, flags=re.IGNORECASE).strip()
        if q_clean and len(q_clean) > 6:
            q = q_clean
        # Tronquer à 50 chars max pour la sidebar
        q = q[:50].strip()
        return q[:1].upper() + q[1:] if q else "Dashboard"

    def _palette(self, n: int, alpha: float = 0.85) -> List[str]:
        """Palette harmonieuse OnePilot — couleurs soft cohérentes avec le thème."""
        colors = [
            (0,   180, 230),   # cyan accent brand
            (99,  179, 237),   # bleu ciel
            (72,  149, 239),   # bleu moyen
            (116, 198, 157),   # vert sauge
            (246, 173,  85),   # ambre doux
            (252, 129, 129),   # corail
            (183, 148, 246),   # lavande
            (76,  201, 175),   # turquoise
            (246, 211, 101),   # jaune miel
            (159, 207, 255),   # bleu pâle
            (255, 159, 164),   # rose pêche
            (130, 204, 221),   # bleu ardoise
        ]
        return [f"rgba({colors[i%len(colors)][0]},{colors[i%len(colors)][1]},{colors[i%len(colors)][2]},{alpha})"
                for i in range(n)]


# ══════════════════════════════════════════════════════════════
# DASHBOARD GENERATOR — point d'entrée principal
# ══════════════════════════════════════════════════════════════

class DashboardGenerator:
    """
    Orchestrateur principal du Dashboard Engine.
    Prend une question NL + source_id → retourne un DashboardSpec.
    """

    def __init__(self):
        self.intent_extractor = DashboardIntentExtractor()
        self.designer         = DashboardDesigner()

    async def generate(
        self,
        question:   str,
        slots,                          # QuerySlots
        schema:     Dict[str, List[str]],
        source_id:  str,
        pg_pool,
        redis=None,
        connector_factory=None,         # Passed from main.py to avoid import issues
    ) -> DashboardSpec:
        """
        Pipeline complet : NLU → SQL → Data → Spec.
        """
        from .query_engine import SQLGenerator

        t0 = time.time()

        # 1. Extraction intent dashboard
        intent = self.intent_extractor.extract(question, slots, schema)

        # 2. Génération des requêtes SQL
        sql_gen = SQLGenerator()
        datasets = []

        # ── JOINs prioritaires : fournisseurs et employés ────────────────────
        # Ces JOINs doivent s'exécuter AVANT _find_relevant_tables pour éviter
        # que le routing générique ne route vers Products/Orders avec mauvaises colonnes
        import unicodedata as _ucd
        q_lower_early = _ucd.normalize("NFD", question.lower()).encode("ascii","ignore").decode()

        def _find_tbl(hints):
            for t in schema:
                if any(h in t.lower().replace(" ","").replace("_","") for h in hints): return t
            return None

        # ── Fournisseurs ─────────────────────────────────────────────────────
        if any(k in q_lower_early for k in ["fournisseur","supplier","vendor"]):
            sup_tbl  = _find_tbl(["suppliers","supplier","bpsupplier"])
            prod_tbl = _find_tbl(["products","product","itmmaster"])
            if sup_tbl and prod_tbl:
                sf = schema.get(sup_tbl, [])
                pf = schema.get(prod_tbl, [])
                sup_id   = next((f for f in sf if "supplierid" in f.lower()), None)
                sup_name = next((f for f in sf if "companyname" in f.lower() or ("name" in f.lower() and "id" not in f.lower())), None)
                prod_sup = next((f for f in pf if "supplierid" in f.lower()), None)
                if sup_id and sup_name and prod_sup:
                    top_n = intent.top_n or 10
                    sql_sup = (
                        f"SELECT TOP {top_n} s.[{sup_name}] AS Fournisseur,\n"
                        f"  COUNT(*) AS Nb_Produits\n"
                        f"FROM [{sup_tbl}] s WITH(NOLOCK)\n"
                        f"JOIN [{prod_tbl}] p WITH(NOLOCK) ON s.[{sup_id}] = p.[{prod_sup}]\n"
                        f"GROUP BY s.[{sup_name}]\n"
                        f"ORDER BY Nb_Produits DESC"
                    )
                    rows_s, cols_s = await self._execute_sql(sql_sup, source_id, pg_pool, connector_factory)
                    if rows_s:
                        datasets.append({"sql": sql_sup, "rows": rows_s, "columns": cols_s, "title": f"Top {top_n} fournisseurs"})
                        logger.info(f"[Dashboard] Supplier JOIN early: {len(rows_s)} rows")

        # ── Employés ─────────────────────────────────────────────────────────
        if any(k in q_lower_early for k in ["employe","employee","staff","personnel"]):
            emp_tbl = _find_tbl(["employees","employee","empl"])
            ord_tbl = _find_tbl(["orders","order","sorder","commandes"])
            if emp_tbl and ord_tbl:
                ef = schema.get(emp_tbl, [])
                of = schema.get(ord_tbl, [])
                emp_id  = next((f for f in ef if "employeeid" in f.lower()), None)
                emp_fn  = next((f for f in ef if "firstname" in f.lower()), None)
                emp_ln  = next((f for f in ef if "lastname" in f.lower()), None)
                ord_emp = next((f for f in of if "employeeid" in f.lower()), None)
                ord_id  = next((f for f in of if "orderid" in f.lower()), None)
                if emp_id and (emp_fn or emp_ln) and ord_emp and ord_id:
                    name_expr = (
                        f"s.[{emp_fn}] + ' ' + s.[{emp_ln}]" if emp_fn and emp_ln
                        else f"s.[{emp_fn or emp_ln}]"
                    )
                    top_n = intent.top_n or 10
                    sql_emp = (
                        f"SELECT TOP {top_n} {name_expr} AS Employé,\n"
                        f"  COUNT(o.[{ord_id}]) AS Nb_Commandes\n"
                        f"FROM [{emp_tbl}] s WITH(NOLOCK)\n"
                        f"JOIN [{ord_tbl}] o WITH(NOLOCK) ON s.[{emp_id}] = o.[{ord_emp}]\n"
                        f"GROUP BY {name_expr}\n"
                        f"ORDER BY Nb_Commandes DESC"
                    )
                    rows_e, cols_e = await self._execute_sql(sql_emp, source_id, pg_pool, connector_factory)
                    if rows_e:
                        datasets.append({"sql": sql_emp, "rows": rows_e, "columns": cols_e, "title": f"Top {top_n} employés"})
                        logger.info(f"[Dashboard] Employee JOIN early: {len(rows_e)} rows")

        # ── Produits par revenus (early JOIN Order Details + Products) ────────
        if any(k in q_lower_early for k in ["produit","product","revenu","revenue","top"]) and not datasets:
            prod_tbl = next((t for t in schema if t.lower().replace(" ","").replace("_","") in ("products","product","itmmaster")), None)
            od_tbl   = next((t for t in schema if t.lower().replace(" ","").replace("_","") in ("orderdetails","orderdetail","order details")), None)
            if prod_tbl and od_tbl:
                pf = schema.get(prod_tbl, [])
                of = schema.get(od_tbl, [])
                if not pf:
                    _, pf = await self._execute_sql(f"SELECT TOP 1 * FROM [{prod_tbl}] WITH(NOLOCK)", source_id, pg_pool, connector_factory)
                    if pf: schema[prod_tbl] = pf
                if not of:
                    _, of = await self._execute_sql(f"SELECT TOP 1 * FROM [{od_tbl}] WITH(NOLOCK)", source_id, pg_pool, connector_factory)
                    if of: schema[od_tbl] = of
                prod_id   = next((f for f in pf if "productid" in f.lower()), None)
                prod_name = next((f for f in pf if "productname" in f.lower()), None)
                od_pid    = next((f for f in of if "productid" in f.lower()), None)
                od_price  = next((f for f in of if "unitprice" in f.lower()), None)
                od_qty    = next((f for f in of if "quantity" in f.lower()), None)
                if prod_id and prod_name and od_pid and od_price and od_qty:
                    top_n = intent.top_n or 10
                    sql_prod = (
                        f"SELECT TOP {top_n} p.[{prod_name}] AS Produit,\n"
                        f"  CAST(SUM(od.[{od_price}] * od.[{od_qty}]) AS FLOAT) AS Revenus,\n"
                        f"  SUM(od.[{od_qty}]) AS Quantite\n"
                        f"FROM [{prod_tbl}] p WITH(NOLOCK)\n"
                        f"JOIN [{od_tbl}] od WITH(NOLOCK) ON p.[{prod_id}] = od.[{od_pid}]\n"
                        f"GROUP BY p.[{prod_name}]\n"
                        f"ORDER BY Revenus DESC"
                    )
                    rows_p, cols_p = await self._execute_sql(sql_prod, source_id, pg_pool, connector_factory)
                    if rows_p:
                        datasets.append({"sql": sql_prod, "rows": rows_p, "columns": cols_p, "title": f"Top {top_n} produits par revenus"})
                        logger.info(f"[Dashboard] Product JOIN early: {len(rows_p)} rows")

        # Si les JOINs prioritaires ont trouvé des données → skip _find_relevant_tables
        if datasets:
            spec = self.designer.design(question, intent, datasets, source_id)
            spec.duration_ms = int((time.time() - t0) * 1000)
            try:
                spec = enrich_spec_sprint3(spec, intent)
            except Exception as _e3:
                logger.warning(f"[Dashboard] Sprint3 enrich error: {_e3}")
            if redis:
                try:
                    cache_key = f"onepilot:dashboard:{source_id}:{hash(question)}"
                    await redis.setex(cache_key, 300, json.dumps(spec.to_dict(), default=str))
                except Exception:
                    pass
            logger.info(f"[Dashboard] Généré '{spec.title}' — {len(spec.widgets)} widgets, {spec.duration_ms}ms")
            return spec

        # Détermine les tables à visualiser
        if slots.table_names:
            tables_to_query = slots.table_names[:3]
        else:
            # Cherche les tables les plus pertinentes via pg_pool
            tables_to_query = await self._find_relevant_tables(
                question, schema, source_id, pg_pool
            )

        logger.info(f"[Dashboard] tables_to_query={tables_to_query[:3]}, schema_keys={list(schema.keys())[:5]}")

        # Détecte si on peut faire un JOIN enrichi
        def _find_join_partner(tbl: str, schema: dict) -> Optional[str]:
            """Pour les tables de détail sans date, cherche une table parente avec date."""
            tbl_n = tbl.lower().replace(" ","").replace("_","")
            if "orderdetail" in tbl_n:
                for t in schema:
                    if t.lower().replace(" ","").replace("_","") in ("orders","order","commandes","sorder"):
                        return t
            return None

        # ── JOIN spécial : ventes par catégorie ──────────────────────────────
        # Si la question contient "catégorie" → JOIN Categories+Products+OrderDetails
        q_lower_join = question.lower()
        is_category_q = any(k in q_lower_join for k in ["catégorie","categorie","category","par catég"])
        if is_category_q:
            # Cherche les tables nécessaires dans le schéma
            def _find_table(hints):
                for t in schema:
                    if any(h in t.lower().replace(" ","").replace("_","") for h in hints):
                        return t
                return None
            cat_tbl = _find_table(["categories","categorie","category"])
            prod_tbl = _find_table(["products","produit","product","itmmaster"])
            od_tbl = _find_table(["orderdetail","lignecommande","detailcommande"])

            if cat_tbl and prod_tbl and od_tbl:
                # Trouver les colonnes clés
                cat_fields  = schema.get(cat_tbl, [])
                prod_fields = schema.get(prod_tbl, [])
                od_fields   = schema.get(od_tbl, [])

                cat_id  = next((f for f in cat_fields if "categoryid" in f.lower() or f.lower()=="id"), None)
                cat_nm  = next((f for f in cat_fields if "categoryname" in f.lower() or "name" in f.lower()), None)
                prod_cat= next((f for f in prod_fields if "categoryid" in f.lower()), None)
                prod_id = next((f for f in prod_fields if "productid" in f.lower() or f.lower()=="id"), None)
                od_prod = next((f for f in od_fields if "productid" in f.lower()), None)
                od_price= next((f for f in od_fields if "unitprice" in f.lower()), None)
                od_qty  = next((f for f in od_fields if "quantity" in f.lower() or f.lower()=="qty"), None)
                od_disc = next((f for f in od_fields if "discount" in f.lower()), None)

                if cat_id and cat_nm and prod_cat and prod_id and od_prod and od_price and od_qty:
                    metric = (f"CAST(SUM(od.[{od_price}]*(1-ISNULL(od.[{od_disc}],0))*od.[{od_qty}]) AS FLOAT)"
                              if od_disc else f"CAST(SUM(od.[{od_price}]*od.[{od_qty}]) AS FLOAT)")
                    sql_cat = (
                        f"SELECT TOP 20\n"
                        f"  c.[{cat_nm}] AS Catégorie,\n"
                        f"  {metric} AS Ventes,\n"
                        f"  COUNT(*) AS Nb_Commandes\n"
                        f"FROM [{od_tbl}] od WITH(NOLOCK)\n"
                        f"JOIN [{prod_tbl}] p WITH(NOLOCK) ON od.[{od_prod}] = p.[{prod_id}]\n"
                        f"JOIN [{cat_tbl}] c WITH(NOLOCK) ON p.[{prod_cat}] = c.[{cat_id}]\n"
                        f"GROUP BY c.[{cat_nm}]\n"
                        f"ORDER BY Ventes DESC"
                    )
                    logger.info(f"[Dashboard] Category JOIN SQL: {sql_cat[:100]}")
                    rows_cat, cols_cat = await self._execute_sql(sql_cat, source_id, pg_pool, connector_factory)
                    if rows_cat:
                        datasets.append({"sql": sql_cat, "rows": rows_cat, "columns": cols_cat, "title": "Ventes par catégorie"})
                        logger.info(f"[Dashboard] Category JOIN: {len(rows_cat)} rows")
                        # Si on a les données catégorie, pas besoin des autres tables
                        spec = self.designer.design(question, intent, datasets, source_id)
                        spec.duration_ms = int((time.time() - t0) * 1000)
                        if redis:
                            try:
                                cache_key = f"onepilot:dashboard:{source_id}:{hash(question)}"
                                await redis.setex(cache_key, 300, json.dumps(spec.to_dict(), default=str))
                            except Exception: pass
                        logger.info(f"[Dashboard] Généré '{spec.title}' — {len(spec.widgets)} widgets, {spec.duration_ms}ms")
                        try:
                            spec = enrich_spec_sprint3(spec, intent)
                        except Exception as _e3:
                            logger.warning(f"[Dashboard] Sprint3 enrich error: {_e3}")
                        return spec

        # ── JOIN spécial : top clients par chiffre d'affaires ───────────────
        # Normaliser la question pour la détection CA
        q_lower_norm = q_lower_join.replace("’","'").replace("‘","'")
        is_top_client_ca = (
            any(k in q_lower_norm for k in [
                "chiffre d'affaire","chiffre d affaire","chiffredaffaire",
                "chiffre d’affaire","ca","revenue","revenu","ventes"
            ]) and
            any(k in q_lower_norm for k in ["client","customer","clients","customers"]) and
            any(k in q_lower_norm for k in ["top","classement","meilleur","premier"])
        )
        if is_top_client_ca and not is_category_q:
            def _find_table_ca(hints_exact, hints_contains):
                """Cherche d'abord une correspondance exacte, puis contains."""
                # Priorité 1 : correspondance exacte normalisée
                for t in schema:
                    tn = t.lower().replace(" ","").replace("_","")
                    if tn in [h.lower().replace(" ","").replace("_","") for h in hints_exact]:
                        return t
                # Priorité 2 : contains
                for t in schema:
                    tn = t.lower().replace(" ","").replace("_","")
                    if any(h in tn for h in hints_contains):
                        return t
                return None

            # Customers : exact "customers" ou "customer" seul (pas CustomerCustomerDemo)
            cust_tbl = _find_table_ca(
                ["customers","customer","clients"],
                ["customers","bpcustomer"]
            )
            # Orders : exact "orders" ou "order" — PAS "order details"
            ord_tbl = _find_table_ca(
                ["orders","order","commandes"],
                ["orders","sorder","commandevente"]
            )
            # Order Details : exact "orderdetails" ou "order details"
            od_tbl2 = _find_table_ca(
                ["orderdetails","order details","orderdetail","lignescommande"],
                ["orderdetail","lignecommande","detailcommande","soilv","soinv"]
            )
            logger.info(f"[Dashboard] Top CA detection: cust={cust_tbl}, ord={ord_tbl}, od={od_tbl2}")

            if cust_tbl and ord_tbl and od_tbl2:
                cust_fields = schema.get(cust_tbl, [])
                ord_fields  = schema.get(ord_tbl, [])
                od_fields2  = schema.get(od_tbl2, [])

                # Colonnes clés
                cust_id  = next((f for f in cust_fields if "customerid" in f.lower()), None)
                cust_nm  = next((f for f in cust_fields if "companyname" in f.lower() or ("name" in f.lower() and "id" not in f.lower())), None)
                ord_cust = next((f for f in ord_fields  if "customerid" in f.lower()), None)
                ord_id   = next((f for f in ord_fields  if "orderid" in f.lower()), None)
                od_ord   = next((f for f in od_fields2  if "orderid" in f.lower()), None)
                od_price = next((f for f in od_fields2  if "unitprice" in f.lower()), None)
                od_qty   = next((f for f in od_fields2  if "quantity" in f.lower() or f.lower()=="qty"), None)
                od_disc  = next((f for f in od_fields2  if "discount" in f.lower()), None)

                top_n = intent.top_n or 10
                if cust_id and cust_nm and ord_cust and ord_id and od_ord and od_price and od_qty:
                    metric = (f"CAST(SUM(od.[{od_price}]*(1-ISNULL(od.[{od_disc}],0))*od.[{od_qty}]) AS FLOAT)"
                              if od_disc else f"CAST(SUM(od.[{od_price}]*od.[{od_qty}]) AS FLOAT)")
                    sql_ca = (
                        f"SELECT TOP {top_n}\n"
                        f"  c.[{cust_nm}] AS Client,\n"
                        f"  {metric} AS Chiffre_Affaires,\n"
                        f"  COUNT(DISTINCT o.[{ord_id}]) AS Nb_Commandes\n"
                        f"FROM [{cust_tbl}] c WITH(NOLOCK)\n"
                        f"JOIN [{ord_tbl}] o WITH(NOLOCK) ON c.[{cust_id}] = o.[{ord_cust}]\n"
                        f"JOIN [{od_tbl2}] od WITH(NOLOCK) ON o.[{ord_id}] = od.[{od_ord}]\n"
                        f"GROUP BY c.[{cust_nm}]\n"
                        f"ORDER BY Chiffre_Affaires DESC"
                    )
                    logger.info(f"[Dashboard] Top clients CA JOIN SQL: {sql_ca[:120]}")
                    rows_ca, cols_ca = await self._execute_sql(sql_ca, source_id, pg_pool, connector_factory)
                    if rows_ca:
                        datasets.append({"sql": sql_ca, "rows": rows_ca, "columns": cols_ca, "title": f"Top {top_n} clients par CA"})
                        logger.info(f"[Dashboard] Top clients CA: {len(rows_ca)} rows")
                        spec = self.designer.design(question, intent, datasets, source_id)
                        spec.duration_ms = int((time.time() - t0) * 1000)
                        if redis:
                            try:
                                cache_key = f"onepilot:dashboard:{source_id}:{hash(question)}"
                                await redis.setex(cache_key, 300, json.dumps(spec.to_dict(), default=str))
                            except Exception: pass
                        logger.info(f"[Dashboard] Généré '{spec.title}' — {len(spec.widgets)} widgets, {spec.duration_ms}ms")
                        try:
                            spec = enrich_spec_sprint3(spec, intent)
                        except Exception as _e3:
                            logger.warning(f"[Dashboard] Sprint3 enrich error: {_e3}")
                        return spec

        # ── JOIN spécial : fournisseurs par nombre de produits ───────────────
        is_supplier_q = any(k in q_lower_join for k in ["fournisseur","supplier","vendors"])
        if is_supplier_q and not datasets:
            def _ft(hints):
                for t in schema:
                    if any(h in t.lower().replace(" ","").replace("_","") for h in hints): return t
                return None
            sup_tbl  = _ft(["suppliers","supplier","bpsupplier"])
            prod_tbl = _ft(["products","product","itmmaster"])
            if sup_tbl and prod_tbl:
                sf = schema.get(sup_tbl, [])
                pf = schema.get(prod_tbl, [])
                sup_id   = next((f for f in sf if "supplierid" in f.lower()), None)
                sup_name = next((f for f in sf if "companyname" in f.lower() or ("name" in f.lower() and "id" not in f.lower())), None)
                prod_sup = next((f for f in pf if "supplierid" in f.lower()), None)
                if sup_id and sup_name and prod_sup:
                    top_n = intent.top_n or 10
                    sql_sup = (
                        f"SELECT TOP {top_n} s.[{sup_name}] AS Fournisseur,\n"
                        f"  COUNT(p.[{prod_sup}]) AS Nb_Produits\n"
                        f"FROM [{sup_tbl}] s WITH(NOLOCK)\n"
                        f"JOIN [{prod_tbl}] p WITH(NOLOCK) ON s.[{sup_id}] = p.[{prod_sup}]\n"
                        f"GROUP BY s.[{sup_name}]\n"
                        f"ORDER BY Nb_Produits DESC"
                    )
                    rows_s, cols_s = await self._execute_sql(sql_sup, source_id, pg_pool, connector_factory)
                    if rows_s:
                        datasets.append({"sql": sql_sup, "rows": rows_s, "columns": cols_s, "title": f"Top {top_n} fournisseurs"})
                        logger.info(f"[Dashboard] Supplier JOIN: {len(rows_s)} rows")

        # ── JOIN spécial : employés par nombre de commandes ──────────────────
        is_employee_q = any(k in q_lower_join for k in ["employé","employe","employee","staff"])
        if is_employee_q and not datasets:
            def _ft2(hints):
                for t in schema:
                    if any(h in t.lower().replace(" ","").replace("_","") for h in hints): return t
                return None
            emp_tbl = _ft2(["employees","employee","empl"])
            ord_tbl = _ft2(["orders","order","sorder","commandes"])
            if emp_tbl and ord_tbl:
                ef = schema.get(emp_tbl, [])
                of = schema.get(ord_tbl, [])
                emp_id    = next((f for f in ef if "employeeid" in f.lower() or f.lower()=="id"), None)
                emp_fn    = next((f for f in ef if "firstname" in f.lower()), None)
                emp_ln    = next((f for f in ef if "lastname" in f.lower()), None)
                ord_emp   = next((f for f in of if "employeeid" in f.lower()), None)
                ord_id    = next((f for f in of if "orderid" in f.lower()), None)
                if emp_id and (emp_fn or emp_ln) and ord_emp and ord_id:
                    name_expr = (
                        f"s.[{emp_fn}] + ' ' + s.[{emp_ln}]" if emp_fn and emp_ln
                        else f"s.[{emp_fn or emp_ln}]"
                    )
                    top_n = intent.top_n or 10
                    sql_emp = (
                        f"SELECT TOP {top_n} {name_expr} AS Employé,\n"
                        f"  COUNT(o.[{ord_id}]) AS Nb_Commandes\n"
                        f"FROM [{emp_tbl}] s WITH(NOLOCK)\n"
                        f"JOIN [{ord_tbl}] o WITH(NOLOCK) ON s.[{emp_id}] = o.[{ord_emp}]\n"
                        f"GROUP BY {name_expr}\n"
                        f"ORDER BY Nb_Commandes DESC"
                    )
                    rows_e, cols_e = await self._execute_sql(sql_emp, source_id, pg_pool, connector_factory)
                    if rows_e:
                        datasets.append({"sql": sql_emp, "rows": rows_e, "columns": cols_e, "title": f"Top {top_n} employés"})
                        logger.info(f"[Dashboard] Employee JOIN: {len(rows_e)} rows")

        for tbl in tables_to_query[:3]:
            if tbl not in schema:
                logger.warning(f"[Dashboard] Table '{tbl}' not in schema")
                continue
            try:
                # Vérifier si JOIN enrichi nécessaire (table détail sans colonne date + intent temporel)
                tbl_fields = schema[tbl]
                _dk = ["date","time","period","at","month","year","jour","mois","annee","ordered","shipped"]
                has_date = any(any(k in f.lower() for k in _dk) for f in tbl_fields)

                if intent.is_trend and not has_date:
                    join_tbl = _find_join_partner(tbl, schema)
                    if join_tbl and join_tbl in schema:
                        # Générer SQL JOIN avec table parente pour avoir la date
                        join_fields = schema[join_tbl]
                        date_col = next((f for f in join_fields if any(k in f.lower() for k in _dk)), None)
                        has_price = next((f for f in tbl_fields if f.lower()=="unitprice"), None)
                        has_qty   = next((f for f in tbl_fields if f.lower() in ("quantity","qty")), None)
                        disc      = next((f for f in tbl_fields if f.lower()=="discount"), None)
                        fk_col    = next((f for f in tbl_fields if f.lower() in ("orderid","order_id","commandeid")), None)
                        pk_col    = next((f for f in join_fields if f.lower() in ("orderid","order_id","id","commandeid")), None)

                        if date_col and has_price and has_qty and fk_col and pk_col:
                            metric = (f"CAST(SUM(d.[{has_price}]*(1-ISNULL(d.[{disc}],0))*d.[{has_qty}]) AS FLOAT)"
                                      if disc else f"SUM(d.[{has_price}]*d.[{has_qty}])")
                            # Filtrer sur une année si mentionnée dans la question (ex: "2023")
                            import re as _re_yr
                            _yr = _re_yr.search(r'\b(20\d{2})\b', question)
                            _yr_filter = f"WHERE YEAR(o.[{date_col}]) = {_yr.group(1)}\n" if _yr else ""
                            sql = (
                                f"SELECT TOP 100\n"
                                f"  CONVERT(varchar(7), o.[{date_col}], 120) AS periode,\n"
                                f"  {metric} AS montant_total,\n"
                                f"  COUNT(*) AS nb_commandes\n"
                                f"FROM [{tbl}] d WITH(NOLOCK)\n"
                                f"JOIN [{join_tbl}] o WITH(NOLOCK) ON d.[{fk_col}] = o.[{pk_col}]\n"
                                f"{_yr_filter}"
                                f"GROUP BY CONVERT(varchar(7), o.[{date_col}], 120)\n"
                                f"ORDER BY periode"
                            )
                            logger.info(f"[Dashboard] JOIN SQL for '{tbl}'+'{join_tbl}': {sql[:120]}")
                            rows, cols = await self._execute_sql(sql, source_id, pg_pool, connector_factory)
                            logger.info(f"[Dashboard] JOIN result: {len(rows)} rows, cols={cols}")
                            if rows:
                                datasets.append({
                                    "sql":     sql,
                                    "rows":    rows,
                                    "columns": cols,
                                    "title":   f"{tbl}",
                                })
                            continue  # Ne pas exécuter le SQL simple pour cette table

                # SQL standard
                sql = self._build_dashboard_sql(tbl, schema[tbl], intent, slots)
                logger.info(f"[Dashboard] Executing SQL for '{tbl}': {sql[:100]}")
                rows, cols = await self._execute_sql(sql, source_id, pg_pool, connector_factory)
                logger.info(f"[Dashboard] Result for '{tbl}': {len(rows)} rows, cols={cols[:3]}")

                if rows:
                    datasets.append({
                        "sql":     sql,
                        "rows":    rows,
                        "columns": cols,
                        "title":   tbl,  # Table name as title, not the question
                    })
            except Exception as e:
                logger.warning(f"[Dashboard] SQL error for {tbl}: {e}")

        # 3. Design du dashboard
        # ── Fallback géo : si la question était géo mais 0 datasets ─────────
        if intent.is_geo and not datasets:
            logger.warning(f"[Dashboard] Geo intent but no datasets — trying metric fallback")
            intent.is_geo = False
            intent.is_composition = False
            intent.chart_hint = None
            intent.geo_field = None
            # Nettoyer la question des termes géo pour éviter des SQL invalides
            geo_terms = ["carte","par pays","par ville","par region","par région",
                         "choropleth","map","géographique","geographic","country","city"]
            clean_q = question
            for term in geo_terms:
                clean_q = clean_q.lower().replace(term, "")
            clean_q = clean_q.strip() or question  # fallback si vide

            fallback_tables = await self._find_relevant_tables(
                clean_q, schema, source_id, pg_pool
            )
            for tbl in fallback_tables[:2]:
                if tbl not in schema:
                    continue
                try:
                    sql = self._build_dashboard_sql(tbl, schema[tbl], intent, slots)
                    rows, cols = await self._execute_sql(sql, source_id, pg_pool, connector_factory)
                    if rows:
                        datasets.append({"sql": sql, "rows": rows, "columns": cols, "title": tbl})
                        logger.info(f"[Dashboard] Geo fallback: {tbl} ({len(rows)} rows)")
                except Exception as e:
                    logger.warning(f"[Dashboard] Geo fallback error for {tbl}: {e}")

        spec = self.designer.design(question, intent, datasets, source_id)
        spec.duration_ms = int((time.time() - t0) * 1000)

        # 4. Cache Redis
        if redis and datasets:
            try:
                cache_key = f"onepilot:dashboard:{source_id}:{hash(question)}"
                await redis.setex(cache_key, 300, json.dumps(spec.to_dict(), default=str))
            except Exception:
                pass

        logger.info(f"[Dashboard] Généré '{spec.title}' — {len(spec.widgets)} widgets, {spec.duration_ms}ms")
        # ── Sprint 3 : enrichissement alt_viz + recommandations ──
        try:
            spec = enrich_spec_sprint3(spec, intent)
        except Exception as _e3:
            logger.warning(f"[Dashboard] Sprint3 enrich error: {_e3}")
        return spec
        """SQL optimal pour visualisation avec métrique calculée."""
        _dk=["date","time","period","at","month","year","jour","mois","annee","ordered","shipped","required","datetime","closingbalancedatetime","integration","trndate","rngdate","debut","fin","maturity","maturite","maturité","début","début"]
        _lk=["name","nom","title","company","companyname","productname","customername","categoryname","city","country","region","contact","firstname","lastname","description","banque","societe","société","code","devises","type_transaction","type","état","etat","groupe","société","banque","libelle"]
        _bk=["discontinued","active","enabled","flag"]
        _tk=["quantityperunit","description","notes","picture","homepage","address","phone","fax","email","url","photo"]
        _mk=["price","amount","total","stock","instock","onorder","salary","cost","revenue","montant","freight","reorderlevel","rate","pct","unitprice","unitsinstock","unitsonorder","weight","size","number","quantity","qty","discount","value","valeur","balance","budget","tax","subtotal","net","gross","closingbalanceamount","montantavecsigne","solde","encours","amounti","prpcntrvamount","prpfinlinerate"]
        def _iid(f): fl=f.lower(); return fl in ["id","rowid","seqno","seqnum","reportsto","trn_id","finline_dtls_id","gnrlstatus","acc_id","isdebiti","closingbalancecreditindicator","state"] or (fl.endswith("id") and len(fl)>2 and fl not in ["acid","valid"])
        def _isk(f): fl=f.lower(); return _iid(f) or any(b in fl for b in _bk) or fl in _tk

        # ── Routing géographique prioritaire ─────────────────────────────────
        # Si la question est géographique, on génère un SQL orienté pays/ville
        if intent.is_geo:
            GEO_COL_HINTS = ["country","pays","countryname","shipcountry","city","ville","shipcity","region","cntr"]
            # Priorité : country > city — les colonnes pays donnent de meilleures cartes
            COUNTRY_HINTS = ["country","pays","countryname","shipcountry"]
            CITY_HINTS    = ["city","ville","shipcity"]
            geo_col = (
                next((f for f in fields if any(h in f.lower() for h in COUNTRY_HINTS)), None)
                or next((f for f in fields if any(h in f.lower() for h in CITY_HINTS)), None)
                or next((f for f in fields if any(h in f.lower() for h in ["region","cntr"])), None)
            )
            if geo_col:
                # Cherche une métrique NUMÉRIQUE — exclure la colonne géo elle-même
                # et exclure les colonnes texte (country codes, nvarchar)
                NUMERIC_METRIC_HINTS = ["amount","price","qty","quantity","revenue","total","montant","freight","unitprice","weight","cost","value","valeur","nb","count"]
                metric_col = next(
                    (f for f in fields
                     if any(h in f.lower() for h in NUMERIC_METRIC_HINTS)
                     and f.lower() != geo_col.lower()
                     and not _iid(f)
                     and not any(geo_h in f.lower() for geo_h in GEO_COL_HINTS)),
                    None
                )
                # Toujours utiliser COUNT(*) — SUM ne marche que sur colonnes numériques
                # et on ne peut pas détecter le type SQL depuis les métadonnées
                if metric_col:
                    sql = (
                        f"SELECT TOP 50 [{geo_col}],\n"
                        f"       COUNT(*) AS nb_lignes\n"
                        f"FROM [{table}] WITH(NOLOCK)\n"
                        f"WHERE [{geo_col}] IS NOT NULL AND [{geo_col}] <> ''\n"
                        f"GROUP BY [{geo_col}]\n"
                        f"ORDER BY nb_lignes DESC"
                    )
                else:
                    sql = (
                        f"SELECT TOP 50 [{geo_col}],\n"
                        f"       COUNT(*) AS nb_lignes\n"
                        f"FROM [{table}] WITH(NOLOCK)\n"
                        f"WHERE [{geo_col}] IS NOT NULL AND [{geo_col}] <> ''\n"
                        f"GROUP BY [{geo_col}]\n"
                        f"ORDER BY nb_lignes DESC"
                    )
                logger.info(f"[Dashboard] Geo SQL for {table}.{geo_col}: {sql[:80]}")
                return sql
        nf=[f for f in fields if not _isk(f) and any(k in f.lower() for k in _mk)]
        cf=[f for f in fields if f not in nf and not _isk(f)][:5]
        df=[f for f in fields if any(k in f.lower() for k in _dk)]
        lh=[f for f in cf if any(h in f.lower() for h in _lk)]
        bl=lh[0] if lh else (cf[0] if cf else None)
        _p=["amount","total","revenue","price","unitprice","quantity","qty","salary","freight","balance","cost","subtotal","net","gross","montant","closingbalanceamount","amounti","prpcntrvamount"]
        pn=[f for f in nf if any(p in f.lower() for p in _p)]
        bm=pn[0] if pn else (nf[0] if nf else None)
        hp=next((f for f in fields if f.lower()=="unitprice"),None)
        hq=next((f for f in fields if f.lower() in ("quantity","qty")),None)
        dc=next((f for f in fields if f.lower()=="discount"),None)
        comp=(f"CAST(SUM([{hp}]*(1-ISNULL([{dc}],0))*[{hq}]) AS FLOAT) AS montant_total" if dc else f"CAST(SUM([{hp}]*[{hq}]) AS FLOAT) AS montant_total") if hp and hq else None

        # SXA Treasury: CLOSINGBALANCEAMOUNT = solde, MontantAvecSigne = montant signé
        sxa_amount = next((f for f in fields if f.lower() == "montant"), None) or \
                     next((f for f in fields if f.lower() in ("closingbalanceamount","montantavecsigne","amounti","prpcntrvamount","prpfinlinerate")), None)
        sxa_date   = next((f for f in fields if any(k in f.lower() for k in ("closingbalancedatetime","trndate","rngdate","date","début","fin","maturity","maturit"))), None)
        sxa_label  = next((f for f in fields if any(k in f.lower() for k in ("type_transaction","banque","sociét","societe","état","etat","description","groupe","libelle","groupe_soci","groupe_de_comptes","groupe_societes","devises","code"))), None)
        # Priorité : utiliser sxa_amount comme meilleure métrique si bm n'est pas déjà meilleur
        if sxa_amount and not comp:
            if not bm or sxa_amount.lower() in ("montant","closingbalanceamount","amounti","prpcntrvamount"):
                bm = sxa_amount  # ← FIX : bm pas best_metric
        if sxa_date and not df:
            df = [sxa_date]
        if sxa_label and not bl:
            bl = sxa_label
        gb=None; ob=None; sp=[]
        if intent.is_trend and df:
            d=df[0]; sp.append(f"CONVERT(varchar(7),[{d}],120) AS periode")
            if comp: sp+=[comp,"COUNT(*) AS nb_commandes"]
            elif bm: sp+=[f"CAST(SUM([{bm}]) AS FLOAT) AS total_{bm.lower()}","COUNT(*) AS nb_commandes"]
            else: sp.append("COUNT(*) AS nb_commandes")
            gb=f"CONVERT(varchar(7),[{d}],120)"; ob=gb
        elif bl and (intent.is_composition or intent.is_top_n or intent.is_kpi or slots.group_by):
            gc=slots.group_by if slots.group_by else bl
            # Skip reserved SQL words as group columns
            if gc and gc.lower() in ("type","key","value","name","date","order","group","index","state","status","level","rank"):
                gc = next((f for f in fields if f.lower()=="type_transaction"), gc)
            sp.append(f"[{gc}]")
            if comp: sp+=[comp,"COUNT(*) AS nb_lignes"]
            elif bm: sp+=[f"CAST(SUM([{bm}]) AS FLOAT) AS total",f"CAST(AVG([{bm}]) AS FLOAT) AS moyenne","COUNT(*) AS nb_lignes"]
            else: sp.append("COUNT(*) AS nb_lignes")
            gb=f"[{gc}]"; ob="2 DESC"
        elif bl and (comp or bm):
            sp.append(f"[{bl}]")
            if comp: sp+=[comp,"COUNT(*) AS nb_lignes"]
            else: sp+=[f"CAST(SUM([{bm}]) AS FLOAT) AS total",f"CAST(AVG([{bm}]) AS FLOAT) AS moyenne","COUNT(*) AS nb_lignes"]
            gb=f"[{bl}]"; ob="2 DESC"
        else:
            pf=(lh+nf+cf)[:8] or [f for f in fields if not _isk(f)][:6] or fields[:6]
            sp=[f"[{f}]" for f in pf]
        tn=intent.top_n or (20 if intent.is_top_n else 100)
        sql="SELECT TOP "+str(tn)+" "+", ".join(sp)+"\nFROM ["+table+"] WITH(NOLOCK)"
        if gb: sql+="\nGROUP BY "+gb
        if ob: sql+="\nORDER BY "+ob
        return sql

    def _build_dashboard_sql(self, table, fields, intent, slots):
        """SQL optimal pour visualisation avec métrique calculée."""
        _dk=["date","time","period","at","month","year","jour","mois","annee","ordered","shipped","required","datetime","closingbalancedatetime","integration","trndate","rngdate","debut","fin","maturity","maturite","maturité","début","début"]
        _lk=["name","nom","title","company","companyname","productname","customername","categoryname","city","country","region","contact","firstname","lastname","description","banque","societe","société","code","devises","type_transaction","type","état","etat","groupe","société","banque","libelle"]
        _bk=["discontinued","active","enabled","flag"]
        _tk=["quantityperunit","description","notes","picture","homepage","address","phone","fax","email","url","photo"]
        _mk=["price","amount","total","stock","instock","onorder","salary","cost","revenue","montant","freight","reorderlevel","rate","pct","unitprice","unitsinstock","unitsonorder","weight","size","number","quantity","qty","discount","value","valeur","balance","budget","tax","subtotal","net","gross","closingbalanceamount","montantavecsigne","solde","encours","amounti","prpcntrvamount","prpfinlinerate"]
        def _iid(f): fl=f.lower(); return fl in ["id","rowid","seqno","seqnum","reportsto","trn_id","finline_dtls_id","gnrlstatus","acc_id","isdebiti","closingbalancecreditindicator","state"] or (fl.endswith("id") and len(fl)>2 and fl not in ["acid","valid"])
        def _isk(f): fl=f.lower(); return _iid(f) or any(b in fl for b in _bk) or fl in _tk

        # ── Routing géographique prioritaire ─────────────────────────────────
        # Si la question est géographique, on génère un SQL orienté pays/ville
        if intent.is_geo:
            GEO_COL_HINTS = ["country","pays","countryname","shipcountry","city","ville","shipcity","region","cntr"]
            # Priorité : country > city — les colonnes pays donnent de meilleures cartes
            COUNTRY_HINTS = ["country","pays","countryname","shipcountry"]
            CITY_HINTS    = ["city","ville","shipcity"]
            geo_col = (
                next((f for f in fields if any(h in f.lower() for h in COUNTRY_HINTS)), None)
                or next((f for f in fields if any(h in f.lower() for h in CITY_HINTS)), None)
                or next((f for f in fields if any(h in f.lower() for h in ["region","cntr"])), None)
            )
            if geo_col:
                # Cherche une métrique NUMÉRIQUE — exclure la colonne géo elle-même
                # et exclure les colonnes texte (country codes, nvarchar)
                NUMERIC_METRIC_HINTS = ["amount","price","qty","quantity","revenue","total","montant","freight","unitprice","weight","cost","value","valeur","nb","count"]
                metric_col = next(
                    (f for f in fields
                     if any(h in f.lower() for h in NUMERIC_METRIC_HINTS)
                     and f.lower() != geo_col.lower()
                     and not _iid(f)
                     and not any(geo_h in f.lower() for geo_h in GEO_COL_HINTS)),
                    None
                )
                # Toujours utiliser COUNT(*) — SUM ne marche que sur colonnes numériques
                # et on ne peut pas détecter le type SQL depuis les métadonnées
                if metric_col:
                    sql = (
                        f"SELECT TOP 50 [{geo_col}],\n"
                        f"       COUNT(*) AS nb_lignes\n"
                        f"FROM [{table}] WITH(NOLOCK)\n"
                        f"WHERE [{geo_col}] IS NOT NULL AND [{geo_col}] <> ''\n"
                        f"GROUP BY [{geo_col}]\n"
                        f"ORDER BY nb_lignes DESC"
                    )
                else:
                    sql = (
                        f"SELECT TOP 50 [{geo_col}],\n"
                        f"       COUNT(*) AS nb_lignes\n"
                        f"FROM [{table}] WITH(NOLOCK)\n"
                        f"WHERE [{geo_col}] IS NOT NULL AND [{geo_col}] <> ''\n"
                        f"GROUP BY [{geo_col}]\n"
                        f"ORDER BY nb_lignes DESC"
                    )
                logger.info(f"[Dashboard] Geo SQL for {table}.{geo_col}: {sql[:80]}")
                return sql
        nf=[f for f in fields if not _isk(f) and any(k in f.lower() for k in _mk)]
        cf=[f for f in fields if f not in nf and not _isk(f)][:5]
        df=[f for f in fields if any(k in f.lower() for k in _dk)]
        lh=[f for f in cf if any(h in f.lower() for h in _lk)]
        bl=lh[0] if lh else (cf[0] if cf else None)
        _p=["amount","total","revenue","price","unitprice","quantity","qty","salary","freight","balance","cost","subtotal","net","gross","montant","closingbalanceamount","amounti","prpcntrvamount"]
        pn=[f for f in nf if any(p in f.lower() for p in _p)]
        bm=pn[0] if pn else (nf[0] if nf else None)
        hp=next((f for f in fields if f.lower()=="unitprice"),None)
        hq=next((f for f in fields if f.lower() in ("quantity","qty")),None)
        dc=next((f for f in fields if f.lower()=="discount"),None)
        comp=(f"CAST(SUM([{hp}]*(1-ISNULL([{dc}],0))*[{hq}]) AS FLOAT) AS montant_total" if dc else f"CAST(SUM([{hp}]*[{hq}]) AS FLOAT) AS montant_total") if hp and hq else None

        # SXA Treasury: CLOSINGBALANCEAMOUNT = solde, MontantAvecSigne = montant signé
        sxa_amount = next((f for f in fields if f.lower() == "montant"), None) or \
                     next((f for f in fields if f.lower() in ("closingbalanceamount","montantavecsigne","amounti","prpcntrvamount","prpfinlinerate")), None)
        sxa_date   = next((f for f in fields if any(k in f.lower() for k in ("closingbalancedatetime","trndate","rngdate","date","début","fin","maturity","maturit"))), None)
        sxa_label  = next((f for f in fields if any(k in f.lower() for k in ("type_transaction","banque","sociét","societe","état","etat","description","groupe","libelle","groupe_soci","groupe_de_comptes","groupe_societes","devises","code"))), None)
        # Priorité : utiliser sxa_amount comme meilleure métrique si bm n'est pas déjà meilleur
        if sxa_amount and not comp:
            if not bm or sxa_amount.lower() in ("montant","closingbalanceamount","amounti","prpcntrvamount"):
                bm = sxa_amount  # ← FIX : bm pas best_metric
        if sxa_date and not df:
            df = [sxa_date]
        if sxa_label and not bl:
            bl = sxa_label
        gb=None; ob=None; sp=[]
        if intent.is_trend and df:
            d=df[0]; sp.append(f"CONVERT(varchar(7),[{d}],120) AS periode")
            if comp: sp+=[comp,"COUNT(*) AS nb_commandes"]
            elif bm: sp+=[f"CAST(SUM([{bm}]) AS FLOAT) AS total_{bm.lower()}","COUNT(*) AS nb_commandes"]
            else: sp.append("COUNT(*) AS nb_commandes")
            gb=f"CONVERT(varchar(7),[{d}],120)"; ob=gb
        elif bl and (intent.is_composition or intent.is_top_n or intent.is_kpi or slots.group_by):
            gc=slots.group_by if slots.group_by else bl
            # Skip reserved SQL words as group columns
            if gc and gc.lower() in ("type","key","value","name","date","order","group","index","state","status","level","rank"):
                gc = next((f for f in fields if f.lower()=="type_transaction"), gc)
            sp.append(f"[{gc}]")
            if comp: sp+=[comp,"COUNT(*) AS nb_lignes"]
            elif bm: sp+=[f"CAST(SUM([{bm}]) AS FLOAT) AS total",f"CAST(AVG([{bm}]) AS FLOAT) AS moyenne","COUNT(*) AS nb_lignes"]
            else: sp.append("COUNT(*) AS nb_lignes")
            gb=f"[{gc}]"; ob="2 DESC"
        elif bl and (comp or bm):
            sp.append(f"[{bl}]")
            if comp: sp+=[comp,"COUNT(*) AS nb_lignes"]
            else: sp+=[f"CAST(SUM([{bm}]) AS FLOAT) AS total",f"CAST(AVG([{bm}]) AS FLOAT) AS moyenne","COUNT(*) AS nb_lignes"]
            gb=f"[{bl}]"; ob="2 DESC"
        else:
            pf=(lh+nf+cf)[:8] or [f for f in fields if not _isk(f)][:6] or fields[:6]
            sp=[f"[{f}]" for f in pf]
        tn=intent.top_n or (20 if intent.is_top_n else 100)
        sql="SELECT TOP "+str(tn)+" "+", ".join(sp)+"\nFROM ["+table+"] WITH(NOLOCK)"
        if gb: sql+="\nGROUP BY "+gb
        if ob: sql+="\nORDER BY "+ob
        return sql

    async def _execute_sql_raw(
        self,
        sql: str,
        source_id: str,
        pg_pool,
        connector_factory=None,
    ) -> Tuple[List[Dict], List[str]]:
        """Exécution SQL brute sans limite de lignes — pour discovery queries."""
        return await self._execute_sql(sql, source_id, pg_pool, connector_factory)

    async def _execute_sql(
        self,
        sql:       str,
        source_id: str,
        pg_pool,
        connector_factory=None,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Exécute le SQL directement.
        Priorité : ConnectorFactory (injecté depuis main.py) → pyodbc direct → import fallback.
        """
        import asyncio as _aio
        import uuid as _uuid

        try:
            # ── Méthode 1 : ConnectorFactory (voie principale) ────────────
            if connector_factory is not None:
                try:
                    # Récupère la source pour avoir les credentials
                    import importlib as _il
                    m = _il.import_module("api.main")
                    src_obj = await m.get_source(_uuid.UUID(source_id))
                    if not src_obj:
                        logger.warning(f"[Dashboard] Source {source_id} not found")
                        return [], []

                    src = src_obj.model_dump()
                    # Charge le mot de passe depuis connection_secrets
                    pwd = await self._load_password(source_id, pg_pool)
                    if pwd:
                        src["password"] = pwd
                        src["_password"] = pwd

                    loop = _aio.get_event_loop()
                    connector = connector_factory.create(src)

                    # Exécution dans un thread pour ne pas bloquer l'event loop
                    rows = await loop.run_in_executor(None, connector.execute_query, sql)

                    logger.info(f"[Dashboard] ConnectorFactory result: {len(rows) if rows else 0} rows for SQL: {sql[:80]}")
                    if rows and isinstance(rows[0], dict):
                        return rows[:200], list(rows[0].keys())
                    # OData peut retourner une liste de dicts sans être vide mais avec structure différente
                    if rows and isinstance(rows, list) and len(rows) > 0:
                        if isinstance(rows[0], dict):
                            return rows[:200], list(rows[0].keys())
                    return [], []

                except Exception as e_cf:
                    logger.warning(f"[Dashboard] ConnectorFactory error: {e_cf} — trying pyodbc direct")
                    # Fallback vers pyodbc direct si ConnectorFactory échoue

            # ── Méthode 2 : pyodbc/psycopg2 direct ───────────────────────
            import importlib as _il
            m = _il.import_module("api.main")
            src_obj = await m.get_source(_uuid.UUID(source_id))
            if not src_obj:
                logger.warning(f"[Dashboard] Source {source_id} not found (direct method)")
                return [], []

            src = src_obj.model_dump()
            connector_type = str(src.get("connector_type", "")).lower()

            pwd = await self._load_password(source_id, pg_pool)
            if pwd:
                src["_password"] = pwd

            loop = _aio.get_event_loop()

            if any(k in connector_type for k in ["mssql", "sage_100", "sage_x3"]):
                rows = await loop.run_in_executor(None, self._run_mssql, sql, src)
            elif any(k in connector_type for k in ["postgresql", "postgres"]):
                rows = await loop.run_in_executor(None, self._run_postgres, sql, src)
            elif any(k in connector_type for k in ["odata", "api_rest", "rest"]):
                # OData : pas de SQL direct → utiliser execute_query qui traduit SQL→OData
                if connector_factory:
                    try:
                        connector = connector_factory.create(src)
                        # Essai 1 : execute_query standard (certains connecteurs OData supportent SQL)
                        try:
                            rows = await loop.run_in_executor(None, connector.execute_query, sql)
                            if rows and isinstance(rows[0], dict):
                                logger.info(f"[Dashboard] OData execute_query: {len(rows)} rows")
                                return rows[:200], list(rows[0].keys())
                        except Exception as e1:
                            logger.warning(f"[Dashboard] OData execute_query failed: {e1}")

                        # Essai 2 : extraire le nom de table du SQL et faire un fetch OData direct
                        import re as _re
                        tbl_match = _re.search(r'FROM\s+\[?(\w[\w\s]*\w|\w+)\]?', sql, _re.IGNORECASE)
                        if tbl_match:
                            tbl_name = tbl_match.group(1).strip()
                            # Essayer get_data ou fetch_entity selon le connecteur
                            for method_name in ['get_data', 'fetch_entity', 'get_entities', 'query']:
                                method = getattr(connector, method_name, None)
                                if method:
                                    try:
                                        rows = await loop.run_in_executor(None, method, tbl_name)
                                        if rows and isinstance(rows[0], dict):
                                            logger.info(f"[Dashboard] OData {method_name}('{tbl_name}'): {len(rows)} rows")
                                            return rows[:200], list(rows[0].keys())
                                    except Exception as e2:
                                        logger.warning(f"[Dashboard] OData {method_name} failed: {e2}")
                                        continue
                        logger.warning(f"[Dashboard] OData: all methods failed")
                    except Exception as e_odata:
                        logger.warning(f"[Dashboard] OData connector error: {e_odata}")
                return [], []
            else:
                logger.warning(f"[Dashboard] Unsupported connector type: {connector_type}")
                return [], []

            if rows and isinstance(rows[0], dict):
                return rows[:200], list(rows[0].keys())
            return [], []

        except Exception as e:
            logger.warning(f"[Dashboard] _execute_sql error: {e}")
            return [], []

    def _get_password(self, src: Dict) -> str:
        """Extrait le mot de passe depuis les champs JSON ou connection_secrets."""
        import json as _j
        # 1. Essai champs JSON directs
        for key in ["secrets_json", "config_json", "credentials_json"]:
            val = src.get(key)
            if not val:
                continue
            if isinstance(val, str):
                try: val = _j.loads(val)
                except: continue
            if isinstance(val, dict):
                pwd = val.get("password") or val.get("db_password") or val.get("pwd", "")
                if pwd:
                    return str(pwd)
        # 2. Champ direct
        if src.get("password"):
            return str(src["password"])
        # 3. Depuis _cached_password (injecté par _load_password)
        if src.get("_password"):
            return str(src["_password"])
        return ""

    async def _load_password(self, source_id: str, pg_pool) -> str:
        """Charge le mot de passe depuis connection_secrets."""
        try:
            import uuid as _uuid
            async with pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT secret_value FROM connection_secrets WHERE source_id=$1 AND secret_key='password'",
                    _uuid.UUID(source_id)
                )
                if row:
                    return row["secret_value"]
        except Exception as e:
            logger.warning(f"[Dashboard] _load_password error: {e}")
        return ""

    def _run_mssql(self, sql: str, src: Dict) -> List[Dict]:
        """Exécute SQL sur MSSQL via pyodbc."""
        try:
            import pyodbc
            host = src.get("host", "")
            port = src.get("port", 1433)
            db   = src.get("database_name", "")
            user = src.get("username", "")
            pwd  = self._get_password(src)
            cs = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={host},{port};DATABASE={db};"
                f"UID={user};PWD={pwd};"
                f"TrustServerCertificate=yes;Encrypt=no;"
            )
            with pyodbc.connect(cs, timeout=15) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                cols = [col[0] for col in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchmany(200)]
        except Exception as e:
            logger.warning(f"[Dashboard] MSSQL exec error: {e}")
            return []

    def _run_postgres(self, sql: str, src: Dict) -> List[Dict]:
        """Exécute SQL sur PostgreSQL via psycopg2."""
        try:
            import psycopg2, psycopg2.extras
            with psycopg2.connect(
                host=src.get("host",""), port=src.get("port",5432),
                dbname=src.get("database_name",""),
                user=src.get("username",""),
                password=self._get_password(src),
                connect_timeout=10
            ) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    return [dict(r) for r in cur.fetchmany(200)]
        except Exception as e:
            logger.warning(f"[Dashboard] PostgreSQL exec error: {e}")
            return []

    async def _find_relevant_tables(
        self,
        question: str,
        schema: Dict[str, List[str]],
        source_id: str,
        pg_pool,
    ) -> List[str]:
        """
        Sélectionne les tables les plus pertinentes pour la question :
        1. Routing géographique (carte, pays, ville) → table avec Country/City
        2. Tables avec le plus de lignes (données réelles)
        3. Tables dont le nom correspond à des mots-clés de la question
        """
        import uuid as _uuid
        q_lower = question.lower()

        def _norm(s): return s.lower().replace(" ","").replace("-","").replace("_","").replace("é","e").replace("è","e").replace("à","a")

        # ── PRIORITÉ 0 : Routing géographique ────────────────────────────────
        # Si la question contient "carte", "pays", "ville", "country", "city"
        # → chercher la table qui a des colonnes géographiques
        is_geo_q = any(k in q_lower for k in [
            "carte","pays","country","ville","city","région","region",
            "géographique","geographic","map","choropleth","bubble map"
        ])
        if is_geo_q:
            GEO_COL_HINTS = ["country","pays","countryname","shipcountry","city","ville","shipcity","region","cntr","pays","nation","localit"]
            METRIC_COL_HINTS = ["amount","price","qty","quantity","revenue","total","montant","freight","unitprice","count","nb","sum"]
            # Cherche une table qui a des colonnes géo dans le schéma indexé
            best_geo_table = None
            best_score = 0
            for tbl, fields in schema.items():
                flds_lower = [f.lower() for f in fields]
                geo_score    = sum(1 for h in GEO_COL_HINTS  if any(h in f for f in flds_lower))
                metric_score = sum(1 for h in METRIC_COL_HINTS if any(h in f for f in flds_lower))
                score = geo_score * 2 + metric_score
                if geo_score > 0 and score > best_score:
                    best_score = score
                    best_geo_table = tbl
            if best_geo_table:
                logger.info(f"[Dashboard] Geo routing (schema) → {best_geo_table} (score={best_score})")
                return [best_geo_table]

            # ── Fallback géo : chercher dans MSSQL Information Schema ──────
            # Le schéma PostgreSQL peut être incomplet (champs non synchronisés)
            logger.warning(f"[Dashboard] No geo table in indexed schema — querying MSSQL sys tables")
            try:
                discovery_sql = """
                    SELECT TOP 5
                        t.name AS table_name,
                        c.name AS column_name
                    FROM sys.tables t
                    JOIN sys.columns c ON c.object_id = t.object_id
                    WHERE (
                        c.name LIKE '%country%' OR c.name LIKE '%COUNTRY%'
                        OR c.name LIKE '%CNTR%' OR c.name LIKE '%cntr%'
                        OR c.name LIKE '%city%'   OR c.name LIKE '%CITY%'
                        OR c.name LIKE '%region%' OR c.name LIKE '%REGION%'
                        OR c.name LIKE '%pays%'   OR c.name LIKE '%ville%'
                    )
                    ORDER BY t.name
                """
                geo_rows, geo_cols = await self._execute_sql_raw(discovery_sql, source_id, pg_pool, connector_factory)
                if geo_rows:
                    # Sélectionne la table avec le plus de colonnes géo
                    from collections import Counter
                    tbl_counts = Counter(r.get("table_name","") for r in geo_rows)
                    best_tbl = tbl_counts.most_common(1)[0][0]
                    # Si cette table est dans le schéma, utilise-la directement
                    if best_tbl in schema:
                        logger.info(f"[Dashboard] Geo routing (sys.columns) → {best_tbl}")
                        return [best_tbl]
                    # Sinon injecte les colonnes découvertes dans le schéma
                    discovered_cols = [r.get("column_name","") for r in geo_rows if r.get("table_name")==best_tbl]
                    if discovered_cols:
                        schema[best_tbl] = discovered_cols  # inject temporaire
                        logger.info(f"[Dashboard] Geo routing (inject) → {best_tbl} cols={discovered_cols}")
                        return [best_tbl]
            except Exception as e_geo:
                logger.warning(f"[Dashboard] Geo sys.columns discovery failed: {e_geo}")

            logger.warning(f"[Dashboard] No geo table found anywhere for: {q_lower[:60]}")

        RICH_FRAGMENTS=["orderdetail","detailcommande","detailvente","lignecommande","lignefacture",
                         "saleline","invoiceline","soilv","soinvoice","soinv"]

        is_amount=any(k in q_lower for k in ["vente","montant","ca","chiffre","évolution","mensuel","revenu","commande","facture",
                                              "tresorerie","trésorerie","solde","bancaire","financement","amortissement","compte","société","société"])
        if is_amount:
            # 1. Tables de lignes de détail ERP (UnitPrice × Quantity)
            for tbl in schema.keys():
                if any(frag in _norm(tbl) for frag in RICH_FRAGMENTS):
                    logger.info(f"[Dashboard] Rich metric table: {tbl}"); return [tbl]

            # 2. Tables SXA trésorerie — ordre de priorité strict (montants réels d'abord)
            SXA_PRIORITY = [
                "dernièreintegrationbancaire","derniereintegrationbancaire",
                "si_bancaire","sibancaire",
                "si_tresorerie","sitresorerie",
                "financement_bi","financementbi",
                "vdtssxaaccountdata","vdtssxaaccountrib",
            ]
            schema_norm = {_norm(k): k for k in schema.keys()}
            KEYWORD_TABLE = {
                "financement": ["financement_bi","financementbi"],
                "amortissement": ["tableauxdamortissement","amortissement"],
                "amortis": ["tableauxdamortissement","amortissement"],
                "mouvement": ["si_bancaire","sibancaire","si_tresorerie","sitresorerie"],
                "transaction": ["si_bancaire","sibancaire"],
            }
            for kw, frags in KEYWORD_TABLE.items():
                if kw in q_lower:
                    for frag in frags:
                        if frag in schema_norm:
                            tbl = schema_norm[frag]
                            logger.info(f"[Dashboard] SXA keyword route '{kw}' → {tbl}")
                            return [tbl]
            for frag in SXA_PRIORITY:
                if frag in schema_norm:
                    tbl = schema_norm[frag]
                    logger.info(f"[Dashboard] SXA treasury table (priority): {tbl}")
                    return [tbl]
            for tbl in schema.keys():
                tn = _norm(tbl)
                if any(f in tn for f in ["tresorerie","bancaire","financement","amort","vdtssxa"]):
                    logger.info(f"[Dashboard] SXA treasury table (fallback): {tbl}"); return [tbl]

        BKW={
            "vente":      ["SO","SOINVOICE","SORDER","INVOICE","FACT","ORDER","ORDERDETAIL","ORDER DETAIL"],
            "commande":   ["SO","ORDER","SORDER","COMMANDE","ORDERDETAIL","ORDER DETAIL"],
            "client":     ["CUSTOMER","CLIENT","BPCUSTOMER","CUS","VDTSSXA"],
            "facture":    ["INVOICE","SOINVOICE","FACT","SINV"],
            "stock":      ["STOCK","ITMMASTER","PRODUCT","ITEM"],
            "produit":    ["PRODUCT","ITEM","ITMMASTER"],
            "catégorie":  ["CATEGORIES","CATEGORY","CAT","PRODUCT"],
            "categorie":  ["CATEGORIES","CATEGORY","CAT","PRODUCT"],
            "category":   ["CATEGORIES","CATEGORY","CAT","PRODUCT"],
            # ── Fournisseurs → Suppliers (Northwind) ───────────────────────
            "fournisseur":["SUPPLIERS","SUPPLIER","BPSUPPLIER","VENDOR"],
            # ── Employés → Employees (Northwind) ──────────────────────────
            "employé":    ["EMPLOYEES","EMPLOYEE","EMPL","STAFF"],
            "employe":    ["EMPLOYEES","EMPLOYEE","EMPL","STAFF"],
            "employee":   ["EMPLOYEES","EMPLOYEE","EMPL","STAFF"],
            "comptabilit":["GACCENTRY","JOURNAL","GL","ACC"],
            "tresorerie": ["SI_T","TRESORERIE","TRS","CASH","VDTSSXA","SI_BANCAIRE","DERNIER"],
            "trésorerie": ["SI_T","TRESORERIE","TRS","VDTSSXA","SI_BANCAIRE"],
            "solde":      ["SI_T","VDTSSXA","COMPTE","SI_BANCAIRE","DERNIER"],
            "bancaire":   ["SI_BANCAIRE","VDTSSXA","DERNIER","COMPTES"],
            "financement":["FINANCEMENT","FIN"],
            "amortissement":["AMORT","TABLEAU"],
            "salaire":    ["PAYROLL","SALARY","EMPL"],
            "compte":     ["COMPTE","VDTSSXAACCOUNT","COMPTES"],
            # ── Northwind CA / revenu → Order Details obligatoire ──────────
            "chiffre":    ["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS","ORDER_DETAIL","ORDER_DETAILS","SOILV","SOINV","LIGNECOMMANDE"],
            "affaires":   ["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS","ORDER_DETAIL","ORDER_DETAILS","SOILV","SOINV","LIGNECOMMANDE"],
            "mensuel":    ["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS","ORDER_DETAIL","ORDER_DETAILS","SOILV","SOINV"],
            "revenue":    ["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS","ORDER_DETAIL","ORDER_DETAILS","SOILV","SOINV"],
            "revenu":     ["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS","ORDER_DETAIL","ORDER_DETAILS","SOILV","SOINV"],
            "top":        ["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS","ORDER_DETAIL","ORDER_DETAILS","CUSTOMER","CUSTOMERS"],
            # ── Entonnoir / bilan / indicateurs / corrélation ─────────────
            "entonnoir":  ["ORDERS","ORDER"],
            "conversion": ["ORDERS","ORDER"],
            "statut":     ["ORDERS","ORDER"],
            "indicateur": ["ORDERS","CUSTOMERS","PRODUCTS"],
            "bilan":      ["ORDERDETAIL","ORDER DETAIL","ORDERS"],
            "corrélation":["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS"],
            "correlation":["ORDERDETAIL","ORDER DETAIL","ORDER DETAILS"],
        }
        ct=[]
        for kw,pf in BKW.items():
            if kw in q_lower:
                for tbl in schema.keys():
                    # Normalise : uppercase, sans espaces, sans underscores
                    tu = tbl.upper().replace(" ","").replace("_","")
                    for p in pf:
                        pn = p.upper().replace(" ","").replace("_","")
                        if (tu.startswith(pn) or pn in tu) and tbl not in ct:
                            ct.append(tbl)
        if ct: logger.info(f"[Dashboard] Keyword-matched: {ct[:2]}"); return ct[:2]
        try:
            async with pg_pool.acquire() as conn:
                rows=await conn.fetch("SELECT name,row_count FROM source_entities WHERE source_id=$1 AND is_visible=TRUE AND row_count>0 ORDER BY row_count DESC LIMIT 4",_uuid.UUID(source_id))
                if rows:
                    top=[r["name"] for r in rows if r["name"] in schema]
                    if top: logger.info(f"[Dashboard] Top tables: {top[:2]}"); return top[:2]
        except Exception as e: logger.warning(f"[Dashboard] _find_relevant_tables error: {e}")
        return list(schema.keys())[:2]

    def _make_title(self, question: str, table: str, intent: DashboardIntent) -> str:
        """Génère un titre lisible pour le widget."""
        # Extrait les mots significatifs de la question
        stop = {"un","une","le","la","les","des","du","de","et","ou","par","sur","dans",
                "dashboard","montre","affiche","génère","crée","donne","show","generate","create"}
        words = [w for w in question.lower().split() if len(w) > 2 and w not in stop]
        q_hint = " ".join(words[:4]).capitalize() if words else ""

        clean = table.replace('_', ' ').title()
        if intent.is_trend:
            return f"Évolution — {q_hint or clean}"
        if intent.is_top_n:
            return f"Top {intent.top_n or 10} — {q_hint or clean}"
        if intent.is_composition:
            return f"Répartition — {q_hint or clean}"
        if intent.is_kpi:
            return f"Indicateurs — {q_hint or clean}"
        return q_hint or clean


# ── Templates pré-configurés ──────────────────────────────────
DASHBOARD_TEMPLATES = [
    {
        "id": "sales_overview",
        "name": "Vue d'ensemble Ventes",
        "description": "KPIs ventes, évolution CA, top clients",
        "icon": "[UP]",
        "keywords": ["ventes", "ca", "chiffre", "commandes", "clients"],
        "widgets": [
            {"type": "kpi_card",  "title": "CA Total",       "metric": "amount"},
            {"type": "bar",       "title": "Top 10 clients",  "group_by": "customer", "top_n": 10},
            {"type": "line",      "title": "Évolution mensuelle", "time_field": True},
            {"type": "pie",       "title": "Répartition produits", "group_by": "category"},
        ]
    },
    {
        "id": "hr_dashboard",
        "name": "Tableau de bord RH",
        "description": "Effectifs, salaires, congés, performance",
        "icon": "👥",
        "keywords": ["rh", "employe", "salaire", "conge", "effectif", "personnel"],
        "widgets": [
            {"type": "kpi_card", "title": "Effectif total",    "metric": "count"},
            {"type": "kpi_card", "title": "Salaire moyen",     "metric": "salary"},
            {"type": "bar",      "title": "Effectifs par dept", "group_by": "department"},
            {"type": "pie",      "title": "Répartition contrats", "group_by": "contract_type"},
        ]
    },
    {
        "id": "finance_dashboard",
        "name": "Tableau de bord Finance",
        "description": "Trésorerie, budgets, écarts, flux financiers",
        "icon": "💰",
        "keywords": ["finance", "tresorerie", "budget", "comptabilit", "flux", "bilan"],
        "widgets": [
            {"type": "kpi_card",  "title": "Solde trésorerie", "metric": "balance"},
            {"type": "waterfall", "title": "Flux de trésorerie", "metric": "amount"},
            {"type": "line",      "title": "Évolution solde",   "time_field": True},
            {"type": "gauge",     "title": "Budget consommé",   "metric": "budget_pct"},
        ]
    },
    {
        "id": "logistics_dashboard",
        "name": "Logistique & Stocks",
        "description": "Niveaux de stock, délais livraison, taux de service",
        "icon": "📦",
        "keywords": ["stock", "logistique", "livraison", "entrepot", "produit", "article"],
        "widgets": [
            {"type": "kpi_card", "title": "Valeur stock",      "metric": "stock_value"},
            {"type": "kpi_card", "title": "Taux de service",   "metric": "service_rate"},
            {"type": "bar",      "title": "Stock par produit",  "group_by": "product"},
            {"type": "treemap",  "title": "Répartition stock",  "group_by": "category"},
        ]
    },
]

def get_dashboard_templates() -> list:
    return DASHBOARD_TEMPLATES

def suggest_template(question: str) -> Optional[dict]:
    """Suggère un template basé sur les mots-clés de la question."""
    q = question.lower()
    for tmpl in DASHBOARD_TEMPLATES:
        if any(kw in q for kw in tmpl["keywords"]):
            return tmpl
    return None

# ── Singleton ─────────────────────────────────────────────────
_dashboard_generator: Optional[DashboardGenerator] = None

def get_dashboard_generator() -> DashboardGenerator:
    global _dashboard_generator
    if _dashboard_generator is None:
        _dashboard_generator = DashboardGenerator()
    return _dashboard_generator


# ══════════════════════════════════════════════════════════════
# SPRINT 3 — SUGGESTIONS DE VISUALISATIONS ALTERNATIVES §2.4.3-G
# ══════════════════════════════════════════════════════════════

# Mapping : type actuel → alternatives intelligentes avec justification
_ALT_VIZ_MAP: Dict[str, List[Dict]] = {
    ChartType.BAR: [
        {"type": ChartType.BAR_H,    "label": "Barres horizontales", "reason": "Meilleur pour les libellés longs"},
        {"type": ChartType.LINE,     "label": "Courbe",               "reason": "Visualiser la tendance"},
        {"type": ChartType.TREEMAP,  "label": "Treemap",              "reason": "Voir la hiérarchie proportionnelle"},
    ],
    ChartType.BAR_H: [
        {"type": ChartType.BAR,      "label": "Barres verticales",    "reason": "Vue classique comparaison"},
        {"type": ChartType.PIE,      "label": "Camembert",            "reason": "Voir les parts du total"},
        {"type": ChartType.TREEMAP,  "label": "Treemap",              "reason": "Proportions en surface"},
    ],
    ChartType.LINE: [
        {"type": ChartType.AREA,     "label": "Aire",                 "reason": "Accentuer le volume cumulatif"},
        {"type": ChartType.BAR,      "label": "Barres",               "reason": "Comparer valeur par valeur"},
        {"type": ChartType.SCATTER,  "label": "Nuage de points",      "reason": "Détecter les corrélations"},
    ],
    ChartType.AREA: [
        {"type": ChartType.LINE,     "label": "Courbe",               "reason": "Vue plus lisible sans remplissage"},
        {"type": ChartType.BAR,      "label": "Barres",               "reason": "Comparaison discrète par période"},
        {"type": ChartType.WATERFALL,"label": "Cascade",              "reason": "Montrer les variations cumulées"},
    ],
    ChartType.PIE: [
        {"type": ChartType.DOUGHNUT, "label": "Donut",                "reason": "Variante avec valeur centrale"},
        {"type": ChartType.BAR_H,    "label": "Barres horizontales",  "reason": "Comparaison précise des valeurs"},
        {"type": ChartType.TREEMAP,  "label": "Treemap",              "reason": "Hiérarchie des proportions"},
    ],
    ChartType.DOUGHNUT: [
        {"type": ChartType.PIE,      "label": "Camembert",            "reason": "Vue secteurs classique"},
        {"type": ChartType.BAR,      "label": "Barres",               "reason": "Comparaison précise des valeurs"},
        {"type": ChartType.TREEMAP,  "label": "Treemap",              "reason": "Proportions en surface"},
    ],
    ChartType.SCATTER: [
        {"type": ChartType.LINE,     "label": "Courbe",               "reason": "Si dimension temporelle détectée"},
        {"type": ChartType.BUBBLE,   "label": "Bulles",               "reason": "Ajouter une 3ème dimension"},
        {"type": ChartType.HEATMAP,  "label": "Heatmap",              "reason": "Voir la densité des points"},
    ],
    ChartType.HEATMAP: [
        {"type": ChartType.BAR,      "label": "Barres empilées",      "reason": "Comparaison par catégorie"},
        {"type": ChartType.SCATTER,  "label": "Nuage de points",      "reason": "Voir la dispersion"},
        {"type": ChartType.TABLE,    "label": "Tableau",              "reason": "Valeurs exactes lisibles"},
    ],
    ChartType.CHOROPLETH: [
        {"type": ChartType.BUBBLE_MAP,"label": "Carte bulles",        "reason": "Taille proportionnelle par lieu"},
        {"type": ChartType.BAR_H,    "label": "Classement",           "reason": "Ranking pays/régions"},
        {"type": ChartType.TABLE,    "label": "Tableau",              "reason": "Valeurs exactes par pays"},
    ],
    ChartType.BUBBLE_MAP: [
        {"type": ChartType.CHOROPLETH,"label": "Carte choroplèthe",   "reason": "Couleur par intensité géo"},
        {"type": ChartType.BAR_H,    "label": "Classement villes",    "reason": "Comparaison ordonnée"},
        {"type": ChartType.SCATTER,  "label": "Nuage de points",      "reason": "Relation entre deux métriques"},
    ],
    ChartType.TREEMAP: [
        {"type": ChartType.PIE,      "label": "Camembert",            "reason": "Répartition circulaire"},
        {"type": ChartType.BAR_H,    "label": "Barres horizontales",  "reason": "Classement linéaire"},
        {"type": ChartType.FUNNEL,   "label": "Entonnoir",            "reason": "Visualiser la décroissance"},
    ],
    ChartType.FUNNEL: [
        {"type": ChartType.BAR_H,    "label": "Barres horizontales",  "reason": "Comparaison étape par étape"},
        {"type": ChartType.LINE,     "label": "Courbe",               "reason": "Évolution du taux de conversion"},
        {"type": ChartType.TABLE,    "label": "Tableau",              "reason": "Chiffres de conversion exacts"},
    ],
    ChartType.WATERFALL: [
        {"type": ChartType.BAR,      "label": "Barres",               "reason": "Vue comparative simple"},
        {"type": ChartType.LINE,     "label": "Courbe",               "reason": "Tendance de variation"},
        {"type": ChartType.AREA,     "label": "Aire",                 "reason": "Volume cumulatif visible"},
    ],
    ChartType.TABLE: [
        {"type": ChartType.BAR,      "label": "Barres",               "reason": "Visualiser les valeurs clés"},
        {"type": ChartType.LINE,     "label": "Courbe",               "reason": "Si données temporelles présentes"},
        {"type": ChartType.PIE,      "label": "Camembert",            "reason": "Répartition des valeurs"},
    ],
    ChartType.SANKEY: [
        {"type": ChartType.BAR,      "label": "Barres empilées",      "reason": "Flux par source/destination"},
        {"type": ChartType.TABLE,    "label": "Tableau",              "reason": "Détail des flux exacts"},
    ],
    ChartType.KPI_CARD: [
        {"type": ChartType.GAUGE,    "label": "Jauge",                "reason": "Objectif vs réalisé visuel"},
        {"type": ChartType.SPARKLINE,"label": "Sparkline",            "reason": "Tendance compacte"},
    ],
}


def get_alt_viz_suggestions(
    chart_type: str,
    rows: List[Dict],
    columns: List[str],
    intent: "DashboardIntent",
    max_suggestions: int = 3,
) -> List[Dict]:
    """
    Retourne les suggestions de visualisations alternatives contextualisées.
    Filtre intelligemment selon les données disponibles.

    Returns:
        List de {type, label, reason, icon} — max max_suggestions items
    """
    base_alts = _ALT_VIZ_MAP.get(chart_type, [])
    if not base_alts:
        return []

    from decimal import Decimal as _Dec
    n_rows = len(rows)
    n_num = sum(1 for c in columns if rows and isinstance(rows[0].get(c), (int, float, _Dec)))
    n_cat = len(columns) - n_num
    has_time = intent.is_trend or bool(intent.time_field)
    has_geo  = intent.is_geo or bool(intent.geo_field)

    # Filtrage contextuel — certaines alts ne sont pertinentes que dans certains cas
    CONTEXT_FILTERS: Dict[str, callable] = {
        ChartType.LINE:      lambda: has_time or n_rows >= 5,
        ChartType.SCATTER:   lambda: n_num >= 2,
        ChartType.HEATMAP:   lambda: n_cat >= 2 and n_rows >= 10,
        ChartType.BUBBLE:    lambda: n_num >= 2,
        ChartType.CHOROPLETH:lambda: has_geo,
        ChartType.BUBBLE_MAP:lambda: has_geo,
        ChartType.SANKEY:    lambda: n_cat >= 2 and n_num >= 1,
        ChartType.FUNNEL:    lambda: n_rows <= 10,
        ChartType.WATERFALL: lambda: has_time or n_rows <= 12,
        ChartType.TREEMAP:   lambda: n_rows >= 3,
        ChartType.PIE:       lambda: 2 <= n_rows <= 10,
        ChartType.DOUGHNUT:  lambda: 2 <= n_rows <= 10,
        ChartType.SPARKLINE: lambda: n_rows >= 4,
        ChartType.GAUGE:     lambda: n_rows == 1 or n_num == 1,
    }

    # Icônes SVG mini par type (inline, stroke)
    _MINI_SVG: Dict[str, str] = {
        "bar":           '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="7" width="3" height="6"/><rect x="5.5" y="4" width="3" height="9"/><rect x="10" y="1" width="3" height="12"/></svg>',
        "bar_horizontal":'<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="1"  width="6" height="3"/><rect x="1" y="5.5" width="10" height="3"/><rect x="1" y="10" width="8" height="3"/></svg>',
        "line":          '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><polyline points="1,12 4,7 7,9 10,4 13,2"/></svg>',
        "area":          '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><polyline points="1,13 4,8 7,10 10,5 13,2"/><path d="M1,13 L1,13 4,8 7,10 10,5 13,2 13,13 Z" opacity=".25" fill="currentColor" stroke="none"/></svg>',
        "pie":           '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="7" r="6"/><line x1="7" y1="7" x2="7" y2="1"/><line x1="7" y1="7" x2="12.2" y2="9"/></svg>',
        "doughnut":      '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="7" r="6"/><circle cx="7" cy="7" r="3"/></svg>',
        "scatter":       '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="3" cy="10" r="1.2"/><circle cx="7" cy="5" r="1.2"/><circle cx="11" cy="8" r="1.2"/><circle cx="5" cy="8" r="1.2"/><circle cx="9" cy="3" r="1.2"/></svg>',
        "treemap":       '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="1" width="7" height="7"/><rect x="9" y="1" width="4" height="3"/><rect x="9" y="5" width="4" height="3"/><rect x="1" y="9" width="3" height="4"/><rect x="5" y="9" width="8" height="4"/></svg>',
        "heatmap":       '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="1" width="3" height="3"/><rect x="5" y="1" width="3" height="3" opacity=".5"/><rect x="9" y="1" width="4" height="3" opacity=".2"/><rect x="1" y="5" width="3" height="3" opacity=".7"/><rect x="5" y="5" width="3" height="3"/><rect x="9" y="5" width="4" height="3" opacity=".4"/><rect x="1" y="9" width="3" height="4" opacity=".3"/><rect x="5" y="9" width="3" height="4" opacity=".8"/><rect x="9" y="9" width="4" height="4" opacity=".6"/></svg>',
        "choropleth":    '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><ellipse cx="7" cy="7" rx="6" ry="6"/><path d="M2 7 Q5 5 7 7 Q9 9 12 7"/><path d="M4 4 Q6 5 7 4 Q8 3 10 4"/><path d="M4 10 Q6 9 7 10 Q8 11 10 10"/></svg>',
        "bubble_map":    '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><ellipse cx="7" cy="7" rx="6" ry="6"/><circle cx="4" cy="7" r="1.5"/><circle cx="9" cy="5" r="2.2"/><circle cx="7" cy="10" r="1"/></svg>',
        "funnel":        '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M1 2 H13 L9.5 7 L9.5 12 L4.5 12 L4.5 7 Z"/></svg>',
        "waterfall":     '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="6" width="2" height="5"/><rect x="4" y="4" width="2" height="3"/><rect x="7" y="7" width="2" height="2"/><rect x="10" y="5" width="2" height="4"/><polyline points="1,6 3,6 3,4 5,4 5,7 7,7 7,5 9,5 9,9 11,9" stroke-dasharray="1.5 1"/></svg>',
        "sankey":        '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M1 3 C5 3 5 6 9 6 C12 6 12 6 13 6" stroke-width="2.5" opacity=".5"/><path d="M1 8 C5 8 5 6 9 6 C12 6 12 6 13 7" stroke-width="1.5" opacity=".5"/><path d="M1 11 C5 11 5 8 9 8 C12 8 12 8 13 9" stroke-width="1" opacity=".5"/></svg>',
        "table":         '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="1" width="12" height="12" rx="1.5"/><line x1="1" y1="5" x2="13" y2="5"/><line x1="1" y1="9" x2="13" y2="9"/><line x1="5" y1="5" x2="5" y2="13"/></svg>',
        "gauge":         '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 11 A6 6 0 0 1 12 11"/><line x1="7" y1="11" x2="4" y2="6" stroke-width="1.8"/><circle cx="7" cy="11" r="1.2"/></svg>',
        "bubble":        '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="4" cy="10" r="2"/><circle cx="9" cy="7"  r="3"/><circle cx="7" cy="3"  r="1.5"/></svg>',
        "sparkline":     '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><polyline points="1,11 3,8 5,10 7,6 9,8 11,4 13,5"/></svg>',
        "kpi_card":      '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="1" y="3" width="12" height="8" rx="1.5"/><line x1="4" y1="7" x2="10" y2="7"/><line x1="4" y1="9" x2="7" y2="9"/></svg>',
    }

    result = []
    for alt in base_alts:
        t = alt["type"]
        # Vérifie si cette alternative est pertinente pour les données actuelles
        filter_fn = CONTEXT_FILTERS.get(t)
        if filter_fn and not filter_fn():
            continue
        result.append({
            "type":   t,
            "label":  alt["label"],
            "reason": alt["reason"],
            "icon":   _MINI_SVG.get(t, ""),
        })
        if len(result) >= max_suggestions:
            break

    return result


# ══════════════════════════════════════════════════════════════
# SPRINT 3 — HEATMAP GÉOGRAPHIQUE §2.4.3-B.4
# Type dédié : GEO_HEATMAP — densité par pays/région
# ══════════════════════════════════════════════════════════════

# Ajout de la constante GEO_HEATMAP dans la classe existante
ChartType.GEO_HEATMAP = "geo_heatmap"


def build_geo_heatmap_data(
    rows: List[Dict],
    columns: List[str],
) -> Optional[Dict]:
    """
    Construit les données pour une heatmap géographique (densité par pays/région).
    Retourne None si pas de colonnes géographiques détectées.

    Format retourné :
        {
            "type": "geo_heatmap",
            "data": [{"country": "France", "value": 45.0}, ...],
            "geo_col": "ShipCountry",
            "val_col": "nb_lignes",
            "colorscale": "YlOrRd",
            "title": "Densité par pays",
        }
    """
    GEO_HINTS    = ["country","pays","countryname","shipcountry","nation","cntr"]
    METRIC_HINTS = ["amount","total","revenue","montant","freight","price","qty","quantity","nb","count","value"]
    ID_PATTERNS  = ["_id","id_","rowid","productid","orderid","customerid"]

    from decimal import Decimal as _Dec

    cat_cols = [c for c in columns if not (rows and isinstance(rows[0].get(c), (int, float, _Dec)))]
    num_cols = [c for c in columns if rows and isinstance(rows[0].get(c), (int, float, _Dec))
                and not any(p in c.lower() for p in ID_PATTERNS)]

    geo_col = next((c for c in cat_cols if any(h in c.lower() for h in GEO_HINTS)), None)
    if not geo_col:
        return None

    val_col = next((c for c in num_cols if any(h in c.lower() for h in METRIC_HINTS)), None)
    if not val_col and num_cols:
        val_col = num_cols[0]

    # Agrégation par pays
    agg: Dict[str, float] = {}
    for r in rows:
        country = str(r.get(geo_col, "") or "").strip()
        if not country or country.lower() in ("none","null",""):
            continue
        val = float(r.get(val_col, 1) or 1) if val_col else 1.0
        agg[country] = agg.get(country, 0.0) + val

    if not agg:
        return None

    geo_data = [{"country": k, "value": round(v, 2)} for k, v in sorted(agg.items(), key=lambda x: -x[1])]

    return {
        "type":       "geo_heatmap",
        "data":       geo_data[:80],
        "geo_col":    geo_col,
        "val_col":    val_col or "count",
        "colorscale": "YlOrRd",   # Jaune → Orange → Rouge (chaleur)
        "title":      (val_col or "nb").replace("_", " ").title() + " par pays",
        "total":      sum(agg.values()),
    }


# ══════════════════════════════════════════════════════════════
# SPRINT 3 — RECOMMANDATIONS CONTEXTUELLES §2.4.3-G.1
# Suggestions intelligentes après génération du dashboard
# ══════════════════════════════════════════════════════════════

@dataclass
class DashboardRecommendation:
    """Une recommandation contextuelle pour enrichir le dashboard."""
    id:       str
    icon_tag: str          # tag SVG inline (ex: "[BOLT]")
    label:    str          # Texte court affiché
    question: str          # Question NL à envoyer si clic
    category: str          # "drill", "compare", "enrich", "export", "template"
    priority: int = 0      # Plus élevé = affiché en premier


class ContextualRecommender:
    """
    Génère des recommandations contextuelles basées sur :
    - Le type de dashboard généré
    - L'intent détecté (tendance, composition, géo…)
    - Les widgets présents
    - Les insights détectés
    """

    def recommend(
        self,
        spec: "DashboardSpec",
        intent: "DashboardIntent",
        max_recs: int = 4,
    ) -> List[DashboardRecommendation]:
        recs: List[DashboardRecommendation] = []
        q_lower = spec.question.lower()

        # ── 1. Drill-down temporel ────────────────────────────
        chart_types = {w.chart_type for w in spec.widgets}
        if ChartType.LINE in chart_types or ChartType.AREA in chart_types:
            recs.append(DashboardRecommendation(
                id="drill_temporal",
                icon_tag="[TREND_UP]",
                label="Drill-down par trimestre",
                question=f"évolution trimestrielle {spec.question}",
                category="drill",
                priority=90,
            ))

        # ── 2. Comparaison géographique ───────────────────────
        if ChartType.CHOROPLETH in chart_types or ChartType.BUBBLE_MAP in chart_types:
            recs.append(DashboardRecommendation(
                id="geo_heatmap",
                icon_tag="[BOLT]",
                label="Carte de chaleur (densité)",
                question=f"carte densité {spec.question}",
                category="enrich",
                priority=85,
            ))
            recs.append(DashboardRecommendation(
                id="geo_top",
                icon_tag="[UP]",
                label="Top 10 pays/régions",
                question=f"top 10 pays par {spec.question}",
                category="drill",
                priority=80,
            ))

        # ── 3. Comparaison N vs N-1 ───────────────────────────
        if intent.is_trend:
            recs.append(DashboardRecommendation(
                id="compare_yn1",
                icon_tag="[TREND_UP]",
                label="Comparer avec N-1",
                question=f"comparaison {spec.question} année précédente",
                category="compare",
                priority=88,
            ))

        # ── 4. Répartition complémentaire ─────────────────────
        if intent.is_top_n or ChartType.BAR_H in chart_types or ChartType.BAR in chart_types:
            recs.append(DashboardRecommendation(
                id="composition",
                icon_tag="[BOLT]",
                label="Répartition en camembert",
                question=f"répartition {spec.question}",
                category="enrich",
                priority=70,
            ))

        # ── 5. Dashboard complet métier ───────────────────────
        is_sales = any(k in q_lower for k in ["vente","ca","chiffre","commande","client","revenu"])
        is_finance = any(k in q_lower for k in ["tresorerie","trésorerie","solde","bancaire","financement","budget"])
        is_stock = any(k in q_lower for k in ["stock","produit","article","logistique","entrepot"])
        is_hr = any(k in q_lower for k in ["salarié","employé","effectif","rh","salaire","conge"])

        if is_sales:
            recs.append(DashboardRecommendation(
                id="tpl_sales",
                icon_tag="[UP]",
                label="Dashboard Ventes complet",
                question="dashboard ventes complet top clients évolution CA répartition produits",
                category="template",
                priority=75,
            ))
        if is_finance:
            recs.append(DashboardRecommendation(
                id="tpl_finance",
                icon_tag="[BOLT]",
                label="Dashboard Finance complet",
                question="dashboard finance trésorerie solde flux mouvements bancaires",
                category="template",
                priority=75,
            ))
        if is_stock:
            recs.append(DashboardRecommendation(
                id="tpl_stock",
                icon_tag="[UP]",
                label="Dashboard Stocks complet",
                question="dashboard stocks niveaux réapprovisionnement produits critiques",
                category="template",
                priority=75,
            ))

        # ── 6. Anomalies → demande de détails ─────────────────
        down_insights = [i for i in spec.insights if "[DOWN]" in i or "TREND_DOWN" in i]
        if down_insights:
            recs.append(DashboardRecommendation(
                id="anomaly_drill",
                icon_tag="[DOWN]",
                label="Analyser les baisses détectées",
                question=f"détail anomalies baisses {spec.question}",
                category="drill",
                priority=95,  # Priorité maximale si anomalie
            ))

        # ── 7. Export suggéré pour rapport ───────────────────
        if len(spec.widgets) >= 3:
            recs.append(DashboardRecommendation(
                id="export_pptx",
                icon_tag="[BOLT]",
                label="Exporter en PowerPoint",
                question="",  # action directe, pas une question NL
                category="export",
                priority=60,
            ))

        # ── 8. Suggestion de filtre temporel ─────────────────
        if not intent.is_trend and len(spec.widgets) >= 1:
            recs.append(DashboardRecommendation(
                id="add_time",
                icon_tag="[TREND_UP]",
                label="Ajouter dimension temporelle",
                question=f"évolution mensuelle {spec.question}",
                category="enrich",
                priority=65,
            ))

        # Dédoublonnage et tri
        seen = set()
        unique_recs = []
        for r in sorted(recs, key=lambda x: -x.priority):
            if r.id not in seen:
                seen.add(r.id)
                unique_recs.append(r)

        return unique_recs[:max_recs]

    def to_dict(self, rec: DashboardRecommendation) -> Dict:
        return {
            "id":       rec.id,
            "icon_tag": rec.icon_tag,
            "label":    rec.label,
            "question": rec.question,
            "category": rec.category,
            "priority": rec.priority,
        }


# ── Singleton recommender ──────────────────────────────────────
_contextual_recommender: Optional[ContextualRecommender] = None

def get_contextual_recommender() -> ContextualRecommender:
    global _contextual_recommender
    if _contextual_recommender is None:
        _contextual_recommender = ContextualRecommender()
    return _contextual_recommender


# ══════════════════════════════════════════════════════════════
# EXTENSION DashboardDesigner — intègre alt_viz + recommendations
# ══════════════════════════════════════════════════════════════

def enrich_spec_sprint3(
    spec: "DashboardSpec",
    intent: "DashboardIntent",
) -> "DashboardSpec":
    """
    Post-traitement Sprint 3 sur un DashboardSpec existant :
    1. Ajoute alt_viz à chaque widget
    2. Génère les recommandations contextuelles globales
    3. Construit la geo_heatmap si applicable

    Appelé depuis DashboardGenerator.generate() après design().
    """
    recommender = get_contextual_recommender()

    # 1. Alt viz par widget
    for w in spec.widgets:
        if w.chart_type == ChartType.KPI_CARD:
            # Les KPI cards n'ont pas d'alternatives de viz
            w.options["alt_viz"] = []
            continue
        rows = []
        cols = []
        # Reconstruire rows/cols depuis widget data si possible
        wd = w.data or {}
        if "rows" in wd and "headers" in wd:
            # TABLE widget
            rows = [{c: r[i] for i, c in enumerate(wd["headers"])} for r in wd.get("rows", [])]
            cols = wd.get("headers", [])
        elif "data" in wd and isinstance(wd["data"], list) and wd["data"] and isinstance(wd["data"][0], dict):
            # Choropleth / bubble_map
            rows = wd["data"]
            cols = list(rows[0].keys()) if rows else []
        elif "labels" in wd and "datasets" in wd:
            # Standard chart
            labels = wd.get("labels", [])
            datasets_data = wd.get("datasets", [])
            if datasets_data and labels:
                first_ds = datasets_data[0]
                col_name = first_ds.get("label", "value")
                rows = [{"label": l, col_name: d} for l, d in zip(labels, first_ds.get("data", []))]
                cols = ["label", col_name]

        alts = get_alt_viz_suggestions(w.chart_type, rows, cols, intent)
        w.options["alt_viz"] = alts

    # 2. Recommandations contextuelles globales
    recs = recommender.recommend(spec, intent)
    spec.options = getattr(spec, "options", {}) or {}
    spec.options["recommendations"] = [recommender.to_dict(r) for r in recs]

    return spec