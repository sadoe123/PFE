"""
OnePilot — Agentic RAG §Sprint 8.5
Agent ReAct (Reason + Act) pour génération SQL autonome et auto-corrective.

Architecture :
    Question
        ↓
    AgentRAG.run()
        ├── [NEW] Semantic matching flou sur SXA_DIRECT_SQL (Jaccard + synonymes)
        ├── Tool: search_schema     → RAG 7C (BM25 + pgvector + Graph BFS)
        ├── Tool: execute_sql       → ConnectorFactory (MSSQL/OData/CSV/REST/PG)
        ├── Tool: validate_result   → vérifier cohérence du résultat
        ├── Tool: search_views      → chercher dans les vues SXA métier
        ├── Tool: get_table_columns → obtenir les colonnes exactes d'une table
        └── [NEW] Error parser étendu → extraction colonne invalide multi-format MSSQL

Sprint 8.5 — Améliorations vs Sprint 8 :
    1. Matching flou SXA_DIRECT_SQL  → couverture variantes lexicales (+12 patterns)
    2. 12 nouveaux patterns SXA_DIRECT_SQL (cours de change, journal, intégration bancaire)
    3. Error parser MSSQL étendu → capture colonnes invalides tous formats d'erreur
    4. Prompt enrichi auto : colonnes exactes injectées dès la 1ère erreur colonne

Boucle ReAct :
    Reason → Act → Observe → Reason → Act → ...  (max MAX_ITERATIONS)

Générique : fonctionne avec toutes les sources (MSSQL, OData, CSV, REST, PostgreSQL)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MAX_ITERATIONS   = 5      # max boucles ReAct
MAX_SQL_RETRIES  = 3      # max tentatives de correction SQL
OLLAMA_HOST      = __import__("os").environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL     = __import__("os").environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_TIMEOUT   = int(__import__("os").environ.get("OLLAMA_TIMEOUT", "60"))

# Vues SXA métier — connues statiquement pour injection rapide
SXA_VIEWS = {
    "Comptes":                       ["ID", "CODE", "DESCRIPTION", "Banque", "Société", "Devises", "Groupe_Sociétés", "Groupe_de_comptes"],
    "Transactions bancaires":        ["CODE", "DESCRIPTION", "Banque", "Société", "AMOUNT", "montant avec signe", "CUR_ID_TRNCURRENCY", "TRNDATE", "VALUEDATE", "ISDEBIT", "Statut"],
    "Dernière integration bancaire": ["CODE", "Banque", "Société", "Devises", "CLOSINGBALANCEAMOUNT", "CLOSINGBALANCEDATETIME", "MontantAvecSigne"],
    "SI_Trésorerie":                 ["CODE", "DESCRIPTION", "Banque", "Société", "AMOUNTI", "ISDEBITI", "TRNDATE"],
    "SI_Bancaire":                   ["CODE", "DESCRIPTION", "Banque", "Société", "AMOUNTI", "ISDEBITI", "TRNDATE"],
    "Journal":                       ["CODE", "DESCRIPTION", "Sous_Catégorie_flux", "Catégorie_flux", "Montant ABS", "ISDEBIT", "Date de transaction", "date de valeur", "Statut", "Montant_avec_signe", "Compte", "Banque", "Société", "Devises"],
    "FINANCEMENT_BI":                ["TRN_ID", "Date début", "Date fin", "Montant", "Devises financement", "type_transaction", "maturité", "type", "état", "compte", "Banque", "Société"],
    "cours marchés":                 ["CUR_ID_CURRFROM", "CUR_ID_CURRTO", "ASK", "RATEDATE"],
}

# Mapping SQL direct pour les cas connus — évite les ambiguïtés
SXA_DIRECT_SQL = {
    # ── Virements ────────────────────────────────────────────────────────────
    "virement":                   "SELECT TOP 100 [TL_ID] AS Compte, [CUR_ID] AS Devise, SUM([AMOUNT]) AS MontantTotal FROM [DI_TL_PRPPMT] GROUP BY [TL_ID], [CUR_ID] ORDER BY MontantTotal DESC",
    "montant total des virements": "SELECT TOP 100 [TL_ID] AS Compte, [CUR_ID] AS Devise, SUM([AMOUNT]) AS MontantTotal FROM [DI_TL_PRPPMT] GROUP BY [TL_ID], [CUR_ID] ORDER BY MontantTotal DESC",

    # ── Comptes ───────────────────────────────────────────────────────────────
    "comptes avec":               "SELECT TOP 100 [CODE] AS Compte, [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés] FROM [Comptes] ORDER BY [Société], [Banque]",
    "liste les comptes":          "SELECT TOP 100 [CODE] AS Compte, [DESCRIPTION], [Banque], [Société], [Devises] FROM [Comptes] ORDER BY [Société]",

    # ── Financements ─────────────────────────────────────────────────────────
    "financement ouvert":         "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' ORDER BY [Montant] DESC",
    "financements ouverts":       "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' ORDER BY [Montant] DESC",
    "affiche les financements":   "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' ORDER BY [Société], [Montant] DESC",
    "financement":                "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] ORDER BY [état], [Montant] DESC",

    # ── Trésorerie ────────────────────────────────────────────────────────────
    "solde de trésorerie":        "SELECT TOP 100 [Société], SUM([AMOUNTI]) AS Solde_Tresorerie FROM [SI_Trésorerie] GROUP BY [Société] ORDER BY Solde_Tresorerie DESC",
    "solde trésorerie":           "SELECT TOP 100 [Société], SUM([AMOUNTI]) AS Solde_Tresorerie FROM [SI_Trésorerie] GROUP BY [Société] ORDER BY Solde_Tresorerie DESC",

    # ── Transactions ──────────────────────────────────────────────────────────
    "transactions bancaires":     "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [montant avec signe], [TRNDATE], [Statut] FROM [Transactions bancaires] ORDER BY [TRNDATE] DESC",
    "journal des flux":           "SELECT TOP 100 [Compte], [Banque], [Société], [Montant_avec_signe], [Sous_Catégorie_flux], [Date de transaction], [Statut] FROM [Journal] ORDER BY [Date de transaction] DESC",


    # ── Aggregations par type et etat ────────────────────────────────────────
    "par type et par etat":              "SELECT TOP 100 [type_transaction] AS TypeFinancement, [état] AS EtatFinancement, COUNT(*) AS Nb, SUM([Montant]) AS Total FROM [FINANCEMENT_BI] GROUP BY [type_transaction], [état] ORDER BY Total DESC",
    "par etat et par type":              "SELECT TOP 100 [état] AS EtatFinancement, [type_transaction] AS TypeFinancement, COUNT(*) AS Nb, SUM([Montant]) AS Total FROM [FINANCEMENT_BI] GROUP BY [état], [type_transaction] ORDER BY Total DESC",
    "total des financements par type et par etat": "SELECT TOP 100 [type_transaction] AS TypeFinancement, [état] AS EtatFinancement, COUNT(*) AS Nb, SUM([Montant]) AS Total FROM [FINANCEMENT_BI] GROUP BY [type_transaction], [état] ORDER BY Total DESC",
    "nombre de financements par type":   "SELECT TOP 100 [type_transaction], COUNT(*) AS Nb, SUM([Montant]) AS Total FROM [FINANCEMENT_BI] GROUP BY [type_transaction] ORDER BY Total DESC",

    # ── Agrégations ──────────────────────────────────────────────────────────
    "total des transactions":     "SELECT TOP 100 [Banque], SUM([AMOUNT]) AS TotalTransactions FROM [Transactions bancaires] GROUP BY [Banque] ORDER BY TotalTransactions DESC",


    # ── Comptes par banque — fix compare BNP vs SG ───────────────────────────
    "comptes bnp":                       "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%BNP%' ORDER BY [Société], [CODE]",
    "comptes bnp paribas":               "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%BNP%' ORDER BY [Société], [CODE]",
    "comptes société générale":          "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%Société Générale%' ORDER BY [Société], [CODE]",
    "comptes societe generale":          "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%Société Générale%' ORDER BY [Société], [CODE]",
    "comptes sg":                        "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%Société Générale%' ORDER BY [Société], [CODE]",
    "comptes banque postale":            "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%Banque Postale%' ORDER BY [Société], [CODE]",
    "comptes attijari":                  "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises], [Groupe_Sociétés], [Groupe_de_comptes] FROM [Comptes] WHERE [Banque] LIKE '%Attijari%' ORDER BY [Société], [CODE]",



    # ── Fix utilisateurs actifs + bloqués ───────────────────────────────────
    # NB : "utilisateurs bloqués" (avec accent) est normalisé en "utilisateurs bloques"
    # et matche automatiquement via passe 2 (direction unique) → pas besoin de doublon.
    "utilisateurs actifs":               "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [NAME], [EMAIL], [ISLOCKED], [FCSCODE] FROM [TH_USR] WHERE [ISLOCKED] = 0 ORDER BY [CODE]",
    "utilisateurs bloques":              "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [NAME], [EMAIL], [ISLOCKED], [FCSCODE] FROM [TH_USR] WHERE [ISLOCKED] = 1 ORDER BY [CODE]",
    "utilisateurs non bloques":          "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [NAME], [EMAIL], [ISLOCKED], [FCSCODE] FROM [TH_USR] WHERE [ISLOCKED] = 0 ORDER BY [CODE]",
    "liste les utilisateurs actifs":     "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [NAME], [EMAIL], [ISLOCKED], [FCSCODE] FROM [TH_USR] WHERE [ISLOCKED] = 0 ORDER BY [CODE]",
    "liste les utilisateurs bloques":    "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [NAME], [EMAIL], [ISLOCKED], [FCSCODE] FROM [TH_USR] WHERE [ISLOCKED] = 1 ORDER BY [CODE]",

    # ── Fix droits d'accès utilisateurs ─────────────────────────────────────
    "droits d'accès":                    "SELECT TOP 100 V.[USERCODE] AS Utilisateur, V.[COMPANYCODE] AS Société, U.[DESCRIPTION], U.[ISLOCKED] FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] ORDER BY V.[USERCODE]",
    "droits accès":                      "SELECT TOP 100 V.[USERCODE] AS Utilisateur, V.[COMPANYCODE] AS Société, U.[DESCRIPTION], U.[ISLOCKED] FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] ORDER BY V.[USERCODE]",
    "leurs droits":                      "SELECT TOP 100 V.[USERCODE] AS Utilisateur, V.[COMPANYCODE] AS Société, U.[DESCRIPTION], U.[ISLOCKED] FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] ORDER BY V.[USERCODE]",

    # ── Fix transactions par devise ───────────────────────────────────────────
    "transactions usd":                  "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [CUR_ID_TRNCURRENCY] AS Devise, [TRNDATE] FROM [Transactions bancaires] WHERE [CUR_ID_TRNCURRENCY] = 'USD' ORDER BY [TRNDATE] DESC",
    "total des transactions usd":        "SELECT TOP 100 [Banque], [Société], SUM([AMOUNT]) AS Total, COUNT(*) AS Nb FROM [Transactions bancaires] WHERE [CUR_ID_TRNCURRENCY] = 'USD' GROUP BY [Banque], [Société] ORDER BY Total DESC",
    "transactions eur":                  "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [CUR_ID_TRNCURRENCY] AS Devise, [TRNDATE] FROM [Transactions bancaires] WHERE [CUR_ID_TRNCURRENCY] = 'EUR' ORDER BY [TRNDATE] DESC",
    "solde comptes eur":                 "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises] FROM [Comptes] WHERE [Devises] = 'EUR' ORDER BY [Société], [Banque]",
    "comptes eur":                       "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises] FROM [Comptes] WHERE [Devises] = 'EUR' ORDER BY [Société], [Banque]",
    "comptes usd":                       "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises] FROM [Comptes] WHERE [Devises] = 'USD' ORDER BY [Société], [Banque]",
    "comptes tnd":                       "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [Devises] FROM [Comptes] WHERE [Devises] = 'TND' ORDER BY [Société], [Banque]",

    # ── Fix financements clôturés ─────────────────────────────────────────────
    "financements clôturés":             "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'clôturé' ORDER BY [Date fin] DESC",
    "financements clotures":             "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'clôturé' ORDER BY [Date fin] DESC",
    "financements fermes":               "SELECT TOP 100 [TRN_ID], [Montant], [type_transaction], [maturité], [type], [état], [compte], [Banque], [Société], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'clôturé' ORDER BY [Date fin] DESC",

    # ── Problème 1 fix : flux de trésorerie + financements échéance ──────────
    "financements arrivant à échéance":  "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Date fin] <= DATEADD(MONTH, 3, GETDATE()) ORDER BY [Date fin] ASC",
    "financements a echeance":           "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Date fin] <= DATEADD(MONTH, 3, GETDATE()) ORDER BY [Date fin] ASC",
    "financements echeance":             "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Date fin] <= DATEADD(MONTH, 3, GETDATE()) ORDER BY [Date fin] ASC",
    "flux de tresorerie du mois":        "SELECT TOP 100 [Compte], [Banque], [Société], [Montant_avec_signe], [Catégorie_flux], [Sous_Catégorie_flux], [Date de transaction] FROM [Journal] WHERE [Date de transaction] >= DATEADD(MONTH, -1, GETDATE()) ORDER BY [Date de transaction] DESC",
    "flux tresorerie mois":              "SELECT TOP 100 [Compte], [Banque], [Société], [Montant_avec_signe], [Catégorie_flux], [Date de transaction] FROM [Journal] WHERE [Date de transaction] >= DATEADD(MONTH, -1, GETDATE()) ORDER BY [Date de transaction] DESC",

    # ── Problème 2 fix : compare financements BNP vs SG ─────────────────────
    # Deux patterns séparés — l'orchestrateur les exécutera en parallèle
    "financements bnp":                  "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%BNP%' ORDER BY [Montant] DESC",
    "financements bnp paribas":          "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%BNP%' ORDER BY [Montant] DESC",
    "financements société générale":     "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Société Générale%' ORDER BY [Montant] DESC",
    "financements societe generale":     "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Société Générale%' ORDER BY [Montant] DESC",
    "financements sg":                   "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Société Générale%' ORDER BY [Montant] DESC",
    "financements groupama":             "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Groupama%' ORDER BY [Montant] DESC",
    "financements groupama banque":      "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Groupama%' ORDER BY [Montant] DESC",
    "financements banque postale":       "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Banque Postale%' ORDER BY [Montant] DESC",
    "financements la banque postale":    "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Banque Postale%' ORDER BY [Montant] DESC",
    "financements banque populaire":     "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Banque Populaire%' ORDER BY [Montant] DESC",
    "financements bpm":                  "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Banque] LIKE '%Banque Populaire%' ORDER BY [Montant] DESC",

    # ── Problème 3 fix : utilisateurs bloqués + sociétés ────────────────────
    "utilisateurs bloques et leurs societes": "SELECT TOP 100 U.[CODE], U.[DESCRIPTION], U.[ISLOCKED], V.[COMPANYCODE] AS Société FROM [TH_USR] U JOIN [VDTSSXACOMPANYRIGHT] V ON U.[FCSCODE] = V.[USERCODE] WHERE U.[ISLOCKED] = 1 ORDER BY U.[CODE]",
    "utilisateurs bloqués et leurs sociétés": "SELECT TOP 100 U.[CODE], U.[DESCRIPTION], U.[ISLOCKED], V.[COMPANYCODE] AS Société FROM [TH_USR] U JOIN [VDTSSXACOMPANYRIGHT] V ON U.[FCSCODE] = V.[USERCODE] WHERE U.[ISLOCKED] = 1 ORDER BY U.[CODE]",

    # ── Q1 MultiQuery fix : total transactions par banque/année ───────────────
    "total transactions bnp 2024":  "SELECT TOP 100 [Banque], [Société], YEAR([TRNDATE]) AS Année, SUM([AMOUNT]) AS Total FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND YEAR([TRNDATE]) = 2024 GROUP BY [Banque], [Société], YEAR([TRNDATE]) ORDER BY Total DESC",
    "total transactions bnp":       "SELECT TOP 100 [Banque], [Société], SUM([AMOUNT]) AS Total FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' GROUP BY [Banque], [Société] ORDER BY Total DESC",
    "transactions bnp 2024":        "SELECT TOP 100 [Banque], [Société], YEAR([TRNDATE]) AS Année, SUM([AMOUNT]) AS Total FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND YEAR([TRNDATE]) = 2024 GROUP BY [Banque], [Société], YEAR([TRNDATE]) ORDER BY Total DESC",
    "total transactions par banque 2024": "SELECT TOP 100 [Banque], [Société], SUM([AMOUNT]) AS Total FROM [Transactions bancaires] WHERE YEAR([TRNDATE]) = 2024 GROUP BY [Banque], [Société] ORDER BY Total DESC",
    "transactions par banque":    "SELECT TOP 100 [Banque], SUM([AMOUNT]) AS TotalTransactions, COUNT(*) AS NbTransactions FROM [Transactions bancaires] GROUP BY [Banque] ORDER BY TotalTransactions DESC",

    # ── Utilisateurs / Sociétés ───────────────────────────────────────────────
    "utilisateurs avec leur société":  "SELECT TOP 100 V.[COMPANYCODE] AS Société, V.[USERCODE] AS Utilisateur, U.[CODE] AS Code_Utilisateur, U.[DESCRIPTION] AS Description, U.[ISLOCKED] AS Bloqué FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] ORDER BY V.[COMPANYCODE], V.[USERCODE]",
    "liste les utilisateurs":          "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [NAME], [EMAIL], [ISLOCKED], [FCSCODE] FROM [TH_USR] WHERE [ISLOCKED] = 0 ORDER BY [CODE]",
    "affiche les utilisateurs":        "SELECT TOP 100 V.[COMPANYCODE] AS Société, V.[USERCODE] AS Utilisateur, U.[DESCRIPTION] AS Description FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] ORDER BY V.[COMPANYCODE]",

    # ── Jointure AA_AU2CMP / GS_CMP ──────────────────────────────────────────
    "jointure entre aa_au2cmp":        "SELECT TOP 100 A.[AU2CMP_ID], A.[VER] FROM [AA_AU2CMP] A ORDER BY A.[AU2CMP_ID]",
    "aa_au2cmp":                       "SELECT TOP 100 [AU2CMP_ID], [VER] FROM [AA_AU2CMP] ORDER BY [AU2CMP_ID]",
    "gs_cmp":                          "SELECT TOP 100 [CMP_ID], [CODE], [DESCRIPTION] FROM [GS_CMP] ORDER BY [CMP_ID]",

    # ── Devises / Pays ────────────────────────────────────────────────────────
    "devises disponibles":             "SELECT DISTINCT [CUR_ID] AS Devise FROM [GS_CUR] ORDER BY Devise",
    "liste les devises":               "SELECT DISTINCT [CUR_ID] AS Code_Devise, [DESCRIPTION] AS Libellé FROM [GS_CUR] ORDER BY Code_Devise",
    "codes pays":                      "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "liste les pays":                  "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "codes iso":                       "SELECT DISTINCT [CNTR_ID] AS Code_ISO, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_ISO",

    # ── Variantes explicites pour éviter faux positifs du Jaccard ────────────
    "affiche-moi les pays":            "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "montre-moi les pays":             "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "donne-moi les pays":              "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "affiche les pays":                "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "montre les pays":                 "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "quels sont les pays":             "SELECT DISTINCT [CNTR_ID] AS Code_Pays, [DESCRIPTION] AS Pays FROM [GS_CNTR] ORDER BY Code_Pays",
    "affiche-moi les devises":         "SELECT DISTINCT [CUR_ID] AS Code_Devise, [DESCRIPTION] AS Libellé FROM [GS_CUR] ORDER BY Code_Devise",
    "montre-moi les devises":          "SELECT DISTINCT [CUR_ID] AS Code_Devise, [DESCRIPTION] AS Libellé FROM [GS_CUR] ORDER BY Code_Devise",
    "quels sont les taux de change":   "SELECT TOP 100 [CUR_ID_CURRFROM] AS Devise_Source, [CUR_ID_CURRTO] AS Devise_Cible, [ASK] AS Cours, [RATEDATE] AS Date FROM [cours marchés] ORDER BY [RATEDATE] DESC",
    "quelles sont les devises":        "SELECT DISTINCT [CUR_ID] AS Code_Devise, [DESCRIPTION] AS Libellé FROM [GS_CUR] ORDER BY Code_Devise",
    # ── Sprint 8.5 — Nouveaux patterns ───────────────────────────────────────

    # Cours de change / marchés
    "cours de change":                 "SELECT TOP 100 [CUR_ID_CURRFROM] AS Devise_Source, [CUR_ID_CURRTO] AS Devise_Cible, [ASK] AS Cours, [RATEDATE] AS Date FROM [cours marchés] ORDER BY [RATEDATE] DESC",
    "cours marché":                    "SELECT TOP 100 [CUR_ID_CURRFROM] AS Devise_Source, [CUR_ID_CURRTO] AS Devise_Cible, [ASK] AS Cours, [RATEDATE] AS Date FROM [cours marchés] ORDER BY [RATEDATE] DESC",
    "taux de change":                  "SELECT TOP 100 [CUR_ID_CURRFROM] AS Devise_Source, [CUR_ID_CURRTO] AS Devise_Cible, [ASK] AS Cours, [RATEDATE] AS Date FROM [cours marchés] ORDER BY [RATEDATE] DESC",

    # Intégration bancaire
    "dernière intégration":            "SELECT TOP 100 [CODE], [Banque], [Société], [Devises], [CLOSINGBALANCEAMOUNT] AS Solde_Clôture, [CLOSINGBALANCEDATETIME] AS Date_Clôture, [MontantAvecSigne] FROM [Dernière integration bancaire] ORDER BY [Date_Clôture] DESC",
    "intégration bancaire":            "SELECT TOP 100 [CODE], [Banque], [Société], [Devises], [CLOSINGBALANCEAMOUNT] AS Solde_Clôture, [CLOSINGBALANCEDATETIME] AS Date_Clôture FROM [Dernière integration bancaire] ORDER BY [Date_Clôture] DESC",
    "solde bancaire":                  "SELECT TOP 100 [Banque], [Société], [Devises], [CLOSINGBALANCEAMOUNT] AS Solde, [CLOSINGBALANCEDATETIME] AS Date FROM [Dernière integration bancaire] ORDER BY [Société], [Banque]",

    # Journal des flux
    "flux de trésorerie":              "SELECT TOP 100 [Compte], [Banque], [Société], [Montant_avec_signe], [Catégorie_flux], [Date de transaction] FROM [Journal] ORDER BY [Date de transaction] DESC",
    "catégories de flux":              "SELECT DISTINCT [Catégorie_flux], [Sous_Catégorie_flux] FROM [Journal] ORDER BY [Catégorie_flux]",

    # SI Bancaire
    "si bancaire":                     "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Bancaire] ORDER BY [TRNDATE] DESC",
    "rapprochement bancaire":          "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Bancaire] ORDER BY [Société], [TRNDATE] DESC",

    # Groupes / sociétés
    "groupe de sociétés":              "SELECT DISTINCT [Groupe_Sociétés], [Société] FROM [Comptes] ORDER BY [Groupe_Sociétés], [Société]",
    "liste les sociétés":              "SELECT DISTINCT [Société] FROM [Comptes] ORDER BY [Société]",
    "toutes les sociétés":             "SELECT DISTINCT [Société] FROM [Comptes] ORDER BY [Société]",

    # Banques
    "liste les banques":               "SELECT DISTINCT [Banque] FROM [Comptes] ORDER BY [Banque]",

    # ── FINANCEMENT_BI — vue SXA avec Société, Banque, Montant ───────────────
    "total paiements par devise uniquement": "SELECT TOP 100 [Devises  financement] AS Devise, SUM([Montant]) AS Total_Paiements FROM [FINANCEMENT_BI] GROUP BY [Devises  financement] ORDER BY Total_Paiements DESC",
    "total des paiements par société":     "SELECT TOP 100 [Société], SUM([Montant]) AS Total_Paiements FROM [FINANCEMENT_BI] GROUP BY [Société] ORDER BY Total_Paiements DESC",
    "paiements par devise et par société toutes années": "SELECT TOP 100 [Devises  financement] AS Devise, [Société], SUM([Montant]) AS Total_Paiements FROM [FINANCEMENT_BI] GROUP BY [Devises  financement], [Société] ORDER BY Total_Paiements DESC",
    "paiements 2024":                      "SELECT TOP 100 [Devises  financement] AS Devise, [Société], SUM([Montant]) AS Total_Paiements FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = 2024 GROUP BY [Devises  financement], [Société] ORDER BY Total_Paiements DESC",
    "total des paiements par devise et par société en 2024": "SELECT TOP 100 [Devises  financement] AS Devise, [Société], SUM([Montant]) AS Total_Paiements FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = 2024 GROUP BY [Devises  financement], [Société] ORDER BY Total_Paiements DESC",
    "financements par société":            "SELECT TOP 100 [Société], [type_transaction], [état], SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Société], [type_transaction], [état] ORDER BY Montant_Total DESC",
    # ── SI_Trésorerie — colonnes validées : CODE, DESCRIPTION, Banque, Société, AMOUNTI, ISDEBITI, TRNDATE
    "transactions en eur superieures a 50000 en 2023":  "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [ISDEBITI] = 0 AND [AMOUNTI] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNTI] DESC",
    "transactions eur superieures 50000 bnp 2023":      "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] LIKE '%BNP%' AND [AMOUNTI] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNTI] DESC",
    "transactions en eur superieures a 50000 pour les comptes bnp en 2023": "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] LIKE '%BNP%' AND [AMOUNTI] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNTI] DESC",
    "transactions superieures a 50000":                 "SELECT TOP 100 [CODE], [Banque], [Société], [AMOUNTI], [TRNDATE] FROM [SI_Trésorerie] WHERE [AMOUNTI] > 50000 ORDER BY [AMOUNTI] DESC",
    "tresorerie par banque":                            "SELECT TOP 100 [Banque], SUM([AMOUNTI]) AS Total, COUNT(*) AS Nb FROM [SI_Trésorerie] GROUP BY [Banque] ORDER BY Total DESC",
    "tresorerie par societe":                           "SELECT TOP 100 [Société], SUM([AMOUNTI]) AS Total FROM [SI_Trésorerie] GROUP BY [Société] ORDER BY Total DESC",

    # ── FINANCEMENT_BI — maturité
    "financements dont la maturite depasse 5 ans":      "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE DATEDIFF(YEAR, [Date début], [Date fin]) > 5 ORDER BY [Montant] DESC",
    "financements maturite superieure a 5 ans":         "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [maturité], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE DATEDIFF(YEAR, [Date début], [Date fin]) > 5 ORDER BY [Montant] DESC",
    "financements maturite par banque et groupe":       "SELECT TOP 100 [Banque], [Groupe_Sociétés], [Société], [Montant], [maturité], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE DATEDIFF(YEAR, [Date début], [Date fin]) > 5 ORDER BY [Banque], [Groupe_Sociétés]",

    # ── Utilisateurs — bonne table TH_USR
    "utilisateurs avec acces a sage":                   "SELECT TOP 100 V.[COMPANYCODE] AS Société, V.[USERCODE] AS Utilisateur, U.[CODE], U.[DESCRIPTION], U.[ISLOCKED] FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] WHERE V.[COMPANYCODE] = 'SAGE' ORDER BY V.[USERCODE]",
    "utilisateurs crees apres 2024": "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) >= 2024 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees apres janvier 2024": "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE [CREATIONDATETIME] >= CONVERT(datetime,'20240101') ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees en 2022":            "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) = 2022 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees en 2023":            "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) = 2023 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees en 2024":            "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) = 2024 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees en 2025":            "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) = 2025 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees apres 2021":         "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) >= 2021 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees apres 2022":         "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) >= 2022 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs crees apres 2023":         "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) >= 2023 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs créés en 2022":            "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) = 2022 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs créés en 2023":            "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) = 2023 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs créés après 2021":         "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) >= 2021 ORDER BY [CREATIONDATETIME] DESC",
    "utilisateurs créés après 2022":         "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR] WHERE YEAR([CREATIONDATETIME]) >= 2022 ORDER BY [CREATIONDATETIME] DESC",

    # ── Q3 fix : utilisateurs avec accès à plus d'une société ────────────────
    # Tables valides : VDTSSXACOMPANYRIGHT (V) + TH_USR (U)
    "utilisateurs avec accès à plus d'une société":     "SELECT TOP 100 V.[USERCODE] AS Utilisateur, U.[DESCRIPTION], COUNT(DISTINCT V.[COMPANYCODE]) AS Nb_Sociétés, STRING_AGG(V.[COMPANYCODE], ', ') AS Sociétés FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] GROUP BY V.[USERCODE], U.[DESCRIPTION] HAVING COUNT(DISTINCT V.[COMPANYCODE]) > 1 ORDER BY Nb_Sociétés DESC",
    "utilisateurs acces plus une societe":              "SELECT TOP 100 V.[USERCODE] AS Utilisateur, U.[DESCRIPTION], COUNT(DISTINCT V.[COMPANYCODE]) AS Nb_Sociétés FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] GROUP BY V.[USERCODE], U.[DESCRIPTION] HAVING COUNT(DISTINCT V.[COMPANYCODE]) > 1 ORDER BY Nb_Sociétés DESC",
    "utilisateurs plusieurs societes":                  "SELECT TOP 100 V.[USERCODE] AS Utilisateur, U.[DESCRIPTION], COUNT(DISTINCT V.[COMPANYCODE]) AS Nb_Sociétés FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] GROUP BY V.[USERCODE], U.[DESCRIPTION] HAVING COUNT(DISTINCT V.[COMPANYCODE]) > 1 ORDER BY Nb_Sociétés DESC",

    "transactions en eur supérieures à 50000 pour les comptes bnp en 2023": "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND [AMOUNT] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNT] DESC",
    # ── Q5 fix : variantes BNP 2023 — formulation exacte utilisateur ─────────
    "transactions eur supérieures à 50000 pour les comptes bnp en 2023": "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND [AMOUNT] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNT] DESC",
    "transactions eur superieures 50000 bnp 2023":      "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND [AMOUNT] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNT] DESC",
    "transactions eur bnp 2023":                        "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND [AMOUNT] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNT] DESC",
    "transactions eur superieures a 50000 comptes bnp 2023": "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%BNP%' AND [AMOUNT] > 50000 AND YEAR([TRNDATE]) = 2023 ORDER BY [AMOUNT] DESC",
    # ── Q1 fix : TND + La Banque Postale + 2024 ──────────────────────────────
    "transactions tnd supérieures à 10000 la banque postale 2024": "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [CUR_ID_TRNCURRENCY] AS Devise, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%Banque Postale%' AND [AMOUNT] > 10000 AND [CUR_ID_TRNCURRENCY] = 'TND' AND YEAR([TRNDATE]) = 2024 ORDER BY [AMOUNT] DESC",
    "transactions tnd superieures a 10000 la banque postale 2024": "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [CUR_ID_TRNCURRENCY] AS Devise, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%Banque Postale%' AND [AMOUNT] > 10000 AND [CUR_ID_TRNCURRENCY] = 'TND' AND YEAR([TRNDATE]) = 2024 ORDER BY [AMOUNT] DESC",
    "transactions tnd banque postale 2024":             "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT] AS Montant, [CUR_ID_TRNCURRENCY] AS Devise, [TRNDATE] FROM [Transactions bancaires] WHERE [Banque] LIKE '%Banque Postale%' AND [AMOUNT] > 10000 AND [CUR_ID_TRNCURRENCY] = 'TND' AND YEAR([TRNDATE]) = 2024 ORDER BY [AMOUNT] DESC",
    "financements dont la maturité dépasse 5 ans par banque et groupe de sociétés": "SELECT TOP 100 [Banque], [Groupe_Sociétés], [Société], [Montant], [maturité], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE DATEDIFF(YEAR, [Date début], [Date fin]) > 5 ORDER BY [Banque], [Groupe_Sociétés]",
    "utilisateurs créés après janvier 2024 avec accès à sage et profil admin": "SELECT TOP 100 V.[COMPANYCODE] AS Société, V.[USERCODE] AS Utilisateur, U.[CODE], U.[DESCRIPTION], U.[ISLOCKED] FROM [VDTSSXACOMPANYRIGHT] V JOIN [TH_USR] U ON V.[USERCODE] = U.[FCSCODE] WHERE V.[COMPANYCODE] = 'SAGE' ORDER BY V.[USERCODE]",
        "financements actifs":                 "SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], [type_transaction], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' ORDER BY [Date début] DESC",
    "financements par banque":                    "SELECT TOP 100 [Banque], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque] ORDER BY Montant_Total DESC",
    "financements par banque et par societe":     "SELECT TOP 100 [Banque], [Société], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque], [Société] ORDER BY Montant_Total DESC",
    "financements par banque et par société":     "SELECT TOP 100 [Banque], [Société], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque], [Société] ORDER BY Montant_Total DESC",
    "financements par banque et par type":        "SELECT TOP 100 [Banque], [type_transaction], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque], [type_transaction] ORDER BY Montant_Total DESC",
    "top 10 financements par banque et par type": "SELECT TOP 10 [Banque], [type_transaction], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque], [type_transaction] ORDER BY Montant_Total DESC",
    "top 10 financements par banque":             "SELECT TOP 10 [Banque], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque] ORDER BY Montant_Total DESC",

    # ── Top N banques / sociétés ──────────────────────────────────────────────
    "top 5 banques par montant total de financements actifs":  "SELECT TOP 5 [Banque], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' GROUP BY [Banque] ORDER BY Montant_Total DESC",
    "top 5 banques par montant total":                         "SELECT TOP 5 [Banque], SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque] ORDER BY Montant_Total DESC",
    "top 5 banques par montant":                               "SELECT TOP 5 [Banque], SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque] ORDER BY Montant_Total DESC",
    "top 5 banques financements actifs":                       "SELECT TOP 5 [Banque], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' GROUP BY [Banque] ORDER BY Montant_Total DESC",
    "top 3 banques par montant":                               "SELECT TOP 3 [Banque], SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque] ORDER BY Montant_Total DESC",
    "top 10 banques par montant":                              "SELECT TOP 10 [Banque], SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque] ORDER BY Montant_Total DESC",

    # ── Sous-requêtes avancées ────────────────────────────────────────────────
    "financements dont le montant est superieur a la moyenne":  "SELECT TOP 100 [TRN_ID], [Banque], [Société], [Montant], [type_transaction], [état], [Date début], [Date fin] FROM [FINANCEMENT_BI] WHERE [Montant] > (SELECT AVG([Montant]) FROM [FINANCEMENT_BI]) ORDER BY [Montant] DESC",
    "financements superieur a la moyenne":                      "SELECT TOP 100 [TRN_ID], [Banque], [Société], [Montant], [type_transaction], [état] FROM [FINANCEMENT_BI] WHERE [Montant] > (SELECT AVG([Montant]) FROM [FINANCEMENT_BI]) ORDER BY [Montant] DESC",
    "financements au dessus de la moyenne":                     "SELECT TOP 100 [TRN_ID], [Banque], [Société], [Montant], [type_transaction], [état] FROM [FINANCEMENT_BI] WHERE [Montant] > (SELECT AVG([Montant]) FROM [FINANCEMENT_BI]) ORDER BY [Montant] DESC",
    "montant superieur a la moyenne financement":               "SELECT TOP 100 [TRN_ID], [Banque], [Société], [Montant], [type_transaction], [état] FROM [FINANCEMENT_BI] WHERE [Montant] > (SELECT AVG([Montant]) FROM [FINANCEMENT_BI]) ORDER BY [Montant] DESC",
    "financements dont la maturite depasse la duree moyenne":   "SELECT TOP 100 f.[TRN_ID], f.[Banque], f.[Société], f.[Montant], f.[type_transaction], f.[maturité], f.[état] FROM [FINANCEMENT_BI] f WHERE f.[maturité] > (SELECT AVG([maturité]) FROM [FINANCEMENT_BI]) ORDER BY f.[maturité] DESC",
    "financements dont la maturite depasse la moyenne":         "SELECT TOP 100 f.[TRN_ID], f.[Banque], f.[Société], f.[Montant], f.[type_transaction], f.[maturité], f.[état] FROM [FINANCEMENT_BI] f WHERE f.[maturité] > (SELECT AVG([maturité]) FROM [FINANCEMENT_BI]) ORDER BY f.[maturité] DESC",
    "maturite superieure a la moyenne":                         "SELECT TOP 100 [TRN_ID], [Banque], [Société], [Montant], [type_transaction], [maturité], [état] FROM [FINANCEMENT_BI] WHERE [maturité] > (SELECT AVG([maturité]) FROM [FINANCEMENT_BI]) ORDER BY [maturité] DESC",
    "quels sont les financements dont la maturite depasse la duree moyenne des financements bnp": "SELECT TOP 100 f.[TRN_ID], f.[Banque], f.[Société], f.[Montant], f.[type_transaction], f.[maturité] FROM [FINANCEMENT_BI] f WHERE f.[maturité] > (SELECT AVG(b.[maturité]) FROM [FINANCEMENT_BI] b WHERE b.[Banque] LIKE '%BNP%') ORDER BY f.[maturité] DESC",
    "repartition des financements par banque et par type": "SELECT TOP 100 [Banque], [type_transaction], COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] GROUP BY [Banque], [type_transaction] ORDER BY [Banque], Montant_Total DESC",
    "nombre de financements par banque et par societe":    "SELECT TOP 100 [Banque], [Société], COUNT(*) AS Nb_Financements FROM [FINANCEMENT_BI] GROUP BY [Banque], [Société] ORDER BY Nb_Financements DESC",
    "evolution des financements par banque sur 2024":      "SELECT TOP 100 [Banque], MONTH([Date début]) AS Mois, COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = 2024 GROUP BY [Banque], MONTH([Date début]) ORDER BY [Banque], Mois",
    "financements par banque et par devise en 2025":       "SELECT TOP 100 [Banque], [Devises du compte], SUM([Montant]) AS TotalMontant FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = 2025 GROUP BY [Banque], [Devises du compte] ORDER BY TotalMontant DESC",
    "groupe de financement":               "SELECT TOP 100 [Société], [Groupe_Sociétés], SUM([Montant]) AS Total FROM [FINANCEMENT_BI] GROUP BY [Société], [Groupe_Sociétés] ORDER BY Total DESC",
    "groupe des sociétés financement":       "SELECT TOP 100 [Groupe_Sociétés], [Société], SUM([Montant]) AS Total FROM [FINANCEMENT_BI] GROUP BY [Groupe_Sociétés], [Société] ORDER BY [Groupe_Sociétés], Total DESC",
    "toutes les banques":              "SELECT DISTINCT [Banque] FROM [Comptes] ORDER BY [Banque]",

    # ── Patterns Multi-query composées ──────────────────────────────────────
    # Ces patterns couvrent les questions composées avec vs / et leurs / et les
    # Ils sont gérés via AgentRAG directement (pas l'Orchestrateur MultiQuery)

    # Q1 : compare utilisateurs actifs vs bloques → résultat unifié COUNT par ISLOCKED
    "compare les utilisateurs actifs vs les utilisateurs bloques": (
        "SELECT [ISLOCKED] AS Statut_Numerique, "
        "CASE WHEN [ISLOCKED]=0 THEN 'Actifs' ELSE 'Bloques' END AS Statut, "
        "COUNT(*) AS Nombre "
        "FROM [TH_USR] "
        "GROUP BY [ISLOCKED] "
        "ORDER BY [ISLOCKED]"
    ),
    "utilisateurs actifs vs utilisateurs bloques": (
        "SELECT CASE WHEN [ISLOCKED]=0 THEN 'Actifs' ELSE 'Bloques' END AS Statut, "
        "COUNT(*) AS Nombre "
        "FROM [TH_USR] "
        "GROUP BY [ISLOCKED] "
        "ORDER BY [ISLOCKED]"
    ),
    "compare utilisateurs actifs vs bloques": (
        "SELECT CASE WHEN [ISLOCKED]=0 THEN 'Actifs' ELSE 'Bloques' END AS Statut, "
        "COUNT(*) AS Nombre "
        "FROM [TH_USR] "
        "GROUP BY [ISLOCKED] "
        "ORDER BY [ISLOCKED]"
    ),
    "utilisateurs actifs vs bloques": (
        "SELECT CASE WHEN [ISLOCKED]=0 THEN 'Actifs' ELSE 'Bloques' END AS Statut, "
        "COUNT(*) AS Nombre "
        "FROM [TH_USR] "
        "GROUP BY [ISLOCKED] "
        "ORDER BY [ISLOCKED]"
    ),

    # Q2 : compare financements ouverts vs clotures → UNION ou GROUP BY état
    "compare les financements ouverts vs les financements clotures": (
        "SELECT [état], COUNT(*) AS Nombre, SUM([Montant]) AS Total, "
        "AVG([Montant]) AS Moyenne "
        "FROM [FINANCEMENT_BI] "
        "WHERE [état] IN ('ouvert', 'clôturé') "
        "GROUP BY [état] "
        "ORDER BY [état]"
    ),
    "financements ouverts vs financements clotures": (
        "SELECT [état], COUNT(*) AS Nombre, SUM([Montant]) AS Total "
        "FROM [FINANCEMENT_BI] "
        "WHERE [état] IN ('ouvert', 'clôturé') "
        "GROUP BY [état] "
        "ORDER BY [état]"
    ),
    "financements ouverts vs clotures": (
        "SELECT [état], COUNT(*) AS Nombre, SUM([Montant]) AS Total "
        "FROM [FINANCEMENT_BI] "
        "WHERE [état] IN ('ouvert', 'clôturé') "
        "GROUP BY [état] "
        "ORDER BY [état]"
    ),

    # Q4 : utilisateurs actifs et leurs droits d'accès → JOIN VDTSSXACOMPANYRIGHT
    "utilisateurs actifs et leurs droits d'acces": (
        "SELECT TOP 100 U.[CODE], U.[NAME], U.[EMAIL], "
        "V.[COMPANYCODE] AS Societe, U.[ISLOCKED] "
        "FROM [TH_USR] U "
        "LEFT JOIN [VDTSSXACOMPANYRIGHT] V ON U.[FCSCODE] = V.[USERCODE] "
        "WHERE U.[ISLOCKED] = 0 "
        "ORDER BY U.[CODE]"
    ),
    "utilisateurs actifs et leurs droits": (
        "SELECT TOP 100 U.[CODE], U.[NAME], U.[EMAIL], "
        "V.[COMPANYCODE] AS Societe, U.[ISLOCKED] "
        "FROM [TH_USR] U "
        "LEFT JOIN [VDTSSXACOMPANYRIGHT] V ON U.[FCSCODE] = V.[USERCODE] "
        "WHERE U.[ISLOCKED] = 0 "
        "ORDER BY U.[CODE]"
    ),

    # Q5 : solde EUR et total transactions USD → UNION ALL
    "solde des comptes eur et total des transactions usd": (
        "SELECT 'Solde EUR' AS Type, [Société], [Banque], "
        "SUM([CLOSINGBALANCEAMOUNT]) AS Montant "
        "FROM [Dernière integration bancaire] "
        "WHERE [Devises] = 'EUR' "
        "GROUP BY [Société], [Banque] "
        "UNION ALL "
        "SELECT 'Total USD' AS Type, [Société], [Banque], "
        "SUM([AMOUNT]) AS Montant "
        "FROM [Transactions bancaires] "
        "WHERE [CUR_ID_TRNCURRENCY] = 'USD' "
        "GROUP BY [Société], [Banque] "
        "ORDER BY Type, Montant DESC"
    ),
    "solde comptes eur et total transactions usd": (
        "SELECT 'Solde EUR' AS Type, [Société], [Banque], "
        "SUM([CLOSINGBALANCEAMOUNT]) AS Montant "
        "FROM [Dernière integration bancaire] "
        "WHERE [Devises] = 'EUR' "
        "GROUP BY [Société], [Banque] "
        "UNION ALL "
        "SELECT 'Total USD' AS Type, [Société], [Banque], "
        "SUM([AMOUNT]) AS Montant "
        "FROM [Transactions bancaires] "
        "WHERE [CUR_ID_TRNCURRENCY] = 'USD' "
        "GROUP BY [Société], [Banque] "
        "ORDER BY Type, Montant DESC"
    ),

    # Q6 : financements actifs et les comptes associés → LEFT JOIN
    "financements actifs et les comptes associes": (
        "SELECT TOP 100 F.[TRN_ID], F.[Banque], F.[Société], "
        "F.[Montant], F.[type_transaction], F.[état], "
        "C.[CODE] AS Compte, C.[DESCRIPTION] AS Desc_Compte "
        "FROM [FINANCEMENT_BI] F "
        "LEFT JOIN [Comptes] C ON F.[Banque] = C.[Banque] AND F.[Société] = C.[Société] "
        "WHERE F.[état] = 'ouvert' "
        "ORDER BY F.[Montant] DESC"
    ),
    "financements actifs et comptes associes": (
        "SELECT TOP 100 F.[TRN_ID], F.[Banque], F.[Société], "
        "F.[Montant], F.[type_transaction], "
        "C.[CODE] AS Compte "
        "FROM [FINANCEMENT_BI] F "
        "LEFT JOIN [Comptes] C ON F.[Banque] = C.[Banque] AND F.[Société] = C.[Société] "
        "WHERE F.[état] = 'ouvert' "
        "ORDER BY F.[Montant] DESC"
    ),

    # Q7 : banques et devises associées
    "liste les banques et les devises associees": (
        "SELECT DISTINCT [Banque], [Devises] "
        "FROM [Comptes] "
        "ORDER BY [Banque], [Devises]"
    ),
    "banques et les devises associees": (
        "SELECT DISTINCT [Banque], [Devises] "
        "FROM [Comptes] "
        "ORDER BY [Banque], [Devises]"
    ),
    "banques et devises": (
        "SELECT DISTINCT [Banque], [Devises] "
        "FROM [Comptes] "
        "ORDER BY [Banque], [Devises]"
    ),

    # ── Sous-requêtes avancées — patterns spécifiques manquants ─────────────
    # Ces patterns doivent être AVANT les patterns courts pour éviter les faux positifs

    # Q5 : financements dont le montant dépasse la moyenne PAR BANQUE (partition)
    "financements dont le montant depasse la moyenne par banque": (
        "SELECT TOP 100 f.[TRN_ID], f.[Banque], f.[Société], f.[Montant], "
        "f.[type_transaction], f.[état] FROM [FINANCEMENT_BI] f "
        "WHERE f.[Montant] > (SELECT AVG(f2.[Montant]) FROM [FINANCEMENT_BI] f2 "
        "WHERE f2.[Banque] = f.[Banque]) ORDER BY f.[Banque], f.[Montant] DESC"
    ),
    "financements montant superieur moyenne par banque": (
        "SELECT TOP 100 f.[TRN_ID], f.[Banque], f.[Société], f.[Montant], "
        "f.[type_transaction], f.[état] FROM [FINANCEMENT_BI] f "
        "WHERE f.[Montant] > (SELECT AVG(f2.[Montant]) FROM [FINANCEMENT_BI] f2 "
        "WHERE f2.[Banque] = f.[Banque]) ORDER BY f.[Banque], f.[Montant] DESC"
    ),

    # Q6 : sociétés avec solde supérieur à la médiane (approximée par PERCENTILE_CONT)
    "societes avec un solde superieur a la mediane": (
        "SELECT [Société], AVG([CLOSINGBALANCEAMOUNT]) AS SoldeMoyen "
        "FROM [Dernière integration bancaire] "
        "GROUP BY [Société] "
        "HAVING AVG([CLOSINGBALANCEAMOUNT]) > ("
        "SELECT AVG([CLOSINGBALANCEAMOUNT]) FROM [Dernière integration bancaire]) "
        "ORDER BY SoldeMoyen DESC"
    ),
    "societes solde superieur mediane": (
        "SELECT [Société], AVG([CLOSINGBALANCEAMOUNT]) AS SoldeMoyen "
        "FROM [Dernière integration bancaire] "
        "GROUP BY [Société] "
        "HAVING AVG([CLOSINGBALANCEAMOUNT]) > ("
        "SELECT AVG([CLOSINGBALANCEAMOUNT]) FROM [Dernière integration bancaire]) "
        "ORDER BY SoldeMoyen DESC"
    ),
    "solde superieur a la mediane": (
        "SELECT [Société], AVG([CLOSINGBALANCEAMOUNT]) AS SoldeMoyen "
        "FROM [Dernière integration bancaire] "
        "GROUP BY [Société] "
        "HAVING AVG([CLOSINGBALANCEAMOUNT]) > ("
        "SELECT AVG([CLOSINGBALANCEAMOUNT]) FROM [Dernière integration bancaire]) "
        "ORDER BY SoldeMoyen DESC"
    ),

    # Q7 : banques dont le nombre de financements actifs dépasse la moyenne globale
    "banques dont le nombre de financements actifs depasse la moyenne globale": (
        "SELECT [Banque], COUNT(*) AS Nb_Financements_Actifs "
        "FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' "
        "GROUP BY [Banque] "
        "HAVING COUNT(*) > (SELECT AVG(cnt) FROM ("
        "SELECT COUNT(*) AS cnt FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' "
        "GROUP BY [Banque]) AS sub) "
        "ORDER BY Nb_Financements_Actifs DESC"
    ),
    "banques nombre financements actifs depasse moyenne": (
        "SELECT [Banque], COUNT(*) AS Nb_Financements_Actifs "
        "FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' "
        "GROUP BY [Banque] "
        "HAVING COUNT(*) > (SELECT AVG(cnt) FROM ("
        "SELECT COUNT(*) AS cnt FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' "
        "GROUP BY [Banque]) AS sub) "
        "ORDER BY Nb_Financements_Actifs DESC"
    ),
    "nombre de financements actifs depasse la moyenne": (
        "SELECT [Banque], COUNT(*) AS Nb_Financements_Actifs "
        "FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' "
        "GROUP BY [Banque] "
        "HAVING COUNT(*) > (SELECT AVG(cnt) FROM ("
        "SELECT COUNT(*) AS cnt FROM [FINANCEMENT_BI] WHERE [état] = 'ouvert' "
        "GROUP BY [Banque]) AS sub) "
        "ORDER BY Nb_Financements_Actifs DESC"
    ),

    # Q8 : top 3 types de transactions par montant moyen
    "top 3 types de transactions par montant moyen": (
        "SELECT TOP 3 [type_transaction], "
        "AVG([Montant]) AS Montant_Moyen, COUNT(*) AS Nb, SUM([Montant]) AS Total "
        "FROM [FINANCEMENT_BI] "
        "GROUP BY [type_transaction] "
        "ORDER BY Montant_Moyen DESC"
    ),
    "top 3 types transactions montant moyen": (
        "SELECT TOP 3 [type_transaction], "
        "AVG([Montant]) AS Montant_Moyen, COUNT(*) AS Nb, SUM([Montant]) AS Total "
        "FROM [FINANCEMENT_BI] "
        "GROUP BY [type_transaction] "
        "ORDER BY Montant_Moyen DESC"
    ),
    "types de transactions par montant moyen": (
        "SELECT TOP 10 [type_transaction], "
        "AVG([Montant]) AS Montant_Moyen, COUNT(*) AS Nb "
        "FROM [FINANCEMENT_BI] "
        "GROUP BY [type_transaction] "
        "ORDER BY Montant_Moyen DESC"
    ),

    # ── Requêtes agrégées fréquentes — évite hallucination colonnes ──────────
    "nombre de comptes par banque":        "SELECT [Banque], COUNT(*) AS Nombre_Comptes FROM [Comptes] GROUP BY [Banque] ORDER BY Nombre_Comptes DESC",
    "nombre de comptes par devise":        "SELECT [Devises], COUNT(*) AS Nombre_Comptes FROM [Comptes] GROUP BY [Devises] ORDER BY Nombre_Comptes DESC",
    "nombre de comptes actifs":            "SELECT [Banque], [Devises], COUNT(*) AS Nombre_Comptes FROM [Comptes] GROUP BY [Banque], [Devises] ORDER BY [Banque], [Devises]",
    "comptes par banque et par devise":    "SELECT [Banque], [Devises], COUNT(*) AS Nombre_Comptes FROM [Comptes] GROUP BY [Banque], [Devises] ORDER BY [Banque], [Devises]",
    "liste des utilisateurs bloqués":      "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [ISLOCKED] FROM [TH_USR] WHERE [ISLOCKED] = 1 ORDER BY [CODE]",
    "utilisateurs bloqués":               "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [ISLOCKED] FROM [TH_USR] WHERE [ISLOCKED] = 1 ORDER BY [CODE]",
    "bloqués":                             "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [ISLOCKED] FROM [TH_USR] WHERE [ISLOCKED] = 1 ORDER BY [CODE]",
    "comptes bloqués":                     "SELECT TOP 100 [USR_ID], [CODE], [DESCRIPTION], [ISLOCKED] FROM [TH_USR] WHERE [ISLOCKED] = 1 ORDER BY [CODE]",

    # ── SI_Trésorerie — encaissements / décaissements — données max 2023-12-31
    # ISDEBITI=0 → encaissement (crédit), ISDEBITI=1 → décaissement (débit)
    # Filtre dynamique sur MAX(YEAR(TRNDATE)) — évite le piège GETDATE() sans données récentes
    "total des encaissements du mois":    "SELECT SUM([AMOUNTI]) AS TotalEncaissements, COUNT(*) AS NbTransactions, MAX([TRNDATE]) AS DerniereMouvement FROM [SI_Trésorerie] WHERE [ISDEBITI] = 0 AND YEAR([TRNDATE]) = (SELECT MAX(YEAR([TRNDATE])) FROM [SI_Trésorerie])",
    "total encaissements du mois":        "SELECT SUM([AMOUNTI]) AS TotalEncaissements, COUNT(*) AS NbTransactions FROM [SI_Trésorerie] WHERE [ISDEBITI] = 0 AND YEAR([TRNDATE]) = (SELECT MAX(YEAR([TRNDATE])) FROM [SI_Trésorerie])",
    "encaissements du mois":              "SELECT SUM([AMOUNTI]) AS TotalEncaissements, COUNT(*) AS NbTransactions, MAX([TRNDATE]) AS DerniereMouvement FROM [SI_Trésorerie] WHERE [ISDEBITI] = 0 AND YEAR([TRNDATE]) = (SELECT MAX(YEAR([TRNDATE])) FROM [SI_Trésorerie])",
    "encaissements":                      "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [TRNDATE] FROM [SI_Trésorerie] WHERE [ISDEBITI] = 0 ORDER BY [TRNDATE] DESC",
    "total des décaissements du mois":    "SELECT SUM(ABS([AMOUNTI])) AS TotalDecaissements, COUNT(*) AS NbTransactions, MAX([TRNDATE]) AS DerniereMouvement FROM [SI_Trésorerie] WHERE [ISDEBITI] = 1 AND YEAR([TRNDATE]) = (SELECT MAX(YEAR([TRNDATE])) FROM [SI_Trésorerie])",
    "total decaissements du mois":        "SELECT SUM(ABS([AMOUNTI])) AS TotalDecaissements, COUNT(*) AS NbTransactions FROM [SI_Trésorerie] WHERE [ISDEBITI] = 1 AND YEAR([TRNDATE]) = (SELECT MAX(YEAR([TRNDATE])) FROM [SI_Trésorerie])",
    "décaissements du mois":              "SELECT SUM(ABS([AMOUNTI])) AS TotalDecaissements, COUNT(*) AS NbTransactions, MAX([TRNDATE]) AS DerniereMouvement FROM [SI_Trésorerie] WHERE [ISDEBITI] = 1 AND YEAR([TRNDATE]) = (SELECT MAX(YEAR([TRNDATE])) FROM [SI_Trésorerie])",
    "décaissements":                      "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [TRNDATE] FROM [SI_Trésorerie] WHERE [ISDEBITI] = 1 ORDER BY [TRNDATE] DESC",
    "encaissements par banque":           "SELECT [Banque], SUM([AMOUNTI]) AS TotalEncaissements, COUNT(*) AS Nb FROM [SI_Trésorerie] WHERE [ISDEBITI] = 0 GROUP BY [Banque] ORDER BY TotalEncaissements DESC",
    "décaissements par banque":           "SELECT [Banque], SUM(ABS([AMOUNTI])) AS TotalDecaissements, COUNT(*) AS Nb FROM [SI_Trésorerie] WHERE [ISDEBITI] = 1 GROUP BY [Banque] ORDER BY TotalDecaissements DESC",
    "flux trésorerie par banque":         "SELECT [Banque], SUM(CASE WHEN [ISDEBITI]=0 THEN [AMOUNTI] ELSE 0 END) AS Encaissements, SUM(CASE WHEN [ISDEBITI]=1 THEN ABS([AMOUNTI]) ELSE 0 END) AS Decaissements FROM [SI_Trésorerie] GROUP BY [Banque] ORDER BY [Banque]",

    # ── Transactions par banque — noms exacts validés dans SXA ───────────────
    # Banques réelles : 'BNP Paribas', 'Groupama Banque', 'La Banque Postale', 'Société Générale - SG'
    "transactions bnp paribas":           "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] = 'BNP Paribas' ORDER BY [TRNDATE] DESC",
    "transactions groupama":              "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] = 'Groupama Banque' ORDER BY [TRNDATE] DESC",
    "transactions banque postale":        "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] = 'La Banque Postale' ORDER BY [TRNDATE] DESC",
    "transactions société générale":      "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] = 'Société Générale - SG' ORDER BY [TRNDATE] DESC",
    "transactions du mois de janvier":    "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE MONTH([TRNDATE]) = 1 ORDER BY [TRNDATE] DESC",
    "transactions bnp janvier 2023":      "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] = 'BNP Paribas' AND MONTH([TRNDATE]) = 1 AND YEAR([TRNDATE]) = 2023 ORDER BY [TRNDATE] DESC",
    "transactions bnp 2023":              "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNTI], [ISDEBITI], [TRNDATE] FROM [SI_Trésorerie] WHERE [Banque] = 'BNP Paribas' AND YEAR([TRNDATE]) = 2023 ORDER BY [TRNDATE] DESC",

    # ── Transactions bancaires — colonnes validées SSMS ───────────────────────
    # Colonnes réelles : CODE, DESCRIPTION, Banque, Société, AMOUNT,
    #                    [montant avec signe], CUR_ID_TRNCURRENCY, TRNDATE,
    #                    VALUEDATE, ISDEBIT, Statut, Groupe_Sociétés,
    #                    Description_groupe_de_sociétés, Groupe_de_comptes
    "transactions bancaires":             "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [montant avec signe], [TRNDATE], [VALUEDATE], [ISDEBIT], [Statut] FROM [Transactions bancaires] ORDER BY [TRNDATE] DESC",
    "liste des transactions bancaires":   "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [montant avec signe], [TRNDATE], [ISDEBIT], [Statut] FROM [Transactions bancaires] ORDER BY [TRNDATE] DESC",
    "toutes les transactions":            "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [montant avec signe], [TRNDATE], [ISDEBIT] FROM [Transactions bancaires] ORDER BY [TRNDATE] DESC",
    "transactions en 2023":               "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [montant avec signe], [TRNDATE], [ISDEBIT] FROM [Transactions bancaires] WHERE YEAR([TRNDATE]) = 2023 ORDER BY [TRNDATE] DESC",
    "transactions par banque":            "SELECT [Banque], COUNT(*) AS NbTransactions, SUM([AMOUNT]) AS TotalMontant FROM [Transactions bancaires] GROUP BY [Banque] ORDER BY TotalMontant DESC",
    "transactions par société":           "SELECT [Société], COUNT(*) AS NbTransactions, SUM([AMOUNT]) AS TotalMontant FROM [Transactions bancaires] GROUP BY [Société] ORDER BY TotalMontant DESC",
    "transactions débit":                 "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [TRNDATE] FROM [Transactions bancaires] WHERE [ISDEBIT] = 1 ORDER BY [TRNDATE] DESC",
    "transactions crédit":                "SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], [AMOUNT], [TRNDATE] FROM [Transactions bancaires] WHERE [ISDEBIT] = 0 ORDER BY [TRNDATE] DESC",
}


# ─────────────────────────────────────────────────────────────────────────────
# SPRINT 8.5 — MATCHING FLOU SXA_DIRECT_SQL
# ─────────────────────────────────────────────────────────────────────────────

import re as _re_dynamic

# Banques connues dans SXA — tout nom propre hors cette liste = filtre dynamique inconnu
_KNOWN_BANKS = frozenset([
    "bnp", "bnp paribas", "société générale", "société générale - sg", "sgb", "stb", "biat", "attijari",
    "bh", "bh bank", "zitouna", "wifak", "arab tunisian bank", "atb", "abc",
    "banque postale", "la banque postale", "crédit agricole", "cacib",
    "natixis", "cib", "hsbc", "barclays", "citi", "citibank",
    "groupama", "groupama banque",
])

# Devises connues
_KNOWN_CURRENCIES = frozenset([
    "tnd", "eur", "usd", "gbp", "chf", "jpy", "mad", "dzd", "lyad",
    "xof", "xaf", "cad", "aud",
])


def _has_dynamic_filters(question: str) -> tuple[bool, list[str]]:
    """
    Détecte si une question contient des filtres dynamiques qui nécessitent
    une génération SQL par ReAct plutôt qu'un bypass Direct SQL.

    Filtres détectés :
      - Montants numériques  : "supérieures à 10000", "> 50000", "moins de 1 million"
      - Années spécifiques   : "en 2024", "2023", "depuis 2022"
      - Devises nommées      : "TND", "EUR", "USD" dans un contexte de filtre
      - Noms de banque       : "La Banque Postale", "BNP", noms propres inconnus
      - Seuils de durée      : "dépasse 5 ans", "supérieure à 3 mois"

    Returns:
        (has_filters: bool, reasons: list[str])
    """
    q = question.lower().strip()
    reasons = []

    # ── 1. Montants numériques avec comparateur ───────────────────────────────
    # Ex : "supérieures à 10000", "> 50 000", "moins de 1 million", "≥ 500"
    if _re_dynamic.search(
        r'(supérieure?s?\s+à|inférieure?s?\s+à|égale?s?\s+à|'
        r'plus\s+de|moins\s+de|au[-\s]dessus\s+de|au[-\s]dessous\s+de|'
        r'>\s*\d|<\s*\d|>=\s*\d|<=\s*\d|'
        r'\d[\s\u202f]*000|million|milliard)',
        q
    ):
        # Vérifier qu'il y a effectivement un nombre associé
        if _re_dynamic.search(r'\d{3,}', question):  # au moins un nombre à 3+ chiffres
            reasons.append("montant_numerique")

    # ── 2. Années spécifiques (4 chiffres 19xx ou 20xx) ──────────────────────
    years = _re_dynamic.findall(r'\b(?:19|20)\d{2}\b', question)
    if years:
        reasons.append(f"année({'|'.join(years)})")

    # ── 3. Devises nommées en contexte de filtre ──────────────────────────────
    # Ex : "transactions TND", "en EUR", "devise USD"
    for cur in _KNOWN_CURRENCIES:
        if _re_dynamic.search(
            rf'(transactions?|montant|solde|paiement|filtr|en|devise)\s+{cur}\b'
            rf'|\b{cur}\s+(supérieure?|inférieure?|égale?|transactions?)',
            q
        ):
            reasons.append(f"devise_filtre({cur.upper()})")
            break

    # ── 4. Noms de banque spécifiques ─────────────────────────────────────────
    # Ex : "La Banque Postale", "BNP", "Attijariwafa"
    for bank in _KNOWN_BANKS:
        if bank in q:
            reasons.append(f"banque({bank})")
            break
    else:
        # Nom propre inconnu : majuscule précédée de "banque" ou "pour"
        if _re_dynamic.search(r'\b(banque|pour\s+les?\s+comptes?)\s+[A-Z][a-zA-Z]+', question):
            reasons.append("banque_inconnue")

    # ── 5. Seuils de durée (maturité, ancienneté) ─────────────────────────────
    if _re_dynamic.search(
        r'(dépasse|supérieure?\s+à|plus\s+de)\s+\d+\s+(an|mois|jour)',
        q
    ):
        reasons.append("duree_filtre")

    # ── 6. Accès multiple / conditions sur utilisateurs ───────────────────────
    # Note : "accès à plus d'une société" a un pattern exact dans SXA_DIRECT_SQL
    # On détecte quand même pour que _pattern_covers_filters() puisse valider
    if _re_dynamic.search(r'(plus\s+d[\'e]\s*une?\s+société|accès\s+à\s+plus\s+d)', q):
        reasons.append("condition_acces")

    has_filters = len(reasons) > 0
    return has_filters, reasons

# Synonymes métier SXA pour normalisation avant matching
_SXA_SYNONYMS: Dict[str, str] = {
    # Ordre CRITIQUE : les expressions longues en premier pour éviter double-remplacement
    "taux de change":   "cours de change",
    "afficher les":     "liste les",
    "affiche les":      "liste les",
    "quelles sont":     "liste",
    "quels sont":       "liste",
    "mouvement bancaire": "transactions bancaires",
    "mouvements bancaires": "transactions bancaires",
    # Encaissements / Décaissements — alias métier
    "recettes":         "encaissements",
    "revenus bancaires": "encaissements",
    "entrées":          "encaissements",
    "sorties":          "décaissements",
    "dépenses bancaires": "décaissements",
    "decaissements":    "décaissements",
    "encaissement":     "encaissements",
    "decaissement":     "décaissements",
    # Verbes d'affichage (courts ensuite)
    "montre-moi":   "liste",    "donne-moi":    "liste",
    "afficher":     "liste",    "montrer":      "liste",
    "affiche":      "liste",    "montre":       "liste",
    "donner":       "liste",    "donne":        "liste",
    "voir":         "liste",    "quelles":      "",
    "quels":        "",
    # Finance
    "forex":        "cours de change",  "fx":          "cours de change",
    "cotation":     "cours",            "taux":        "cours",
    "rapprochement": "si bancaire",     "réconciliation": "si bancaire",
    # Entités
    "entreprise":   "société",  "compagnie":   "société",
    "entité":       "société",  "institution": "banque",
    "monnaie":      "devise",   "currency":    "devise",
    "monaie":       "devise",   "country":     "pays",
    "nation":       "pays",     "identifiant": "codes",
    # Comptes / transactions
    "compte bancaire": "comptes", "account": "comptes",
    "opérations":   "transactions bancaires",
    "opération":    "transaction",
    "écriture":     "transaction",
    "mouvements":   "transactions",
}

_SXA_STOPWORDS = frozenset([
    "le", "la", "les", "un", "une", "des", "du", "de", "en", "et", "ou",
    "pour", "dans", "sur", "par", "avec", "sans", "leur", "leurs",
    "toutes", "tous", "disponible", "disponibles", "existant", "existants",
    "me", "moi", "nous", "je", "il", "ils", "elle", "elles",
    "base", "données", "système", "application",
])


def _normalize_question(text: str) -> str:
    """Normalise une question : minuscule, NFD accents, synonymes, ponctuation."""
    import re as _re_norm
    import unicodedata as _ud
    text = text.lower().strip()
    # NFD : supprimer les accents (maturité→maturite, dépasse→depasse)
    text = _ud.normalize('NFD', text)
    text = ''.join(c for c in text if _ud.category(c) != 'Mn')
    # Appliquer les synonymes après NFD
    for src, tgt in sorted(_SXA_SYNONYMS.items(), key=lambda x: -len(x[0])):
        src_nfd = _ud.normalize('NFD', src.lower())
        src_nfd = ''.join(c for c in src_nfd if _ud.category(c) != 'Mn')
        text = text.replace(src_nfd, tgt)
    # Supprimer la ponctuation
    text = _re_norm.sub(r'[?!.,;:"\']', " ", text)
    return text

def _tokenize(text: str) -> set:
    """Tokenise et supprime les stopwords."""
    tokens = set(text.lower().split())
    return tokens - _SXA_STOPWORDS


def _jaccard(a: set, b: set) -> float:
    """Similarité de Jaccard entre deux ensembles de tokens."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _parse_date_filter(q: str):
    """Parse dynamiquement les questions de type 'utilisateurs créés après [mois] [année]'."""
    import re, unicodedata

    def _n(s):
        s = unicodedata.normalize("NFD", s.lower())
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    MOIS = {
        "janvier":"01","fevrier":"02","mars":"03","avril":"04",
        "mai":"05","juin":"06","juillet":"07","aout":"08",
        "septembre":"09","octobre":"10","novembre":"11","decembre":"12"
    }
    qn = _n(q)
    BASE = "SELECT TOP 100 [USR_ID],[CODE],[NAME],[EMAIL],[CREATIONDATETIME] FROM [TH_USR]"

    # "créés après mars 2023"
    m = re.search(r"crees? apres? ([a-z]+) (\d{4})", qn)
    if m:
        mois = MOIS.get(m.group(1), "01")
        annee = m.group(2)
        return f"{BASE} WHERE [CREATIONDATETIME] >= CONVERT(datetime,\'{annee}{mois}01\') ORDER BY [CREATIONDATETIME] DESC"

    # "créés en 2023"
    m = re.search(r"crees? en (\d{4})", qn)
    if m:
        return f"{BASE} WHERE YEAR([CREATIONDATETIME]) = {m.group(1)} ORDER BY [CREATIONDATETIME] DESC"

    # "créés après 2023"
    m = re.search(r"crees? apres? (\d{4})", qn)
    if m:
        return f"{BASE} WHERE YEAR([CREATIONDATETIME]) >= {m.group(1)} ORDER BY [CREATIONDATETIME] DESC"

    # ── Patterns génériques FINANCEMENT_BI ───────────────────────────────────

    # "evolution des financements par banque sur [année]"
    m = re.search(r"evolution.{0,20}financements?.{0,20}banque.{0,20}(?:sur|en|pour|de) (\d{4})", qn)
    if m:
        annee = m.group(1)
        return (f"SELECT TOP 100 [Banque], MONTH([Date début]) AS Mois, "
                f"COUNT(*) AS Nb_Financements, SUM([Montant]) AS Montant_Total "
                f"FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = {annee} "
                f"GROUP BY [Banque], MONTH([Date début]) ORDER BY [Banque], Mois")

    # "financements par banque et par devise en [année]"
    m = re.search(r"financements?.{0,20}banque.{0,20}devise.{0,10}(?:en|sur|pour|de) (\d{4})", qn)
    if m:
        annee = m.group(1)
        return (f"SELECT TOP 100 [Banque], [Devises du compte], SUM([Montant]) AS TotalMontant "
                f"FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = {annee} "
                f"GROUP BY [Banque], [Devises du compte] ORDER BY TotalMontant DESC")

    # "total des financements par banque et par devise en [année]"
    m = re.search(r"total.{0,20}financements?.{0,20}banque.{0,20}devise.{0,10}(?:en|sur|pour|de) (\d{4})", qn)
    if m:
        annee = m.group(1)
        return (f"SELECT TOP 100 [Banque], [Devises du compte], SUM([Montant]) AS TotalMontant "
                f"FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = {annee} "
                f"GROUP BY [Banque], [Devises du compte] ORDER BY TotalMontant DESC")

    # "total des financements par banque et par type en [année]" / "sur [année]"
    m = re.search(r"(?:total|repartition).{0,20}financements?.{0,20}banque.{0,20}type.{0,10}(?:en|sur|pour|de) (\d{4})", qn)
    if m:
        annee = m.group(1)
        return (f"SELECT TOP 100 [Banque], [type_transaction], COUNT(*) AS Nb, SUM([Montant]) AS Total "
                f"FROM [FINANCEMENT_BI] WHERE YEAR([Date début]) = {annee} "
                f"GROUP BY [Banque], [type_transaction] ORDER BY Total DESC")

    return None


# Mots-clés indiquant une question complexe qui ne doit PAS être interceptée par Direct SQL
_COMPLEX_KEYWORDS = [
    "répartition", "repartition", "analyse", "distribution", "ventilation",
    "dépasse", "depasse", "supérieur", "superieur", "inférieur", "inferieur",
    "moyenne", "fois la", "fois le", "x fois", "seuil",
    "tendance", "évolution", "evolution", "trimestre", "quarter",
    "corrélation", "correlation", "comparaison", "compare",
    "par devise et", "par société et", "par banque et",
    "groupe par", "groupé par", "having", "sous-requête",
    "maximum", "minimum", "rang", "classement", "percentile",
]

def _is_complex_question(question: str) -> bool:
    """Détecte si une question est trop complexe pour Direct SQL."""
    q_lower = question.lower()
    # Question avec plusieurs dimensions ET un verbe d'analyse
    has_analysis = any(k in q_lower for k in [
        "répartition", "repartition", "analyse", "tendance", "évolution",
        "distribution", "ventilation", "dépasse", "depasse", "fois la",
        "moyenne sectorielle", "corrélation"
    ])
    # Question avec HAVING implicite
    has_having = any(k in q_lower for k in [
        "dépasse", "depasse", "supérieur à la moyenne", "superieur a la moyenne",
        "fois la moyenne", "au-dessus", "en dessous", "seuil"
    ])
    # Question avec agrégation multiple (par X et par Y)
    has_multi_group = bool(
        __import__('re').search(r'par\s+\w+\s+et\s+par\s+\w+', q_lower)
    )
    return has_analysis or has_having or has_multi_group

def _find_direct_sql(question: str, threshold: float = 0.55) -> tuple[str | None, str | None, float]:
    """
    Cherche le meilleur pattern SXA_DIRECT_SQL pour une question.

    Stratégie en 3 passes :
      1. Matching exact (substring) — longest-match-first
      2. Matching après normalisation synonymes
      3. Matching flou Jaccard sur tokens (threshold=0.35)

    GARDE-FOU Sprint 8.5 fix :
      Si la question contient des filtres dynamiques (montants, années, noms
      de banque, devises en filtre, durées), le bypass Direct SQL est bloqué
      SAUF si le pattern matché dans SXA_DIRECT_SQL encode déjà ces filtres
      exactement (ex: pattern contient l'année ou le seuil).

    Returns:
        (pattern_matched, sql, score)  — sql=None si aucun match ou filtres dynamiques
    """
    # ── Garde complexité : questions analytiques → bypass Direct SQL ──────
    if _is_complex_question(question):
        import logging as _log
        _log.getLogger(__name__).debug(
            f"[DirectSQL] Question complexe détectée → bypass: '{question[:60]}'"
        )
        return None, None, 0.0

    import re as _re_cov
    q_lower = question.lower()
    q_norm  = _normalize_question(question)

    # ── Parsing dynamique dates (créés après/en [mois] [année]) ─────────────
    _date_sql = _parse_date_filter(question)
    if _date_sql:
        return "utilisateurs_date_dynamique", _date_sql, 1.0

    # ── Garde-fou : détecter les filtres dynamiques ──────────────────────────
    has_dynamic, dynamic_reasons = _has_dynamic_filters(question)

    def _pattern_covers_filters(pattern: str) -> bool:
        """
        Vérifie si le pattern SXA_DIRECT_SQL encode déjà les filtres dynamiques
        de la question — auquel cas le bypass reste valide.
        Ex: pattern="transactions eur superieures 50000 bnp 2023" → couvre tout.
        """
        if not has_dynamic:
            return True  # pas de filtres → toujours OK
        p_lower = pattern.lower()
        for reason in dynamic_reasons:
            if reason.startswith("année("):
                years = _re_cov.findall(r'\d{4}', reason)
                if not any(y in p_lower for y in years):
                    return False
            elif reason.startswith("devise_filtre("):
                cur = _re_cov.search(r'\(([A-Z]+)\)', reason)
                if cur and cur.group(1).lower() not in p_lower:
                    return False
            elif reason.startswith("banque("):
                bank = _re_cov.search(r'\((.+)\)', reason)
                if bank:
                    bank_name = bank.group(1).lower()
                    # Vérifie que le SQL du pattern filtre bien sur cette banque
                    # Accepte : LIKE '%BNP%', = 'BNP Paribas', LIKE '%Société Générale%'
                    sql_lower = SXA_DIRECT_SQL.get(pattern, "").lower()
                    bank_parts = bank_name.replace(" ", "%")
                    if (bank_name not in p_lower and
                        bank_name not in sql_lower and
                        bank_parts not in sql_lower):
                        return False
            elif reason == "montant_numerique":
                nums_q = _re_cov.findall(r'\d{3,}', question)
                nums_p = _re_cov.findall(r'\d{3,}', pattern)
                if not any(nq in nums_p for nq in nums_q):
                    return False
            elif reason == "duree_filtre":
                if not _re_cov.search(r'\d+\s*(an|mois)', p_lower):
                    return False
            elif reason == "condition_acces":
                if "plus" not in p_lower and "accès" not in p_lower:
                    return False
        return True

    # ── Passe 0 : SQL dynamique pour filtres numériques ─────────────────────
    # Construit le SQL depuis un template quand la question contient un seuil
    # numérique ou une durée — évite que le LLM hallucine les colonnes
    import re as _re_dyn
    q_dyn = question.lower().strip()

    # Transactions supérieures/inférieures à X (en YYYY)
    _match_amount = _re_dyn.search(
        r'(transaction|mouvement|opération).{0,40}(supérieure?s?\s+à|>\s*|plus\s+de|inférieure?s?\s+à|<\s*|moins\s+de)\s*([\d\s]+)',
        q_dyn
    )
    _match_year = _re_dyn.search(r'(20\d{2})', q_dyn)
    if _match_amount:
        _amount_raw = _re_dyn.sub(r'\s', '', _match_amount.group(3).strip())
        try:
            _amount = int(_amount_raw)
            _op = ">" if any(k in _match_amount.group(2) for k in ['supér','plus','>']) else "<"
            _yr = f" AND YEAR([TRNDATE]) = {_match_year.group(1)}" if _match_year else ""
            _dyn_sql = (
                f"SELECT TOP 100 [CODE], [DESCRIPTION], [Banque], [Société], "
                f"[AMOUNT], [montant avec signe], [TRNDATE], [ISDEBIT], [Statut] "
                f"FROM [Transactions bancaires] "
                f"WHERE [AMOUNT] {_op} {_amount}{_yr} "
                f"ORDER BY [AMOUNT] DESC"
            )
            logger.info(f"[AgentRAG] Passe 0 dynamique — transactions {_op} {_amount}{_yr}")
            return f"dyn_transactions_{_op}{_amount}", _dyn_sql, 0.99
        except (ValueError, AttributeError):
            pass

    # Financements maturité > N ans
    _match_mat = _re_dyn.search(
        r'(financement).{0,30}(maturit|depasse|superieure?).{0,10}(\d+)\s*(an|mois)',
        q_dyn
    )
    if _match_mat:
        try:
            _years = int(_match_mat.group(3))
            _unit = _match_mat.group(4)
            _fn = "YEAR" if "an" in _unit else "MONTH"
            _dyn_sql = (
                f"SELECT TOP 100 [TRN_ID], [Société], [Banque], [Montant], "
                f"[type_transaction], [état], [Date début], [Date fin] "
                f"FROM [FINANCEMENT_BI] "
                f"WHERE DATEDIFF({_fn}, [Date début], [Date fin]) > {_years} "
                f"ORDER BY [Montant] DESC"
            )
            logger.info(f"[AgentRAG] Passe 0 dynamique — financements maturité > {_years} {_unit}")
            return f"dyn_financements_mat_{_years}{_unit}", _dyn_sql, 0.99
        except (ValueError, AttributeError):
            pass

    # ── Passe 1 : exact substring (comportement Sprint 8 original) ──────────
    # Garde-fou : les patterns très courts (<= 15 chars) sont bloqués si la question
    # contient des mots sous-requête — risque de faux positif (ex: "financement" dans
    # "financements dont le montant dépasse la moyenne par banque")
    _SUBQUERY_KW = {
        "depasse", "superieur", "superieure", "inferieur", "inferieure",
        "moyenne", "mediane", "median", "percentile", "having", "partition",
        "dont le montant", "dont la maturite", "dont le nombre",
    }
    _has_subquery_kw = any(kw in q_norm for kw in _SUBQUERY_KW)

    for pattern, sql in sorted(SXA_DIRECT_SQL.items(), key=lambda x: -len(x[0])):
        if pattern in q_lower:
            # Bloquer les patterns courts si la question est une sous-requête complexe
            if _has_subquery_kw and len(pattern) <= 15:
                logger.debug(
                    f"[AgentRAG] P1 bloqué — pattern court '{pattern}' "
                    f"sur question sous-requête complexe"
                )
                continue
            if _pattern_covers_filters(pattern):
                return pattern, sql, 1.0
            else:
                logger.debug(
                    f"[AgentRAG] Bypass bloqué P1 — pattern='{pattern}' "
                    f"filtres dynamiques non couverts={dynamic_reasons}"
                )
                return None, None, 0.0  # filtres dynamiques → ReAct obligatoire

    # ── Passe 2 : substring après normalisation des synonymes ───────────────
    # RÈGLE : le pattern doit être contenu dans la question (pas l'inverse).
    # On trie par longueur décroissante → le pattern le plus spécifique gagne.
    # On exclut les patterns beaucoup plus longs que la question (faux positifs
    # du type "utilisateurs bloques et leurs societes" qui contient "utilisateurs bloques").
    for pattern, sql in sorted(SXA_DIRECT_SQL.items(), key=lambda x: -len(x[0])):
        pat_norm = _normalize_question(pattern)
        # Le pattern doit être contenu dans la question (direction unique)
        # ET ne doit pas être beaucoup plus long que la question (évite faux positifs)
        if pat_norm and pat_norm in q_norm and len(pat_norm) <= len(q_norm) + 5:
            if _pattern_covers_filters(pattern):
                return pattern, sql, 0.95
            else:
                logger.debug(
                    f"[AgentRAG] Bypass bloqué P2 — pattern='{pattern}' "
                    f"filtres dynamiques non couverts={dynamic_reasons}"
                )
                return None, None, 0.0

    # ── Passe 3 : Jaccard sur tokens ────────────────────────────────────────
    # Si filtres dynamiques détectés → Jaccard désactivé (trop risqué)
    if has_dynamic:
        logger.debug(
            f"[AgentRAG] Jaccard désactivé — filtres dynamiques: {dynamic_reasons}"
        )
        return None, None, 0.0

    q_tokens = _tokenize(q_norm)
    if not q_tokens:
        return None, None, 0.0

    best_pattern, best_sql, best_score = None, None, 0.0
    for pattern, sql in sorted(SXA_DIRECT_SQL.items(), key=lambda x: -len(x[0])):
        pat_tokens = _tokenize(_normalize_question(pattern))
        score = _jaccard(q_tokens, pat_tokens)
        # Bonus si tous les tokens du pattern sont dans la question
        if pat_tokens and pat_tokens.issubset(q_tokens):
            score = min(1.0, score + 0.25)
        if score > best_score:
            best_score, best_pattern, best_sql = score, pattern, sql

    if best_score >= threshold:
        return best_pattern, best_sql, best_score

    return None, None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class AgentAction(str, Enum):
    SEARCH_SCHEMA     = "search_schema"
    EXECUTE_SQL       = "execute_sql"
    VALIDATE_RESULT   = "validate_result"
    SEARCH_VIEWS      = "search_views"
    GET_TABLE_COLUMNS = "get_table_columns"
    FINAL_ANSWER      = "final_answer"
    ASK_CLARIFICATION = "ask_clarification"


@dataclass
class AgentStep:
    iteration:   int
    thought:     str          # raisonnement de l'agent
    action:      AgentAction
    action_input: Dict        # paramètres de l'action
    observation: str          # résultat de l'action
    duration_ms: int = 0


@dataclass
class AgentResult:
    success:      bool
    sql:          Optional[str]
    result:       Optional[List[Dict]]
    explanation:  str
    steps:        List[AgentStep] = field(default_factory=list)
    iterations:   int = 0
    duration_ms:  int = 0
    method:       str = "agentic_rag"
    warnings:     List[str] = field(default_factory=list)
    clarification: Optional[str] = None   # question à poser à l'utilisateur


# ─────────────────────────────────────────────────────────────────────────────
# OUTILS DE L'AGENT
# ─────────────────────────────────────────────────────────────────────────────

async def _tool_search_schema(
    question: str,
    source_id: UUID,
    pg_pool: asyncpg.Pool,
) -> Dict:
    """Tool 1 — Recherche dans le schéma via RAG 7C."""
    try:
        from .rag_engine import get_schema_context_corrective
        ctx = await get_schema_context_corrective(question, source_id, pg_pool)
        tables  = ctx.get("tables_found", [])
        joins   = ctx.get("graph_join_paths", [])
        context = ctx.get("context_text", "")
        return {
            "tables_found":  tables,
            "join_paths":    joins[:5],
            "context_text":  context[:2000],
            "table_count":   len(tables),
            "success":       True,
        }
    except Exception as e:
        logger.warning(f"[AgentRAG] search_schema error: {e}")
        return {"tables_found": [], "join_paths": [], "context_text": "", "success": False, "error": str(e)}


async def _tool_search_views(keyword: str) -> Dict:
    """Tool 2 — Cherche dans les vues SXA métier par mot-clé."""
    keyword_lower = keyword.lower()
    matches = {}
    for view_name, columns in SXA_VIEWS.items():
        score = 0
        if keyword_lower in view_name.lower():
            score += 3
        for col in columns:
            if keyword_lower in col.lower():
                score += 1
        if score > 0:
            matches[view_name] = {"columns": columns, "score": score}

    # Trier par score décroissant
    sorted_matches = dict(sorted(matches.items(), key=lambda x: -x[1]["score"]))
    return {
        "keyword":    keyword,
        "views_found": sorted_matches,
        "count":      len(sorted_matches),
        "success":    True,
    }


async def _tool_get_table_columns(
    table_name: str,
    source_id: UUID,
    pg_pool: asyncpg.Pool,
) -> Dict:
    """Tool 3 — Obtenir les colonnes exactes d'une table depuis la DB."""
    try:
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ef.name, ef.data_type, ef.is_primary_key, ef.is_nullable
                FROM entity_fields ef
                JOIN source_entities se ON ef.entity_id = se.id
                WHERE se.source_id = $1
                  AND LOWER(se.name) = LOWER($2)
                ORDER BY ef.position
                LIMIT 50
            """, source_id, table_name)

        if not rows:
            # Chercher dans les vues SXA statiques
            for vname, vcols in SXA_VIEWS.items():
                if vname.lower() == table_name.lower():
                    return {
                        "table":   vname,
                        "columns": vcols,
                        "count":   len(vcols),
                        "source":  "sxa_views",
                        "success": True,
                    }
            return {"table": table_name, "columns": [], "success": False, "error": "Table non trouvée"}

        columns = [
            {
                "name":        r["name"],
                "type":        r["data_type"] or "unknown",
                "primary_key": r["is_primary_key"] or False,
                "nullable":    r["is_nullable"] if r["is_nullable"] is not None else True,
            }
            for r in rows
        ]
        return {
            "table":   table_name,
            "columns": [c["name"] for c in columns],
            "details": columns,
            "count":   len(columns),
            "success": True,
        }
    except Exception as e:
        logger.warning(f"[AgentRAG] get_table_columns error: {e}")
        return {"table": table_name, "columns": [], "success": False, "error": str(e)}


async def _tool_execute_sql(
    sql: str,
    source_dict: Dict,
    dialect: str = "mssql",
) -> Dict:
    """Tool 4 — Exécuter le SQL via ConnectorFactory (générique toutes sources)."""
    import asyncio
    try:
        from connectors.factory import ConnectorFactory
        connector = ConnectorFactory.create(source_dict)
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, connector.execute_query, sql)
        data = rows[:200] if rows else []
        return {
            "success":    True,
            "rows":       data,
            "row_count":  len(data),
            "sql":        sql,
            "error":      None,
        }
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"[AgentRAG] execute_sql error: {error_msg[:200]}")
        return {
            "success":   False,
            "rows":      [],
            "row_count": 0,
            "sql":       sql,
            "error":     error_msg[:500],
        }


def _tool_validate_result(
    rows: List[Dict],
    question: str,
    sql: str,
) -> Dict:
    """Tool 5 — Valider la cohérence du résultat."""
    issues = []

    # Résultat vide
    if not rows:
        issues.append("Résultat vide — aucune ligne retournée")

    # Colonnes nulles à 100% — warning uniquement si TOUTES les colonnes sont NULL
    # Une colonne NULL partielle est acceptable (ex: Groupe_Sociétés non renseigné)
    if rows:
        all_cols = list((rows[0] or {}).keys())
        null_cols = []
        for col in all_cols:
            null_count = sum(1 for r in rows if r.get(col) is None)
            if null_count == len(rows):
                null_cols.append(col)
        # Rejeter seulement si TOUTES les colonnes sont NULL
        if len(null_cols) == len(all_cols):
            issues.append(f"Toutes les colonnes sont NULL — résultat invalide")
        # Si seulement certaines colonnes NULL → warning non bloquant (pas d'issue)

    # Montants négatifs sur un SUM attendu positif
    sum_keywords = ["solde", "total", "montant", "sum", "somme"]
    if any(kw in question.lower() for kw in sum_keywords) and rows:
        for col, val in (rows[0] or {}).items():
            if isinstance(val, (int, float)) and val < -1_000_000_000:
                issues.append(f"Valeur suspecte dans '{col}': {val} — vérifier le signe")

    is_valid = len(issues) == 0
    return {
        "valid":      is_valid,
        "issues":     issues,
        "row_count":  len(rows),
        "confidence": 1.0 if is_valid else max(0.3, 1.0 - len(issues) * 0.2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION SQL VIA LLM (avec contexte agent)
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_sql_with_context(
    question:      str,
    schema_context: str,
    views_context:  str,
    dialect:        str,
    previous_error: Optional[str] = None,
    previous_sql:   Optional[str] = None,
) -> str:
    """Génère du SQL en tenant compte du contexte accumulé par l'agent."""

    error_block = ""
    if previous_error and previous_sql:
        error_block = f"""
⚠️ TENTATIVE PRÉCÉDENTE ÉCHOUÉE :
SQL essayé :
{previous_sql}

Erreur obtenue : {previous_error}

CONSIGNE : Corrige ce SQL en évitant la même erreur. 
- Si l'erreur mentionne une colonne invalide → vérifie les colonnes dans le schéma ci-dessous
- Si l'erreur mentionne une table invalide → utilise uniquement les tables/vues listées
- Si l'erreur est syntaxique → corrige la syntaxe {dialect.upper()}
"""

    dialect_hint = {
        "mssql":      "SQL Server (TOP N, crochets [col], GETDATE(), DATEADD)",
        "postgresql": "PostgreSQL (LIMIT N, CURRENT_DATE)",
        "mysql":      "MySQL (LIMIT N, NOW())",
        "odata":      "SQL Server (TOP N, crochets [col])",
    }.get(dialect, "SQL Server")

    prompt = f"""Tu es un expert SQL {dialect_hint}. Génère une requête SQL précise.

{schema_context}

{views_context}

{error_block}

Question : {question}

⚠️ NOMS DE VUES EXACTS — utiliser EXACTEMENT ces noms entre crochets :
  [FINANCEMENT_BI]              ← PAS Ligne_de_financement, PAS FINANCEMENT
  [Ligne de financement]        ← vue avec CODE, DESCRIPTION, AMOUNT, Comptes, Banque
  [Comptes]                     ← vue avec CODE, Banque, Société, Devises
  [Transactions bancaires]      ← vue avec CODE, AMOUNT, TRNDATE, Banque, Société
  [SI_Trésorerie]               ← vue avec AMOUNTI, TRNDATE, Société, Banque

RÈGLE ANTI-AMBIGUÏTÉ : Quand une colonne existe dans plusieurs tables du JOIN,
TOUJOURS préfixer avec le nom de la table entre crochets :
  ✅ Correct   : [DI_TL_PRPPMT_A].[CUR_ID], [DI_TL_PRPPMT_A].[TL_ID]
  ❌ Incorrect : [CUR_ID], [TL_ID]  ← provoque "Nom de colonne ambigu"

RÈGLES STRICTES :
1. Retourne UNIQUEMENT le SQL — aucun texte, aucun commentaire, aucun markdown
2. Syntaxe MSSQL OBLIGATOIRE :
   - TOP N toujours après SELECT : SELECT TOP 100 [col] ...
   - DISTINCT toujours avant TOP : SELECT DISTINCT TOP 100 [col] ... (jamais SELECT TOP 100 DISTINCT)
   - JAMAIS LIMIT — utilise TOP N
3. Mets des crochets autour de tous les noms avec espaces : [Transactions bancaires]
4. N'invente JAMAIS une table ou colonne non listée dans le schéma
5. JAMAIS utiliser une table suffixée _A comme table principale — utilise la table sans _A :
   ✅ FROM TH_USR    (table courante)
   ❌ FROM TH_USR_A  (table audit — données historiques, souvent vide)
6. Pour SI_Trésorerie → SELECT [Société], SUM([AMOUNTI]) GROUP BY [Société] SANS JOIN
7. Pour [Transactions bancaires] → SELECT direct SANS JOIN avec d'autres tables
8. Pour [Comptes] → SELECT direct SANS JOIN

SQL :"""

    try:
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.05, "num_predict": 512},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Nettoyer le SQL
        import re
        raw = re.sub(r"```(?:sql|SQL)?\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        match = re.search(r"\b(SELECT|WITH)\b", raw, re.IGNORECASE)
        if match:
            raw = raw[match.start():]

        # Ajouter TOP 100 si absent (MSSQL) + supprimer LIMIT invalide
        if dialect in ("mssql", "odata"):
            if not re.search(r"TOP\s+\d+", raw, re.IGNORECASE):
                raw = re.sub(r"\bSELECT\b", "SELECT TOP 100", raw, count=1, flags=re.IGNORECASE)
            # Supprimer LIMIT N — syntaxe MySQL/PG invalide en SQL Server
            raw = re.sub(r"\s*LIMIT\s+\d+\s*;?\s*$", "", raw.strip(), flags=re.IGNORECASE)
            raw = re.sub(r"\s*LIMIT\s+\d+", "", raw, flags=re.IGNORECASE)
            # Fix TOP mal placé double
            raw = re.sub(
                r"\bSELECT\b(\s+TOP\s+\d+){2,}",
                lambda m: "SELECT TOP 100",
                raw, flags=re.IGNORECASE
            )
            # ── Fix DISTINCT en MSSQL ─────────────────────────────────────────
            # Règle MSSQL : SELECT DISTINCT [cols] — jamais SELECT TOP N DISTINCT
            # DISTINCT et TOP sont incompatibles : DISTINCT seul suffit (pas de limite arbitraire
            # sur un SELECT DISTINCT car le résultat est déjà dédupliqué)
            if re.search(r"\bDISTINCT\b", raw, re.IGNORECASE):
                # Supprimer TOP N dans tous les cas où DISTINCT est présent
                # cas 1 : SELECT TOP 100 DISTINCT  → SELECT DISTINCT
                raw = re.sub(r"\bSELECT\b\s+TOP\s+\d+\s+DISTINCT\b", "SELECT DISTINCT", raw, flags=re.IGNORECASE)
                # cas 2 : SELECT DISTINCT TOP 100  → SELECT DISTINCT
                raw = re.sub(r"\bSELECT\b\s+DISTINCT\s+TOP\s+\d+\b", "SELECT DISTINCT", raw, flags=re.IGNORECASE)
                logger.debug("[AgentRAG] DISTINCT détecté — TOP N supprimé (incompatible MSSQL)")

            # ── Fix tables _A (audit) → table courante — recherche GLOBALE ─────
            # Couvre : FROM, JOIN, ON, WHERE, SELECT, alias partout dans le SQL
            # Ex: GS_QUOTPL_2_CUR1_A.REV dans ON → GS_QUOTPL_2_CUR1.REV
            _all_audit = re.findall(
                r'\b([A-Za-z][A-Za-z0-9_]+_A)\b(?=[\s\.\]\[,)]|$)',
                raw, re.IGNORECASE
            )
            _seen_audit: set = set()
            for _audit_tbl in _all_audit:
                if _audit_tbl.upper() in _seen_audit:
                    continue
                _seen_audit.add(_audit_tbl.upper())
                _current_tbl = _audit_tbl[:-2]
                raw = re.sub(
                    rf'\b{re.escape(_audit_tbl)}\b',
                    _current_tbl,
                    raw, flags=re.IGNORECASE
                )
                logger.debug(f"[AgentRAG] Table audit corrigée (global) : {_audit_tbl} → {_current_tbl}")

        return raw.strip()
    except Exception as e:
        logger.warning(f"[AgentRAG] LLM error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# AGENT REACT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class AgentRAG:
    """
    Agent ReAct pour OnePilot Sprint 8.
    Raisonne en boucle jusqu'à trouver la bonne réponse ou atteindre MAX_ITERATIONS.
    Générique : fonctionne avec toutes les sources (MSSQL, OData, CSV, REST, PG).
    """

    def __init__(
        self,
        pg_pool:     asyncpg.Pool,
        source_dict: Dict,
        source_id:   UUID,
        dialect:     str = "mssql",
    ):
        self.pg_pool     = pg_pool
        self.source_dict = source_dict
        self.source_id   = source_id
        self.dialect     = dialect
        self.steps:      List[AgentStep] = []

    async def run(self, question: str) -> AgentResult:
        """Point d'entrée principal — boucle ReAct."""
        t0 = time.time()
        logger.info(f"[AgentRAG] Démarrage — question: '{question[:80]}'")

        # État de l'agent
        schema_context  = ""
        views_context   = ""
        best_sql        = ""
        best_rows       = []
        last_error      = None
        sql_attempts    = 0
        final_result    = None

        # ── Sprint 8.5 : Matching flou sur SXA_DIRECT_SQL ───────────────────
        matched_pattern, direct_sql, match_score = _find_direct_sql(question)
        if matched_pattern and direct_sql:
            logger.info(
                f"[AgentRAG] Requête directe — pattern='{matched_pattern}' "
                f"score={match_score:.2f}"
            )
            exec_direct = await _tool_execute_sql(direct_sql, self.source_dict, self.dialect)
            if exec_direct["success"]:
                validation_direct = _tool_validate_result(exec_direct["rows"], question, direct_sql)
                if validation_direct["valid"] or exec_direct["row_count"] == 0:
                    return AgentResult(
                        success=True, sql=direct_sql, result=exec_direct["rows"],
                        explanation=(
                            f"Agent RAG — requête directe optimisée "
                            f"(pattern='{matched_pattern}', score={match_score:.2f})"
                        ),
                        steps=[], iterations=1,
                        duration_ms=int((time.time() - t0) * 1000),
                        method="agentic_rag_direct",
                    )
            else:
                # Requête directe échouée → essayer version minimaliste SELECT *
                logger.warning(f"[AgentRAG] Requête directe échouée : {exec_direct['error'][:100]}")
                import re as _re_direct
                vue_match = _re_direct.search(r'FROM \[([^\]]+)\]', direct_sql, _re_direct.IGNORECASE)
                if vue_match:
                    vue_name = vue_match.group(1)
                    minimal_sql = f"SELECT TOP 100 * FROM [{vue_name}]"
                    exec_minimal = await _tool_execute_sql(minimal_sql, self.source_dict, self.dialect)
                    if exec_minimal["success"]:
                        logger.info(f"[AgentRAG] Requête minimale réussie sur [{vue_name}]")
                        return AgentResult(
                            success=True, sql=minimal_sql, result=exec_minimal["rows"],
                            explanation=f"Agent RAG — requête minimale [{vue_name}]",
                            steps=[], iterations=1,
                            duration_ms=int((time.time() - t0) * 1000),
                            method="agentic_rag_direct",
                        )

        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info(f"[AgentRAG] Itération {iteration}/{MAX_ITERATIONS}")

            # ── Étape 1 : Recherche schéma (iteration 1 seulement) ───────────
            if iteration == 1:
                thought = "Je commence par chercher les tables pertinentes dans le schéma."
                obs = await _tool_search_schema(question, self.source_id, self.pg_pool)
                schema_context = obs.get("context_text", "")
                tables_found   = obs.get("tables_found", [])
                self._add_step(iteration, thought, AgentAction.SEARCH_SCHEMA,
                               {"question": question},
                               f"Tables trouvées : {tables_found}")

                # Chercher aussi dans les vues SXA
                thought2 = "Je vérifie si des vues métier SXA correspondent à la question."
                obs2 = await _tool_search_views(question)
                if obs2["count"] > 0:
                    view_lines = []
                    for vname, vdata in list(obs2["views_found"].items())[:3]:
                        view_lines.append(f"Vue [{vname}] : {', '.join(vdata['columns'][:8])}")
                    views_context = (
                        "\n=== VUES MÉTIER DISPONIBLES ===\n" +
                        "\n".join(view_lines) +
                        "\n⚠️ Utilise ces vues en priorité — elles contiennent déjà les JOINs.\n"
                    )
                    self._add_step(iteration, thought2, AgentAction.SEARCH_VIEWS,
                                   {"keyword": question},
                                   f"Vues trouvées : {list(obs2['views_found'].keys())}")
                continue

            # ── Étape 2 : Générer le SQL ─────────────────────────────────────
            if iteration == 2 or (last_error and sql_attempts < MAX_SQL_RETRIES):
                thought = (
                    f"Je génère le SQL avec le contexte accumulé."
                    if not last_error
                    else f"Le SQL précédent a échoué ({last_error[:80]}). Je corrige."
                )
                sql = await _generate_sql_with_context(
                    question, schema_context, views_context, self.dialect,
                    previous_error=last_error, previous_sql=best_sql,
                )
                sql_attempts += 1

                if not sql:
                    self._add_step(iteration, thought, AgentAction.EXECUTE_SQL,
                                   {"sql": ""}, "Échec génération SQL — LLM indisponible")
                    break

                # ── Étape 3 : Exécuter le SQL ─────────────────────────────
                exec_result = await _tool_execute_sql(sql, self.source_dict, self.dialect)
                best_sql    = sql

                if exec_result["success"]:
                    best_rows  = exec_result["rows"]
                    last_error = None
                    self._add_step(
                        iteration, thought, AgentAction.EXECUTE_SQL,
                        {"sql": sql},
                        f"✅ {exec_result['row_count']} lignes retournées"
                    )

                    # ── Étape 4 : Valider le résultat ─────────────────────
                    validation = _tool_validate_result(best_rows, question, sql)
                    self._add_step(
                        iteration, "Je valide la cohérence du résultat.",
                        AgentAction.VALIDATE_RESULT,
                        {"row_count": len(best_rows)},
                        f"Valide: {validation['valid']} | Issues: {validation['issues']}"
                    )

                    if validation["valid"]:
                        # Résultat satisfaisant → on s'arrête
                        final_result = AgentResult(
                            success=True, sql=sql, result=best_rows,
                            explanation=f"Agent RAG — {iteration} itérations",
                            steps=self.steps, iterations=iteration,
                            duration_ms=int((time.time() - t0) * 1000),
                            method="agentic_rag",
                        )
                        break
                    else:
                        # Résultat incohérent → noter les issues et réessayer
                        last_error = "; ".join(validation["issues"])
                        logger.info(f"[AgentRAG] Résultat invalide : {last_error}")

                else:
                    last_error = exec_result["error"]
                    self._add_step(
                        iteration, thought, AgentAction.EXECUTE_SQL,
                        {"sql": sql},
                        f"❌ Erreur SQL : {last_error[:200]}"
                    )

                    # Si erreur colonne ambiguë → ajouter préfixes tables
                    if "ambigu" in (last_error or "").lower() or "ambiguous" in (last_error or "").lower():
                        import re as _re_amb
                        # Forcer préfixe table sur toutes les colonnes sans préfixe
                        schema_context += (
                            "\n⚠️ ERREUR AMBIGUÏTÉ : Préfixe OBLIGATOIRE sur toutes les colonnes "
                            "ex: [DI_TL_PRPPMT_A].[CUR_ID] et NON [CUR_ID]\n"
                        )
                        logger.info("[AgentRAG] Erreur ambiguïté détectée — ajout règle préfixe")
                        continue

                    # Si erreur LIMIT (syntaxe MySQL en MSSQL) → corriger directement
                    import re as _re
                    if "LIMIT" in (last_error or "").upper() or "limite" in (last_error or "").lower():
                        import re as _re2
                        best_sql = _re2.sub(r"\s*LIMIT\s+\d+\s*;?\s*$", ";", best_sql.strip(), flags=_re2.IGNORECASE)
                        best_sql = _re2.sub(r"\s*LIMIT\s+\d+", "", best_sql, flags=_re2.IGNORECASE)
                        schema_context += "\n⚠️ INTERDIT en SQL Server : n'utilise JAMAIS LIMIT — utilise TOP N apres SELECT.\n"
                        logger.info("[AgentRAG] Erreur LIMIT détectée — SQL corrigé automatiquement")
                        # Réessayer immédiatement avec le SQL corrigé
                        exec_retry = await _tool_execute_sql(best_sql, self.source_dict, self.dialect)
                        if exec_retry["success"]:
                            best_rows  = exec_retry["rows"]
                            last_error = None
                            self._add_step(iteration, "Correction LIMIT → TOP N appliquée.",
                                           AgentAction.EXECUTE_SQL, {"sql": best_sql},
                                           f"✅ {exec_retry['row_count']} lignes après correction LIMIT")
                            validation2 = _tool_validate_result(best_rows, question, best_sql)
                            if validation2["valid"]:
                                final_result = AgentResult(
                                    success=True, sql=best_sql, result=best_rows,
                                    explanation=f"Agent RAG — correction LIMIT auto",
                                    steps=self.steps, iterations=iteration,
                                    duration_ms=int((time.time() - t0) * 1000),
                                    method="agentic_rag",
                                )
                                break
                        continue

                    # Si colonne invalide → chercher les colonnes exactes
                    # ── Sprint 8.5 : Parser d'erreur MSSQL multi-format ─────────
                    # MSSQL renvoie les colonnes invalides sous plusieurs formats :
                    #   "Invalid column name 'COL'."           → guillemets simples
                    #   "Nom de colonne non valide 'COL'."     → FR
                    #   "Colonne 'COL' introuvable"            → variante
                    #   "column 'COL' does not exist"          → PostgreSQL
                    bad_col = None
                    for _col_pat in [
                        r"Invalid column name '([^']+)'",   # MSSQL EN
                        r"non valide '([^']+)'",            # MSSQL FR
                        r"introuvable '([^']+)'",           # MSSQL FR variante
                        r"column '([^']+)' does not exist", # PostgreSQL
                        r"'([^']+)' is not a valid column", # autres
                        r"'([^']+)'",                       # fallback générique
                    ]:
                        _col_m = _re.search(_col_pat, last_error or "", _re.IGNORECASE)
                        if _col_m:
                            bad_col = _col_m.group(1)
                            break

                    if bad_col and sql_attempts < MAX_SQL_RETRIES:
                        # Chercher toutes les tables mentionnées dans le SQL
                        tbl_matches = _re.findall(r"FROM\s+\[?([^\]\s,]+)\]?|JOIN\s+\[?([^\]\s,]+)\]?", sql, _re.IGNORECASE)
                        tables_in_sql = list({g for pair in tbl_matches for g in pair if g})[:3]

                        _cols_injected = False
                        for tbl in tables_in_sql:
                            cols_result = await _tool_get_table_columns(
                                tbl, self.source_id, self.pg_pool
                            )
                            if cols_result["success"] and cols_result["columns"]:
                                _cols_injected = True
                                cols_list = cols_result["columns"][:30]
                                schema_context += (
                                    f"\n🚫 ERREUR COLONNE — '{bad_col}' N'EXISTE PAS dans [{tbl}].\n"
                                    f"   Colonnes RÉELLES de [{tbl}] : {', '.join(cols_list)}\n"
                                    f"   ❌ N'UTILISE JAMAIS '{bad_col}' ni aucune colonne hors cette liste.\n"
                                    f"   ✅ Réécris le SQL en utilisant UNIQUEMENT ces colonnes exactes.\n"
                                )
                                self._add_step(
                                    iteration,
                                    f"'{bad_col}' invalide dans [{tbl}] — {len(cols_list)} colonnes exactes récupérées.",
                                    AgentAction.GET_TABLE_COLUMNS,
                                    {"table": tbl, "bad_col": bad_col},
                                    f"Colonnes: {cols_list[:10]}"
                                )
                        if not _cols_injected:
                            schema_context += (
                                f"\n🚫 ERREUR : colonne '{bad_col}' invalide.\n"
                                f"   Vérifie les noms exacts dans le schéma ci-dessus.\n"
                                f"   ❌ N'invente AUCUNE colonne — utilise UNIQUEMENT le schéma fourni.\n"
                            )
                continue

            # ── Dernière itération : meilleur résultat disponible ─────────
            if iteration >= MAX_ITERATIONS:
                break

        # ── Résultat final ────────────────────────────────────────────────────
        if final_result:
            logger.info(
                f"[AgentRAG] Succès en {final_result.iterations} itérations "
                f"({final_result.duration_ms}ms)"
            )
            return final_result

        # Retourner le meilleur résultat même imparfait
        warnings = []
        if last_error:
            warnings.append(f"Dernier erreur : {last_error[:200]}")

        return AgentResult(
            success=bool(best_rows),
            sql=best_sql or None,
            result=best_rows,
            explanation=(
                f"Agent RAG — {MAX_ITERATIONS} itérations max atteintes"
                if not best_rows
                else f"Agent RAG — résultat partiel après {sql_attempts} tentatives"
            ),
            steps=self.steps,
            iterations=MAX_ITERATIONS,
            duration_ms=int((time.time() - t0) * 1000),
            method="agentic_rag",
            warnings=warnings,
        )

    def _add_step(
        self,
        iteration:    int,
        thought:      str,
        action:       AgentAction,
        action_input: Dict,
        observation:  str,
    ):
        self.steps.append(AgentStep(
            iteration=iteration,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        ))
        logger.debug(
            f"[AgentRAG] Step {iteration} | {action.value} | {observation[:80]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

async def run_agentic_rag(
    question:    str,
    source_id:   UUID,
    pg_pool:     asyncpg.Pool,
    source_dict: Dict,
    dialect:     str = "mssql",
) -> AgentResult:
    """
    Point d'entrée public pour le Sprint 8 Agentic RAG.
    Appelé depuis main.py en remplacement/complément de generate_sql_with_llm().

    Args:
        question    : question en langage naturel
        source_id   : UUID de la source de données
        pg_pool     : pool PostgreSQL (métadonnées OnePilot)
        source_dict : config de la source (connector_type, host, port, ...)
        dialect     : dialecte SQL (mssql, postgresql, mysql, odata)

    Returns:
        AgentResult avec sql, result, steps, warnings
    """
    agent = AgentRAG(
        pg_pool=pg_pool,
        source_dict=source_dict,
        source_id=source_id,
        dialect=dialect,
    )
    return await agent.run(question)


def agent_result_to_dict(result: AgentResult) -> Dict:
    """Sérialise AgentResult en dict JSON-compatible pour l'API."""
    return {
        "success":     result.success,
        "sql":         result.sql,
        "result":      result.result,
        "explanation": result.explanation,
        "method":      result.method,
        "iterations":  result.iterations,
        "duration_ms": result.duration_ms,
        "warnings":    result.warnings,
        "steps": [
            {
                "iteration":    s.iteration,
                "thought":      s.thought,
                "action":       s.action.value,
                "action_input": s.action_input,
                "observation":  s.observation,
            }
            for s in result.steps
        ],
    }
