"""
OnePilot — ConnectorFactory
Dispatche vers le bon connecteur selon connector_type de la source.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# BASE CONNECTOR (inline — évite import circulaire)
# ══════════════════════════════════════════════════════════════

class BaseConnector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connected = False

    def connect(self) -> bool:
        raise NotImplementedError

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        raise NotImplementedError

    def disconnect(self):
        self._connected = False

    def _get_password(self) -> str:
        import json as _j
        for key in ["secrets_json", "config_json", "credentials_json"]:
            val = self.config.get(key)
            if not val:
                continue
            if isinstance(val, str):
                try: val = _j.loads(val)
                except: continue
            if isinstance(val, dict):
                pwd = val.get("password") or val.get("db_password") or val.get("pwd", "")
                if pwd:
                    return str(pwd)
        return self.config.get("password", "")


# ══════════════════════════════════════════════════════════════
# MSSQL CONNECTOR
# ══════════════════════════════════════════════════════════════

class MSSQLConnector(BaseConnector):
    """Connecteur SQL Server via pyodbc."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def _build_conn_str(self) -> str:
        host = self.config.get("host", "")
        port = self.config.get("port", 1433)
        db   = self.config.get("database_name", "")
        user = self.config.get("username", "")
        pwd  = self._get_password()
        return (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={host},{port};DATABASE={db};"
            f"UID={user};PWD={pwd};"
            f"TrustServerCertificate=yes;Encrypt=no;"
        )

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        try:
            import pyodbc
            with pyodbc.connect(self._build_conn_str(), timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                if cursor.description is None:
                    return []
                cols = [col[0] for col in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchmany(500)]
        except Exception as e:
            logger.error(f"[MSSQLConnector] execute_query error: {e}")
            raise


# ══════════════════════════════════════════════════════════════
# POSTGRESQL CONNECTOR
# ══════════════════════════════════════════════════════════════

class PostgreSQLConnector(BaseConnector):
    """Connecteur PostgreSQL via psycopg2."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        try:
            import psycopg2
            import psycopg2.extras
            with psycopg2.connect(
                host     = self.config.get("host", ""),
                port     = self.config.get("port", 5432),
                dbname   = self.config.get("database_name", ""),
                user     = self.config.get("username", ""),
                password = self._get_password(),
                connect_timeout = 15,
            ) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, params or {})
                    return [dict(r) for r in cur.fetchmany(500)]
        except Exception as e:
            logger.error(f"[PostgreSQLConnector] execute_query error: {e}")
            raise


# ══════════════════════════════════════════════════════════════
# ODATA CONNECTOR
# ══════════════════════════════════════════════════════════════

class ODataConnector(BaseConnector):
    """Connecteur OData REST."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        try:
            import requests
            base_url = self.config.get("base_url", "").rstrip("/")
            url = f"{base_url}{query}" if query.startswith("/") else f"{base_url}/{query}"
            headers = {"Accept": "application/json", "OData-MaxVersion": "4.0"}
            token = self._get_password()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("value", [data])
        except Exception as e:
            logger.error(f"[ODataConnector] execute_query error: {e}")
            raise


# ══════════════════════════════════════════════════════════════
# REST CONNECTOR
# ══════════════════════════════════════════════════════════════

class RESTConnector(BaseConnector):
    """Connecteur REST API générique."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        try:
            import requests
            base_url = self.config.get("base_url", "").rstrip("/")
            url = f"{base_url}{query}" if query.startswith("/") else query
            headers = {"Accept": "application/json"}
            token = self._get_password()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        except Exception as e:
            logger.error(f"[RESTConnector] execute_query error: {e}")
            raise


# ══════════════════════════════════════════════════════════════
# EXCEL CONNECTOR
# ══════════════════════════════════════════════════════════════

class ExcelConnector(BaseConnector):
    """Connecteur fichier Excel."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        try:
            import pandas as pd
            file_path = self.config.get("file_path", "")
            df = pd.read_excel(file_path)
            return df.head(500).to_dict(orient="records")
        except Exception as e:
            logger.error(f"[ExcelConnector] execute_query error: {e}")
            raise


# ══════════════════════════════════════════════════════════════
# CONNECTOR FACTORY
# ══════════════════════════════════════════════════════════════

class ConnectorFactory:
    """
    Factory principale — crée le bon connecteur selon connector_type.
    Utilisé par main.py, dashboard_engine.py, execute-plan route.
    """

    # Mapping connector_type → classe
    _REGISTRY: Dict[str, type] = {
        "mssql":        MSSQLConnector,
        "sage_100":     MSSQLConnector,
        "sage_x3":      MSSQLConnector,
        "postgresql":   PostgreSQLConnector,
        "mysql":        MSSQLConnector,   # via pyodbc aussi
        "odata":        ODataConnector,
        "rest_api":     RESTConnector,
        "graphql":      RESTConnector,
        "excel":        ExcelConnector,
        "dynamics_365": ODataConnector,
        "sap_odata":    ODataConnector,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseConnector:
        """
        Crée et retourne un connecteur pour la source donnée.

        Args:
            config: dict de la source (model_dump() d'un objet Source)

        Returns:
            Instance de connecteur avec execute_query() disponible
        """
        # Normalise connector_type
        raw_type = config.get("connector_type", "")
        if hasattr(raw_type, "value"):
            raw_type = raw_type.value
        ct = str(raw_type).lower().replace("-", "_")

        connector_class = cls._REGISTRY.get(ct)

        if connector_class is None:
            # Fallback : essaie de détecter depuis l'URL ou le host
            if config.get("base_url"):
                connector_class = RESTConnector
            elif config.get("host"):
                connector_class = MSSQLConnector
            else:
                raise ValueError(
                    f"Connector type '{ct}' non supporté. "
                    f"Types disponibles : {list(cls._REGISTRY.keys())}"
                )

        connector = connector_class(config)
        connector.connect()
        logger.debug(f"[ConnectorFactory] Créé {connector_class.__name__} pour type='{ct}'")
        return connector

    @classmethod
    def register(cls, connector_type: str, connector_class: type):
        """Enregistre un nouveau type de connecteur."""
        cls._REGISTRY[connector_type.lower()] = connector_class
