"""
OnePilot – Pydantic Schemas
Validation des requêtes et sérialisation des réponses
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────

class ConnectorType(str, Enum):
    # Databases
    POSTGRESQL  = "postgresql"
    MYSQL       = "mysql"
    MSSQL       = "mssql"
    SQLITE      = "sqlite"
    # Web Services
    REST        = "rest"
    ODATA       = "odata"
    GRAPHQL     = "graphql"
    SOAP        = "soap"
    # SAP
    SAP_RFC     = "sap_rfc"
    SAP_HANA    = "sap_hana"
    SAP_ODATA   = "sap_odata"       # SAP S/4HANA OData / SAP Gateway
    # Microsoft Dynamics 365 / Dataverse
    DYNAMICS365 = "dynamics365"
    # SAGE
    SAGE_X3     = "sage_x3"         # SAGE X3 Web Services
    SAGE_100    = "sage_100"         # SAGE 100 (SQL Server)
    SAGE_CLOUD  = "sage_cloud"       # SAGE Business Cloud API
    # Files
    FILE_CSV    = "file_csv"
    FILE_EXCEL  = "file_excel"
    FILE_XLSX   = "file_xlsx"
    FILE_JSON   = "file_json"
    FILE_PARQUET = "file_parquet"
    FILE_AVRO   = "file_avro"
    # Custom plugins uploadés via la console admin
    CUSTOM      = "custom"


class SourceCategory(str, Enum):
    DATABASE   = "database"
    WEBSERVICE = "webservice"
    FILE       = "file"
    CUSTOM     = "custom"


class AuthType(str, Enum):
    NONE         = "none"
    BASIC        = "basic"
    BEARER       = "bearer"
    OAUTH2       = "oauth2"
    API_KEY      = "api_key"
    WINDOWS_AUTH = "windows_auth"


class ConnectionStatus(str, Enum):
    PENDING  = "pending"
    ACTIVE   = "active"
    ERROR    = "error"
    DISABLED = "disabled"


# ── Category mapping ──────────────────────────────────────────

CONNECTOR_CATEGORY_MAP: Dict[ConnectorType, SourceCategory] = {
    # Databases
    ConnectorType.POSTGRESQL:  SourceCategory.DATABASE,
    ConnectorType.MYSQL:       SourceCategory.DATABASE,
    ConnectorType.MSSQL:       SourceCategory.DATABASE,
    ConnectorType.SQLITE:      SourceCategory.DATABASE,
    ConnectorType.SAP_HANA:    SourceCategory.DATABASE,
    ConnectorType.SAGE_100:    SourceCategory.DATABASE,   # SAGE 100 = SQL Server
    # Web Services
    ConnectorType.REST:        SourceCategory.WEBSERVICE,
    ConnectorType.ODATA:       SourceCategory.WEBSERVICE,
    ConnectorType.GRAPHQL:     SourceCategory.WEBSERVICE,
    ConnectorType.SOAP:        SourceCategory.WEBSERVICE,
    ConnectorType.SAP_RFC:     SourceCategory.WEBSERVICE,
    ConnectorType.SAP_ODATA:   SourceCategory.WEBSERVICE,
    ConnectorType.DYNAMICS365: SourceCategory.WEBSERVICE,
    ConnectorType.SAGE_X3:     SourceCategory.WEBSERVICE,
    ConnectorType.SAGE_CLOUD:  SourceCategory.WEBSERVICE,
    # Files
    ConnectorType.FILE_CSV:    SourceCategory.FILE,
    ConnectorType.FILE_EXCEL:  SourceCategory.FILE,
    ConnectorType.FILE_XLSX:   SourceCategory.FILE,
    ConnectorType.FILE_JSON:   SourceCategory.FILE,
    ConnectorType.FILE_PARQUET:SourceCategory.FILE,
    ConnectorType.FILE_AVRO:   SourceCategory.FILE,
    # Custom
    ConnectorType.CUSTOM:      SourceCategory.CUSTOM,
}


# ── Auth configs ──────────────────────────────────────────────

class AuthNone(BaseModel):
    type: AuthType = AuthType.NONE

class AuthBasic(BaseModel):
    type: AuthType = AuthType.BASIC
    username: str
    password: str

class AuthBearer(BaseModel):
    type: AuthType = AuthType.BEARER
    token: str

class AuthOAuth2(BaseModel):
    type: AuthType = AuthType.OAUTH2
    token_url: str
    client_id: str
    client_secret: str
    scope: Optional[str] = None

class AuthApiKey(BaseModel):
    type: AuthType = AuthType.API_KEY
    header: str = "X-API-Key"
    value: str


# ── Source Create/Update ──────────────────────────────────────

class DataSourceCreate(BaseModel):
    name:           str = Field(..., min_length=1, max_length=255)
    description:    Optional[str] = None
    connector_type: ConnectorType

    @field_validator('connector_type', mode='before')
    @classmethod
    def accept_custom_connector_types(cls, v):
        """Accepte les types custom (plugins) en les mappant sur CUSTOM."""
        known = {e.value for e in ConnectorType}
        if isinstance(v, str) and v not in known:
            # Type inconnu = plugin custom
            return ConnectorType.CUSTOM
        return v

    # DB fields
    host:           Optional[str] = None
    port:           Optional[int] = None
    database_name:  Optional[str] = None
    schema_name:    Optional[str] = None

    # Web service fields
    base_url:       Optional[str] = None

    # Auth
    auth_type:      AuthType = AuthType.NONE
    username:       Optional[str] = None
    password:       Optional[str] = None
    token:          Optional[str] = None
    client_id:      Optional[str] = None
    client_secret:  Optional[str] = None
    token_url:      Optional[str] = None
    scope:          Optional[str] = None
    api_key_header: Optional[str] = Field(default="X-API-Key")
    api_key_value:  Optional[str] = None

    # Options (paramètres spécifiques par connecteur)
    options:        Dict[str, Any] = Field(default_factory=dict)
    tags:           List[str]      = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_fields(self) -> "DataSourceCreate":
        category = CONNECTOR_CATEGORY_MAP.get(self.connector_type)
        ct = self.connector_type

        if category == SourceCategory.DATABASE:
            # SAGE 100 utilise SQL Server — host + database requis
            if not self.host:
                raise ValueError("host est requis pour une connexion base de données")
            if not self.database_name:
                raise ValueError("database_name est requis pour une connexion base de données")

        elif category == SourceCategory.WEBSERVICE:
            # SAP RFC : host requis (pas de base_url)
            if ct == ConnectorType.SAP_RFC:
                if not self.host:
                    raise ValueError("host est requis pour SAP RFC (ex: sap.company.com)")
            # Tous les autres web services : base_url requise
            elif not self.base_url:
                raise ValueError("base_url est requis pour un web service")

        elif category == SourceCategory.CUSTOM:
            # Plugins custom : aucun champ obligatoire
            pass

        return self


class DataSourceUpdate(BaseModel):
    name:           Optional[str] = None
    description:    Optional[str] = None
    host:           Optional[str] = None
    port:           Optional[int] = None
    database_name:  Optional[str] = None
    schema_name:    Optional[str] = None
    base_url:       Optional[str] = None
    auth_type:      Optional[AuthType] = None
    username:       Optional[str] = None
    password:       Optional[str] = None
    token:          Optional[str] = None
    options:        Optional[Dict[str, Any]] = None
    tags:           Optional[List[str]] = None


# ── Response Models ───────────────────────────────────────────

class FieldOut(BaseModel):
    id:             UUID
    name:           str
    display_name:   Optional[str]
    data_type:      str
    native_type:    Optional[str]
    is_nullable:    bool
    is_primary_key: bool
    is_foreign_key: bool
    position:       int

class EntityOut(BaseModel):
    id:           UUID
    name:         str
    display_name: Optional[str]
    entity_type:  str
    description:  Optional[str]
    row_count:    Optional[int]
    field_count:  int = 0
    fields:       List[FieldOut] = []

class RelationOut(BaseModel):
    id:               UUID
    source_entity_id: UUID
    target_entity_id: UUID
    source_field:     str
    target_field:     str
    relation_type:    str
    confidence:       float
    is_confirmed:     bool

class ConnectionTestResult(BaseModel):
    source_id:  UUID
    success:    bool
    message:    str
    latency_ms: int
    tested_at:  datetime

class DataSourceOut(BaseModel):
    id:              UUID
    name:            str
    description:     Optional[str]
    category:        SourceCategory
    connector_type:  ConnectorType
    status:          ConnectionStatus
    host:            Optional[str]
    port:            Optional[int]
    database_name:   Optional[str]
    schema_name:     Optional[str]
    base_url:        Optional[str]
    auth_type:       AuthType
    username:        Optional[str]
    options:         Dict[str, Any]
    tags:            List[str]
    entity_count:    int
    test_latency_ms: Optional[int]
    error_message:   Optional[str]
    created_at:      datetime
    updated_at:      datetime
    last_tested_at:  Optional[datetime]
    last_synced_at:  Optional[datetime]

class DataSourceDetail(DataSourceOut):
    entities: List[EntityOut] = []

class DataSourceList(BaseModel):
    total:   int
    sources: List[DataSourceOut]

class MetadataSyncResult(BaseModel):
    source_id:      UUID
    success:        bool
    entity_count:   int
    field_count:    int
    relation_count: int
    duration_ms:    int
    message:        str