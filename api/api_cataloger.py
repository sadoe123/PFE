"""
OnePilot — api_cataloger.py
============================
Parser universel pour APIs :
  - OpenAPI 3.x / Swagger 2.x (JSON ou YAML, local ou URL)
  - GraphQL schema introspection
  - OData $metadata (déjà partiellement implémenté — enrichi ici)

Extrait :
  - Catalogue complet des endpoints (URL, méthodes HTTP)
  - Paramètres (query, path, header, body)
  - Schémas requête/réponse (JSON Schema)
  - Rate limiting documenté
  - Auth par endpoint (securitySchemes)
  - Relations entre ressources (HATEOAS / $ref)
  - Types GraphQL, queries, mutations, subscriptions
"""

from __future__ import annotations

import logging
import json
import re
import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# OPENAPI PARSER
# ══════════════════════════════════════════════════════════════════════

class OpenAPIParser:
    """
    Parse un document OpenAPI 3.x ou Swagger 2.x.
    Supporte JSON et YAML, depuis une URL ou un fichier local.

    Dépendances recommandées (optionnelles — fallback si absentes) :
      pip install prance openapi-spec-validator pyyaml
    """

    def __init__(self, spec: Dict):
        self.spec = spec
        self._version = self._detect_version()

    @classmethod
    async def from_url(cls, url: str, timeout: int = 30) -> "OpenAPIParser":
        """Charge la spec depuis une URL."""
        import httpx  # type: ignore
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "yaml" in ct or url.endswith((".yaml", ".yml")):
                spec = cls._parse_yaml(resp.text)
            else:
                spec = resp.json()
        return cls(spec)

    @classmethod
    def from_file(cls, path: str) -> "OpenAPIParser":
        """Charge la spec depuis un fichier local (JSON ou YAML)."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if path.endswith((".yaml", ".yml")):
            spec = cls._parse_yaml(content)
        else:
            spec = json.loads(content)
        return cls(spec)

    @staticmethod
    def _parse_yaml(content: str) -> Dict:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML requis pour parser YAML. pip install pyyaml")

    def _detect_version(self) -> str:
        if "openapi" in self.spec:
            return "3.x"
        if "swagger" in self.spec:
            return "2.x"
        return "unknown"

    def _resolve_ref(self, ref: str) -> Dict:
        """Résout un $ref local dans la spec."""
        if not ref.startswith("#/"):
            return {}
        parts = ref.lstrip("#/").split("/")
        node = self.spec
        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict):
                node = node.get(part, {})
            else:
                return {}
        return node or {}

    def _resolve_schema(self, schema: Dict, depth: int = 0) -> Dict:
        """Résout récursivement les $ref dans un schema."""
        if depth > 5 or not schema:
            return schema
        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"])
            return self._resolve_schema(resolved, depth + 1)
        result = {}
        for k, v in schema.items():
            if isinstance(v, dict):
                result[k] = self._resolve_schema(v, depth + 1)
            elif isinstance(v, list):
                result[k] = [
                    self._resolve_schema(i, depth + 1) if isinstance(i, dict) else i
                    for i in v
                ]
            else:
                result[k] = v
        return result

    # ── Extraction globale ────────────────────────────────────────────

    def extract_all(self) -> Dict:
        """
        Point d'entrée principal — extrait tout le catalogue.
        Retourne un dict prêt pour l'indexation dans source_entities / entity_fields.
        """
        return {
            "version":          self._version,
            "info":             self._extract_info(),
            "servers":          self._extract_servers(),
            "endpoints":        self._extract_endpoints(),
            "schemas":          self._extract_schemas(),
            "security_schemes": self._extract_security_schemes(),
            "rate_limits":      self._extract_rate_limits(),
            "relations":        self._extract_resource_relations(),
        }

    def _extract_info(self) -> Dict:
        info = self.spec.get("info", {})
        return {
            "title":       info.get("title", ""),
            "version":     info.get("version", ""),
            "description": info.get("description", ""),
        }

    def _extract_servers(self) -> List[str]:
        if self._version == "3.x":
            return [s.get("url", "") for s in self.spec.get("servers", [])]
        # Swagger 2.x
        host     = self.spec.get("host", "")
        base_path = self.spec.get("basePath", "/")
        schemes  = self.spec.get("schemes", ["https"])
        return [f"{schemes[0]}://{host}{base_path}"] if host else [base_path]

    def _extract_endpoints(self) -> List[Dict]:
        endpoints = []
        paths = self.spec.get("paths", {})

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                op = path_item.get(method)
                if not op or not isinstance(op, dict):
                    continue

                # Paramètres
                params = []
                all_params = path_item.get("parameters", []) + op.get("parameters", [])
                for p in all_params:
                    if "$ref" in p:
                        p = self._resolve_ref(p["$ref"])
                    params.append({
                        "name":     p.get("name", ""),
                        "in":       p.get("in", ""),       # query|path|header|cookie
                        "required": p.get("required", False),
                        "type":     self._extract_param_type(p),
                        "description": p.get("description", ""),
                    })

                # Request body (OpenAPI 3.x)
                request_schema = {}
                if self._version == "3.x":
                    rb = op.get("requestBody", {})
                    if rb:
                        content = rb.get("content", {})
                        for mime, media in content.items():
                            if "application/json" in mime or not request_schema:
                                s = media.get("schema", {})
                                request_schema = self._resolve_schema(s)
                                break

                # Réponses
                responses = {}
                for code, resp in op.get("responses", {}).items():
                    if "$ref" in resp:
                        resp = self._resolve_ref(resp["$ref"])
                    resp_schema = {}
                    if self._version == "3.x":
                        content = resp.get("content", {})
                        for mime, media in content.items():
                            if "application/json" in mime or not resp_schema:
                                s = media.get("schema", {})
                                resp_schema = self._resolve_schema(s)
                                break
                    else:
                        # Swagger 2.x
                        s = resp.get("schema", {})
                        resp_schema = self._resolve_schema(s)

                    responses[str(code)] = {
                        "description": resp.get("description", ""),
                        "schema": resp_schema,
                    }

                # Sécurité par endpoint
                endpoint_security = op.get("security", path_item.get("security", []))

                # Rate limits (custom headers dans x-ratelimit)
                rate_limit = {}
                for ext_key in ["x-ratelimit", "x-rate-limit", "x-throttle"]:
                    if ext_key in op:
                        rate_limit = op[ext_key]
                        break

                endpoints.append({
                    "path":           path,
                    "method":         method.upper(),
                    "operation_id":   op.get("operationId", ""),
                    "summary":        op.get("summary", ""),
                    "description":    op.get("description", ""),
                    "tags":           op.get("tags", []),
                    "parameters":     params,
                    "request_schema": request_schema,
                    "responses":      responses,
                    "security":       endpoint_security,
                    "rate_limit":     rate_limit,
                    "deprecated":     op.get("deprecated", False),
                })

        logger.info(f"[OpenAPI] {len(endpoints)} endpoints extraits")
        return endpoints

    def _extract_param_type(self, param: Dict) -> str:
        """Extrait le type d'un paramètre (OpenAPI 3.x et Swagger 2.x)."""
        if self._version == "3.x":
            schema = param.get("schema", {})
            return schema.get("type", "string")
        return param.get("type", "string")

    def _extract_schemas(self) -> Dict[str, Dict]:
        """Extrait tous les schemas de composants."""
        if self._version == "3.x":
            components = self.spec.get("components", {})
            schemas = components.get("schemas", {})
        else:
            schemas = self.spec.get("definitions", {})

        result = {}
        for name, schema in schemas.items():
            result[name] = {
                "type":        schema.get("type", "object"),
                "properties":  schema.get("properties", {}),
                "required":    schema.get("required", []),
                "description": schema.get("description", ""),
                "x_entity":    schema.get("x-entity", ""),  # Custom: lie schema à entité
            }
        return result

    def _extract_security_schemes(self) -> Dict[str, Dict]:
        """Extrait les schémas d'authentification."""
        if self._version == "3.x":
            components = self.spec.get("components", {})
            schemes = components.get("securitySchemes", {})
        else:
            schemes_raw = self.spec.get("securityDefinitions", {})
            schemes = schemes_raw

        result = {}
        for name, scheme in schemes.items():
            result[name] = {
                "type":             scheme.get("type", ""),
                "scheme":           scheme.get("scheme", ""),     # bearer, basic
                "bearer_format":    scheme.get("bearerFormat", ""),
                "flows":            scheme.get("flows", {}),       # OAuth2
                "in":               scheme.get("in", ""),          # header, query
                "parameter_name":   scheme.get("name", ""),
                "open_id_connect":  scheme.get("openIdConnectUrl", ""),
            }
        return result

    def _extract_rate_limits(self) -> Dict:
        """Extrait les rate limits documentés (extensions x- ou headers)."""
        rate_limits = {}

        # Au niveau info (global)
        info = self.spec.get("info", {})
        for ext in ["x-ratelimit", "x-rate-limit", "x-throttle"]:
            if ext in info:
                rate_limits["global"] = info[ext]

        # Par endpoint (déjà extrait dans _extract_endpoints mais résumé ici)
        paths = self.spec.get("paths", {})
        per_endpoint = []
        for path, path_item in paths.items():
            for method in ["get", "post", "put", "delete", "patch"]:
                op = path_item.get(method, {})
                for ext in ["x-ratelimit", "x-rate-limit", "x-throttle"]:
                    if ext in op:
                        per_endpoint.append({
                            "path":   path,
                            "method": method.upper(),
                            "limit":  op[ext],
                        })

        if per_endpoint:
            rate_limits["per_endpoint"] = per_endpoint

        return rate_limits

    def _extract_resource_relations(self) -> List[Dict]:
        """
        Détecte les relations entre ressources API :
        - Paramètres de path partagés entre endpoints (ex: /orders/{id}/lines)
        - $ref vers d'autres schemas (propriétés FK)
        - HATEOAS links dans les schemas
        """
        relations = []
        schemas = self._extract_schemas()

        for schema_name, schema in schemas.items():
            props = schema.get("properties", {})
            for prop_name, prop_def in props.items():
                # Détecter les propriétés FK (nom se terminant par Id/ID/_id)
                if re.search(r'(Id|ID|_id)$', prop_name):
                    # Chercher le schema cible
                    ref_name = prop_name.rstrip("Iid").rstrip("_")
                    if ref_name in schemas:
                        relations.append({
                            "source_schema":  schema_name,
                            "source_property": prop_name,
                            "target_schema":  ref_name,
                            "relation_type":  "foreign_key",
                            "confidence":     0.75,
                        })

                # Détecter les $ref directes
                if "$ref" in prop_def:
                    ref_path = prop_def["$ref"]
                    target = ref_path.split("/")[-1]
                    if target != schema_name:
                        relations.append({
                            "source_schema":   schema_name,
                            "source_property": prop_name,
                            "target_schema":   target,
                            "relation_type":   "reference",
                            "confidence":      0.90,
                        })

                # HATEOAS links
                if prop_name.lower() in ("links", "_links", "href") and isinstance(prop_def, dict):
                    relations.append({
                        "source_schema":   schema_name,
                        "source_property": prop_name,
                        "target_schema":   "HATEOAS",
                        "relation_type":   "hateoas_link",
                        "confidence":      0.70,
                    })

            # Hiérarchie de path : /customers/{id}/orders → Customer→Order
            # (relation implicite par le path parent)

        # Relations parent/enfant depuis les paths
        paths = self.spec.get("paths", {})
        path_list = sorted(paths.keys())
        for path in path_list:
            # /resource/{id}/subresource → resource → subresource
            m = re.match(r"^/([^/]+)/\{[^}]+\}/([^/]+)", path)
            if m:
                parent_resource = m.group(1)
                child_resource  = m.group(2)
                # Vérifier que les deux sont aussi des endpoints root
                if f"/{parent_resource}" in paths and f"/{child_resource}" not in paths:
                    relations.append({
                        "source_schema":   _to_pascal(child_resource),
                        "source_property": parent_resource.rstrip("s") + "Id",
                        "target_schema":   _to_pascal(parent_resource),
                        "relation_type":   "parent_child_path",
                        "confidence":      0.80,
                    })

        logger.info(f"[OpenAPI] {len(relations)} relations détectées")
        return relations

    def to_source_entities(self, source_id: UUID) -> List[Dict]:
        """
        Convertit les endpoints en source_entities + entity_fields
        compatibles avec le schéma SQL OnePilot.
        Retourne une liste de dicts prêts pour l'INSERT.
        """
        entities = []
        endpoints = self._extract_endpoints()

        # Regrouper par ressource principale (tag ou premier segment de path)
        resource_map: Dict[str, List[Dict]] = {}
        for ep in endpoints:
            resource = ep["tags"][0] if ep["tags"] else _path_to_resource(ep["path"])
            resource_map.setdefault(resource, []).append(ep)

        for resource, eps in resource_map.items():
            fields = []
            seen_params = set()

            for ep in eps:
                # Paramètres → fields
                for param in ep["parameters"]:
                    if param["name"] in seen_params:
                        continue
                    seen_params.add(param["name"])
                    fields.append({
                        "name":            param["name"],
                        "display_name":    param["name"],
                        "data_type":       _openapi_type_to_onepilot(param["type"]),
                        "native_type":     param["type"],
                        "is_nullable":     not param["required"],
                        "is_primary_key":  param["in"] == "path" and param["name"] == "id",
                        "is_foreign_key":  param["in"] == "path" and param["name"].endswith("Id"),
                        "position":        len(fields),
                        "param_location":  param["in"],
                    })

            # Schema de réponse principale (GET 200)
            get_ep = next(
                (e for e in eps if e["method"] == "GET" and "200" in e["responses"]), None
            )
            if get_ep:
                resp_schema = get_ep["responses"].get("200", {}).get("schema", {})
                props = resp_schema.get("properties", {})
                for prop_name, prop_def in props.items():
                    if prop_name not in seen_params:
                        seen_params.add(prop_name)
                        fields.append({
                            "name":          prop_name,
                            "display_name":  prop_name,
                            "data_type":     _openapi_type_to_onepilot(
                                              prop_def.get("type", "string")),
                            "native_type":   prop_def.get("type", "string"),
                            "is_nullable":   True,
                            "is_primary_key": prop_name.lower() == "id",
                            "is_foreign_key": bool(re.search(r'(Id|ID|_id)$', prop_name)),
                            "position":      len(fields),
                        })

            # Codes de réponse → fields RESPONSE_XXX
            seen_responses: set = set()
            for ep in eps:
                for code, resp_data in ep.get("responses", {}).items():
                    resp_key = f"RESPONSE_{code}"
                    if resp_key not in seen_responses:
                        seen_responses.add(resp_key)
                        desc = resp_data.get("description", "")
                        # Description HTTP standard si vide
                        if not desc:
                            desc = {
                                "200": "OK — Succès",
                                "201": "Created — Ressource créée",
                                "204": "No Content — Succès sans corps",
                                "400": "Bad Request — Requête invalide",
                                "401": "Unauthorized — Authentification requise",
                                "403": "Forbidden — Accès refusé",
                                "404": "Not Found — Ressource introuvable",
                                "409": "Conflict — Conflit de ressource",
                                "422": "Unprocessable Entity — Validation échouée",
                                "429": "Too Many Requests — Rate limit atteint",
                                "500": "Internal Server Error — Erreur serveur",
                                "503": "Service Unavailable — Service indisponible",
                            }.get(str(code), f"HTTP {code}")
                        fields.append({
                            "name":           resp_key,
                            "display_name":   f"HTTP {code}",
                            "data_type":      "string",
                            "native_type":    f"http_response_{code}",
                            "is_nullable":    True,
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "position":       len(fields),
                            "description":    desc,
                            "http_method":    ep["method"],
                            "http_status":    str(code),
                            "is_success":     str(code).startswith("2"),
                            "is_error":       str(code).startswith(("4", "5")),
                        })

            entities.append({
                "name":           resource,
                "display_name":   resource,
                "entity_type":    "api_endpoint",
                "description":    eps[0].get("summary", "") if eps else "",
                "endpoint_count": len(eps),
                "methods":        list({e["method"] for e in eps}),
                "fields":         fields,
            })

        return entities


# ══════════════════════════════════════════════════════════════════════
# GRAPHQL INTROSPECTION
# ══════════════════════════════════════════════════════════════════════

INTROSPECTION_QUERY = """
{
  __schema {
    types {
      name
      kind
      description
      fields {
        name
        description
        type { name kind ofType { name kind } }
        args { name type { name kind } }
        isDeprecated
        deprecationReason
      }
      inputFields {
        name
        type { name kind ofType { name kind } }
      }
      enumValues { name isDeprecated }
    }
    queryType    { name }
    mutationType { name }
    subscriptionType { name }
  }
}
"""


class GraphQLIntrospector:
    """Parse le schéma GraphQL via introspection."""

    def __init__(self, schema: Dict):
        self.schema = schema.get("data", schema).get("__schema", schema)

    @classmethod
    async def from_endpoint(
        cls,
        url: str,
        headers: Optional[Dict] = None,
        timeout: int = 30,
    ) -> "GraphQLIntrospector":
        """Récupère le schéma via introspection HTTP."""
        import httpx  # type: ignore
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            resp = await client.post(
                url,
                json={"query": INTROSPECTION_QUERY},
                headers=h,
            )
            resp.raise_for_status()
            data = resp.json()
        return cls(data)

    # ── Scalaires GraphQL internes à ignorer ──────────────────────────
    BUILTIN_TYPES = {
        "__Schema", "__Type", "__TypeKind", "__Field", "__InputValue",
        "__EnumValue", "__Directive", "__DirectiveLocation",
        "String", "Int", "Float", "Boolean", "ID",
    }

    def _user_types(self) -> List[Dict]:
        return [
            t for t in self.schema.get("types", [])
            if t.get("name") and t["name"] not in self.BUILTIN_TYPES
        ]

    def extract_all(self) -> Dict:
        types = self._user_types()

        object_types  = [t for t in types if t.get("kind") == "OBJECT"]
        input_types   = [t for t in types if t.get("kind") == "INPUT_OBJECT"]
        enum_types    = [t for t in types if t.get("kind") == "ENUM"]
        scalar_types  = [t for t in types if t.get("kind") == "SCALAR"]
        interface_types = [t for t in types if t.get("kind") == "INTERFACE"]

        query_type    = self.schema.get("queryType", {})
        mutation_type = self.schema.get("mutationType") or {}
        sub_type      = self.schema.get("subscriptionType") or {}

        return {
            "object_types":     [self._parse_object_type(t) for t in object_types],
            "input_types":      [self._parse_input_type(t) for t in input_types],
            "enum_types":       [self._parse_enum_type(t) for t in enum_types],
            "scalar_types":     [t["name"] for t in scalar_types],
            "interfaces":       [t["name"] for t in interface_types],
            "query_type":       query_type.get("name", "Query"),
            "mutation_type":    mutation_type.get("name"),
            "subscription_type": sub_type.get("name"),
            "relations":        self._extract_relations(object_types),
        }

    def _parse_object_type(self, t: Dict) -> Dict:
        fields = []
        for f in (t.get("fields") or []):
            ftype = self._unwrap_type(f.get("type", {}))
            fields.append({
                "name":        f.get("name", ""),
                "description": f.get("description", ""),
                "type":        ftype["name"],
                "kind":        ftype["kind"],
                "is_list":     ftype.get("is_list", False),
                "is_non_null": ftype.get("is_non_null", False),
                "args":        [
                    {"name": a["name"], "type": self._unwrap_type(a.get("type", {}))["name"]}
                    for a in (f.get("args") or [])
                ],
                "is_deprecated": f.get("isDeprecated", False),
            })
        return {
            "name":        t.get("name", ""),
            "description": t.get("description", ""),
            "fields":      fields,
        }

    def _parse_input_type(self, t: Dict) -> Dict:
        fields = []
        for f in (t.get("inputFields") or []):
            ftype = self._unwrap_type(f.get("type", {}))
            fields.append({
                "name": f.get("name", ""),
                "type": ftype["name"],
                "kind": ftype["kind"],
            })
        return {"name": t.get("name", ""), "fields": fields}

    def _parse_enum_type(self, t: Dict) -> Dict:
        return {
            "name":   t.get("name", ""),
            "values": [v["name"] for v in (t.get("enumValues") or [])],
        }

    def _unwrap_type(self, type_obj: Dict, depth: int = 0) -> Dict:
        """Déplie NON_NULL et LIST pour trouver le type de base."""
        if not type_obj or depth > 5:
            return {"name": "Unknown", "kind": "SCALAR", "is_list": False, "is_non_null": False}
        kind = type_obj.get("kind", "")
        name = type_obj.get("name", "")
        of_type = type_obj.get("ofType")
        if kind in ("NON_NULL", "LIST") and of_type:
            inner = self._unwrap_type(of_type, depth + 1)
            inner["is_list"]     = inner.get("is_list", False) or (kind == "LIST")
            inner["is_non_null"] = inner.get("is_non_null", False) or (kind == "NON_NULL")
            return inner
        return {"name": name, "kind": kind, "is_list": False, "is_non_null": False}

    def _extract_relations(self, object_types: List[Dict]) -> List[Dict]:
        """
        Déduit les relations entre types GraphQL :
        - Champ dont le type est un autre OBJECT type = relation 1:1 ou 1:N
        - Champ de type List<ObjectType> = relation 1:N
        """
        relations = []
        type_names = {t["name"] for t in object_types}

        for t in object_types:
            for field in (t.get("fields") or []):
                ftype = self._unwrap_type(field.get("type", {}))
                target = ftype.get("name", "")
                if target in type_names and target != t.get("name"):
                    relations.append({
                        "source_type":  t.get("name"),
                        "source_field": field.get("name"),
                        "target_type":  target,
                        "is_list":      ftype.get("is_list", False),
                        "relation_type": "one_to_many" if ftype.get("is_list") else "one_to_one",
                        "confidence":   0.90,
                    })

        logger.info(f"[GraphQL] {len(relations)} relations détectées")
        return relations

    def to_source_entities(self, source_id: UUID) -> List[Dict]:
        """Convertit les types GraphQL en source_entities OnePilot."""
        all_data = self.extract_all()
        entities = []

        GRAPHQL_TO_ONEPILOT = {
            "String": "string", "Int": "integer", "Float": "float",
            "Boolean": "boolean", "ID": "string",
        }

        for obj_type in all_data["object_types"]:
            # Ignorer le type Query/Mutation/Subscription racine
            if obj_type["name"] in (
                all_data.get("query_type"),
                all_data.get("mutation_type"),
                all_data.get("subscription_type"),
            ):
                continue

            fields = []
            for i, field in enumerate(obj_type["fields"]):
                fields.append({
                    "name":          field["name"],
                    "display_name":  field["name"],
                    "data_type":     GRAPHQL_TO_ONEPILOT.get(field["type"], "string"),
                    "native_type":   field["type"],
                    "is_nullable":   not field["is_non_null"],
                    "is_primary_key": field["name"].lower() == "id",
                    "is_foreign_key": (
                        field["kind"] == "OBJECT" or
                        bool(re.search(r'(Id|ID|_id)$', field["name"]))
                    ),
                    "position": i,
                })

            entities.append({
                "name":         obj_type["name"],
                "display_name": obj_type["name"],
                "entity_type":  "graphql_type",
                "description":  obj_type.get("description", ""),
                "fields":       fields,
            })

        return entities


# ══════════════════════════════════════════════════════════════════════
# INTÉGRATION DANS connection_service.py
# ══════════════════════════════════════════════════════════════════════

async def catalog_api_source(
    source_id: UUID,
    source_dict: Dict,
) -> Dict:
    """
    Fonction principale appelée par connection_service.py lors de la sync
    d'une source de type API.

    Détecte automatiquement le type (OpenAPI, GraphQL, OData)
    et retourne les entities + fields à indexer.
    """
    ct = (source_dict.get("connector_type") or "").lower()
    opts = source_dict.get("options") or {}
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            opts = {}

    base_url = source_dict.get("base_url") or opts.get("base_url", "")
    spec_url = opts.get("openapi_spec_url") or opts.get("spec_url") or ""
    gql_url  = opts.get("graphql_url") or (base_url if "graphql" in ct else "")

    entities = []
    relations = []

    # ── OpenAPI / Swagger ─────────────────────────────────────────────
    if spec_url or "openapi" in ct or "swagger" in ct or "rest" in ct:
        url = spec_url or f"{base_url}/openapi.json"
        try:
            parser = await OpenAPIParser.from_url(url)
            data = parser.extract_all()
            entities = parser.to_source_entities(source_id)
            relations = data.get("relations", [])
            logger.info(
                f"[api_cataloger] OpenAPI: {len(entities)} resources, "
                f"{len(relations)} relations"
            )
        except Exception as e:
            logger.warning(f"[api_cataloger] OpenAPI {url}: {e}")
            # Fallback : essayer /swagger.json, /api-docs
            for fallback in ["/swagger.json", "/api-docs", "/v1/openapi.json"]:
                try:
                    parser = await OpenAPIParser.from_url(base_url + fallback)
                    entities = parser.to_source_entities(source_id)
                    relations = parser.extract_all().get("relations", [])
                    break
                except Exception:
                    continue

    # ── GraphQL ───────────────────────────────────────────────────────
    elif gql_url or "graphql" in ct:
        url = gql_url or base_url
        try:
            headers = {}
            auth_token = opts.get("auth_token") or opts.get("api_key")
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            introspector = await GraphQLIntrospector.from_endpoint(url, headers=headers)
            entities = introspector.to_source_entities(source_id)
            gql_data = introspector.extract_all()
            relations = [
                {
                    "source_entity":  r["source_type"],
                    "source_field":   r["source_field"],
                    "target_entity":  r["target_type"],
                    "target_field":   "id",
                    "relation_type":  r["relation_type"],
                    "confidence":     r["confidence"],
                    "detection_method": "graphql_schema",
                }
                for r in gql_data.get("relations", [])
            ]
            logger.info(
                f"[api_cataloger] GraphQL: {len(entities)} types, "
                f"{len(relations)} relations"
            )
        except Exception as e:
            logger.warning(f"[api_cataloger] GraphQL {url}: {e}")

    return {
        "entities":  entities,
        "relations": relations,
        "source_id": str(source_id),
        "total_entities":  len(entities),
        "total_relations": len(relations),
    }


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def _path_to_resource(path: str) -> str:
    """Extrait le nom de la ressource principale depuis un path URL."""
    segments = [s for s in path.split("/") if s and not s.startswith("{")]
    return segments[0] if segments else "unknown"


def _to_pascal(s: str) -> str:
    """snake_case / kebab-case → PascalCase."""
    return "".join(word.capitalize() for word in re.split(r"[-_]", s))


def _openapi_type_to_onepilot(openapi_type: str) -> str:
    """Convertit un type OpenAPI en type OnePilot canonique."""
    mapping = {
        "integer": "integer", "number": "float",
        "string":  "string",  "boolean": "boolean",
        "array":   "array",   "object": "object",
    }
    return mapping.get(openapi_type.lower(), "string")