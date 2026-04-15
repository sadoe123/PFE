"""
OnePilot – GraphQL Relation Saver + HATEOAS §2.2.2 completion
Sauvegarde les relations GraphQL nested types + HATEOAS links
dans entity_relations
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


class GraphQLRelationSaver:
    """
    Sauvegarde les relations GraphQL (nested types) dans entity_relations.
    Les relations sont déjà extraites par GraphQLIntrospector._extract_relations()
    mais n'étaient jamais persistées dans la DB.
    """

    def __init__(self, pg_pool: asyncpg.Pool):
        self.pg = pg_pool

    async def save_graphql_relations(
        self, source_id: UUID, relations: List[Dict]
    ) -> int:
        """
        Persiste les relations GraphQL dans entity_relations.
        relations = [
            {
                "source_type": "Order",
                "source_field": "customer",
                "target_type": "Customer",
                "is_list": False,
                "relation_type": "one_to_one",
                "confidence": 0.90
            }
        ]
        """
        count = 0
        for rel in relations:
            source_entity = rel.get("source_type", "")
            target_entity = rel.get("target_type", "")
            source_field  = rel.get("source_field", "")
            rel_type      = rel.get("relation_type", "one_to_one")
            confidence    = rel.get("confidence", 0.90)

            if not source_entity or not target_entity:
                continue

            # Vérifie si la relation existe déjà
            exists = await self.pg.fetchval("""
                SELECT 1 FROM entity_relations
                WHERE source_id     = $1
                  AND source_entity = $2
                  AND source_field  = $3
                  AND target_entity = $4
                LIMIT 1
            """, source_id, source_entity, source_field, target_entity)

            if exists:
                continue

            await self.pg.execute("""
                INSERT INTO entity_relations
                    (source_id, source_entity, source_field,
                     target_entity, target_field,
                     relation_type, confidence, detection_method,
                     is_confirmed)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,FALSE)
            """,
                source_id,
                source_entity,
                source_field,
                target_entity,
                "id",
                rel_type,
                confidence,
                "graphql_nested_type",
            )
            count += 1

        logger.info(
            f"[GraphQLRelationSaver] {count} relations GraphQL "
            f"sauvegardées pour source {source_id}"
        )
        return count


class HATEOASLinkExtractor:
    """
    Détecte et sauvegarde les relations HATEOAS depuis les réponses API.
    HATEOAS = Hypermedia As The Engine Of Application State
    Ex: { "_links": { "customer": { "href": "/customers/{id}" } } }
    """

    def __init__(self, pg_pool: asyncpg.Pool):
        self.pg = pg_pool

    def extract_from_openapi(self, spec: Dict) -> List[Dict]:
        """
        Extrait les relations HATEOAS depuis une spec OpenAPI.
        Cherche les patterns :
        - $ref entre schémas
        - x-links extensions
        - Properties de type URI avec pattern /resource/{id}
        """
        relations = []
        schemas = (
            spec.get("components", {}).get("schemas", {})  # OpenAPI 3.x
            or spec.get("definitions", {})                  # Swagger 2.x
        )

        for schema_name, schema_def in schemas.items():
            props = schema_def.get("properties", {})

            for prop_name, prop_def in props.items():
                # Pattern 1 : $ref vers un autre schéma
                ref = prop_def.get("$ref", "")
                if ref:
                    target = ref.split("/")[-1]
                    if target != schema_name:
                        relations.append({
                            "source_type":   schema_name,
                            "source_field":  prop_name,
                            "target_type":   target,
                            "relation_type": "many_to_one",
                            "confidence":    0.85,
                            "method":        "hateoas_ref",
                        })

                # Pattern 2 : x-links (HATEOAS extension)
                x_links = prop_def.get("x-links", {})
                for link_name, link_def in x_links.items():
                    href = link_def.get("href", "")
                    target = self._extract_resource_from_href(href)
                    if target:
                        relations.append({
                            "source_type":   schema_name,
                            "source_field":  prop_name,
                            "target_type":   target,
                            "relation_type": "many_to_one",
                            "confidence":    0.80,
                            "method":        "hateoas_xlinks",
                        })

                # Pattern 3 : format URI avec pattern /resource/{id}
                fmt = prop_def.get("format", "")
                if fmt == "uri" and prop_name.endswith(("_url", "_uri", "_href", "Url", "Uri")):
                    # Déduit la ressource depuis le nom du champ
                    target = re.sub(r'(Url|Uri|_url|_uri|_href)$', '', prop_name)
                    if target and target != schema_name:
                        relations.append({
                            "source_type":   schema_name,
                            "source_field":  prop_name,
                            "target_type":   target.title(),
                            "relation_type": "many_to_one",
                            "confidence":    0.65,
                            "method":        "hateoas_uri_field",
                        })

                # Pattern 4 : _links objet HAL standard
                if prop_name == "_links":
                    hal_props = prop_def.get("properties", {})
                    for link_name, link_def in hal_props.items():
                        if link_name not in ("self", "first", "last", "next", "prev"):
                            relations.append({
                                "source_type":   schema_name,
                                "source_field":  f"_links.{link_name}",
                                "target_type":   link_name.title(),
                                "relation_type": "many_to_one",
                                "confidence":    0.75,
                                "method":        "hateoas_hal",
                            })

        logger.info(f"[HATEOAS] {len(relations)} liens HATEOAS extraits")
        return relations

    def _extract_resource_from_href(self, href: str) -> Optional[str]:
        """Extrait le nom de la ressource depuis un href HATEOAS."""
        if not href:
            return None
        # Ex: /api/customers/{id} → customers
        m = re.search(r'/([a-zA-Z][a-zA-Z0-9_-]+)(?:/\{[^}]+\})?/?$', href)
        if m:
            resource = m.group(1)
            # Singularise basiquement
            if resource.endswith('s'):
                resource = resource[:-1]
            return resource.title()
        return None

    async def save_hateoas_relations(
        self, source_id: UUID, relations: List[Dict]
    ) -> int:
        """Persiste les relations HATEOAS dans entity_relations."""
        count = 0
        for rel in relations:
            source_entity = rel.get("source_type", "")
            target_entity = rel.get("target_type", "")
            source_field  = rel.get("source_field", "")
            rel_type      = rel.get("relation_type", "many_to_one")
            confidence    = rel.get("confidence", 0.75)
            method        = rel.get("method", "hateoas")

            if not source_entity or not target_entity:
                continue

            exists = await self.pg.fetchval("""
                SELECT 1 FROM entity_relations
                WHERE source_id     = $1
                  AND source_entity = $2
                  AND source_field  = $3
                  AND target_entity = $4
                LIMIT 1
            """, source_id, source_entity, source_field, target_entity)

            if exists:
                continue

            await self.pg.execute("""
                INSERT INTO entity_relations
                    (source_id, source_entity, source_field,
                     target_entity, target_field,
                     relation_type, confidence, detection_method,
                     is_confirmed)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,FALSE)
            """,
                source_id, source_entity, source_field,
                target_entity, "id",
                rel_type, confidence, method,
            )
            count += 1

        logger.info(
            f"[HATEOAS] {count} liens sauvegardés pour source {source_id}"
        )
        return count