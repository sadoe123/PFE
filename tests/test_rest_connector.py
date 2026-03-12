from connectors.rest_connector import infer_type, json_to_entity

def test_rest_infer_more_types():
    assert infer_type(100) == "integer"
    assert infer_type(12.5) == "float"
    assert infer_type(False) == "boolean"


def test_json_to_entity_list():
    data = [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"}
    ]

    e = json_to_entity("users", data)

    assert e.name == "users"
    assert len(e.fields) == 2