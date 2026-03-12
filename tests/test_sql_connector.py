from connectors.sql_connector import SQLConnector


def test_sql_connector_init():
    cfg = {
        "type": "postgres",
        "host": "localhost",
        "database": "test"
    }

    c = SQLConnector(cfg)

    assert c.config["host"] == "localhost"