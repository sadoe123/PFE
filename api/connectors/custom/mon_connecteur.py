from core.base_connector import BaseConnector, ConnectorMetadata

class MonConnecteurTest(BaseConnector):
    def connect(self) -> bool:
        return True
    
    def test_connection(self) -> dict:
        return {"success": True, "latency_ms": 10}
    
    def get_metadata(self) -> ConnectorMetadata:
        return ConnectorMetadata(
            connector_id=self.config.get("id", "test"),
            connector_type="custom",
            entities=[]
        )
    
    def execute_query(self, query: str, params=None) -> list:
        return []