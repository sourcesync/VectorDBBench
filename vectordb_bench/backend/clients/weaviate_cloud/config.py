import pydantic
from pydantic import BaseModel, SecretStr
import weaviate

from ..api import DBConfig, DBCaseConfig, MetricType

import logging

class MyClass: 
    pass

class WeaviateConfig(DBConfig):
    v: MyClass
    url: SecretStr
    api_key: SecretStr
    
    class Config:
            arbitrary_types_allowed = True

    log = logging.getLogger("__main__")
    log.warning("")
    log.warning("")
    log.warning("")
    log.warning("GET WEAVIATE INDEX CONFIG")
    log.warning("")

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "auth_client_secret": weaviate.AuthApiKey(api_key=self.api_key.get_secret_value()),
        }


class WeaviateIndexConfig(BaseModel, DBCaseConfig):
    v: MyClass
    metric_type: MetricType | None = None
    ef: int | None = -1
    efConstruction: int | None = None
    maxConnections: int | None = None

    class Config:
            arbitrary_types_allowed = True
    log = logging.getLogger("__main__")
    log.warning("")
    log.warning("")
    log.warning("")
    log.warning("GET WEAVIATE INDEX CONFIG")
    log.warning("")

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2-squared"
        elif self.metric_type == MetricType.IP:
            return "dot"
        return "cosine"

    def index_param(self) -> dict:
        if self.maxConnections is not None and self.efConstruction is not None:
            params = {
                "distance": self.parse_metric(),
                "maxConnections": self.maxConnections,
                "efConstruction": self.efConstruction,
            }
        else:
            params = {"distance": self.parse_metric()}
        return params

    def search_param(self) -> dict:
        return {
            "ef": self.ef,
        }
