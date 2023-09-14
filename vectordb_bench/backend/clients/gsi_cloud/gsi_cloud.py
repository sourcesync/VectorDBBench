"""Wrapper around the Weaviate vector database over VectorDB"""

import logging
from typing import Iterable, Type
from contextlib import contextmanager

import swagger_client
from swagger_client.models import *
import numpy as np
import time, os

from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import GSIConfig, GSIIndexConfig


log = logging.getLogger(__name__)


class GSICloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        
        self.verbose = True
        self.dataset_id = None
        """Initialize wrapper around the weaviate vector database."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        self._scalar_field = "key"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        config = swagger_client.Configuration()
        config.verify_ssl = False

        config.host = f"http://{db_config.host}:{db_config.port}/v1.0"
        api_config = swagger_client.ApiClient(config)
        api_config.default_headers["allocationToken"] = db_config.allocation_id    
        self.allocation_id = db_config.allocation_id

        self.datasets_apis = swagger_client.DatasetsApi(api_config)
        self.search_apis = swagger_client.SearchApi(api_config)
        self.utilities_apis = swagger_client.UtilitiesApi(api_config)

        # self._create_collection(client)

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return GSIConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return GSIIndexConfig

    @contextmanager
    def init(self) -> None:
        # TODO: wtf...
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        from weaviate import Client
        self.client = Client(**self.db_config)
        yield
        self.client = None
        del(self.client)

    def ready_to_load(self):
        """Should call insert first, do nothing"""
        pass

    def optimize(self):
        assert self.client.schema.exists(self.collection_name)
        self.client.schema.update_config(self.collection_name, {"vectorIndexConfig": self.case_config.search_param() } )

    def _create_collection(self, client):
        if not client.schema.exists(self.collection_name):
            log.info(f"Create collection: {self.collection_name}")
            class_obj = {
                "class": self.collection_name,
                "vectorizer": "none",
                "properties": [
                    {
                        "dataType": ["int"],
                        "name": self._scalar_field,
                    },
                ]
            }
            class_obj["vectorIndexConfig"] = self.case_config.index_param()
            try:
                client.schema.create_class(class_obj)
            except WeaviateBaseError as e:
                log.warning(f"Failed to create collection: {self.collection_name} error: {str(e)}")
                raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        
        if self.verbose:
            print("GSI Import Dataset: load vectors to npy file...")

        dataset_file_path = self.db_config['dataset_file_path']
        if os.path.exists(dataset_file_path):
            os.remove(dataset_file_path)
        tmp = np.array(embeddings)
        np.save(dataset_file_path, tmp)

        if self.verbose:
            print("GSI Import Dataset: import dataset to FVS...")

        # import dataset FVS
        resp = self.datasets_apis.controllers_dataset_controller_create_dataset(
            ImportDatasetRequest(records=dataset_file_path, search_type=self.db_config['search_type'],
                                 train_ind=True, nbits=self.db_config['nbits']),
            allocation_token=self.allocation_id
        )
        self.dataset_id = resp.dataset_id

        # train status
        resp = self.datasets_apis.controllers_dataset_controller_get_dataset_status(
            self.dataset_id, self.allocation_id
        )
        train_status = resp.dataset_status
        while train_status != "completed":
            if self.verbose:
                print(f"GSI Train Status: currently {train_status}, waiting for \"completed\"")
            time.sleep(3)

            train_status = self.datasets_apis.controllers_dataset_controller_get_dataset_status(
                self.dataset_id, self.allocation_id
            ).dataset_status
        
        # load dataset
        resp = self.datasets_apis.controllers_dataset_controller_load_dataset(
            LoadDatasetRequest(allocation_id=self.allocation_id, dataset_id=self.dataset_id),
            allocation_token=self.allocation_id
        )
        if self.verbose:
            print("GSI Load Dataset: response", resp)
        
        # focus dataset
        resp = self.datasets_apis.controllers_dataset_controller_focus_dataset(
            FocusDatasetRequest(allocation_id=self.allocation_id, dataset_id=self.dataset_id)
        )
        if self.verbose:
            print("GSI Focus Dataset")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results with distance.
        Should call self.init() first.
        """

        if self.verbose:
            print("GSI Search: importing query vector(s) to npy file")
        queries_file_path = self.db_config['query_file_path']
        if os.path.exists(queries_file_path):
            os.remove(queries_file_path)
        tmp = np.reshape(np.array(query), (1, len(query)))
        np.save(queries_file_path, tmp)

        if self.verbose:
            print("GSI Search: import queries")
        self.utilities_apis.controllers_utilities_controller_import_queries(
            ImportQueriesRequest(queries_file_path=queries_file_path),
            allocation_token=self.allocation_id
        )

        if self.verbose:
            print("GSI Search: time for search hells yeah, k =", k)
        resp = self.search_apis.controllers_search_controller_search(
            SearchRequest(allocation_id=self.allocation_id, dataset_id=self.dataset_id, 
                          queries_file_path=queries_file_path, topk=k),
            allocation_token=self.allocation_id
        )

        return resp.indices

