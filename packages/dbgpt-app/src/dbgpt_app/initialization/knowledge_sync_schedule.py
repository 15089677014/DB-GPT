import asyncio
import io
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, Any
import pandas as pd
import schedule
from sqlalchemy import create_engine
from dbgpt.component import BaseComponent, SystemApp
from dbgpt_client import Client
from dbgpt_client.datasource import get_datasource
from dbgpt_client.schema import DocumentModel, SyncModel
from dbgpt_client.knowledge import sync_document
from dbgpt.configs.model_config import ROOT_PATH as DBGPT_ROOT_PATH, KNOWLEDGE_UPLOAD_ROOT_PATH
from dbgpt.util.configure import ConfigurationManager
from dbgpt_ext.rag.chunk_manager import SplitterType
from dbgpt_ext.rag import ChunkParameters


class SyncConfigManager:
    """Manage synchronization configuration"""
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from file"""
        config_file = os.getenv("DBGPT_CONFIG_FILE", './configs/dbgpt-proxy-deepseek.toml')
        try:
            self.cfg = ConfigurationManager.from_file(
                file_path=os.path.join(DBGPT_ROOT_PATH, config_file)
            )
            print("Configuration loaded successfully")
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            raise

    def get_sync_configs(self) -> Dict[str, Any]:
        """Get synchronization configurations"""
        return self.cfg.config["agent"]["system_prompt"]["synchronous"]


class KnowledgeSyncSchedule():
    name = "dbgpt_knowledge_sync_schedule"

    def __init__(
        self,
        # system_app: SystemApp
    ):
        # super().__init__(system_app)
        # self.system_app = system_app

        self.client = Client(api_key=os.getenv("DBGPT_API_KEY", "dbgpt"))
        self.config_manager = SyncConfigManager()
    # def init_app(self, system_app: SystemApp):
    #     self.system_app = system_app

    async def create_temp_file(self, df: pd.DataFrame) -> str:
        """Create temporary file with context management"""
        try:
            os.makedirs(os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, "synchronous"), exist_ok=True)
            with tempfile.NamedTemporaryFile(
                    mode='w+',
                    dir=os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, "synchronous"),
                    delete=False,
                    suffix='.csv'
            ) as tmp_file:
                df.to_csv(tmp_file.name)
                print(f"Created temporary file: {tmp_file.name}")
                return tmp_file.name
        except Exception as e:
            print(f"Failed to create temporary file: {e}")
            raise

    async def get_db_engine(self, sync_info: Dict[str, Any]):
        """Get database engine based on configuration"""
        try:
            if sync_info["is_customize"] == 1:
                return create_engine(
                    f"mysql+pymysql://{sync_info['db_user']}:{sync_info['db_pwd']}"
                    f"@{sync_info['db_host']}:{sync_info['db_port']}/{sync_info['db_name']}"
                    f"?{sync_info.get('ext_config', '')}"
                )
            else:
                db_info = await self.client.get("/datasources/" + sync_info["datasource_id"])
                db_info = db_info.json()["data"]
                return create_engine(
                    f"mysql+pymysql://{db_info['params']['user']}:{db_info['params']['password']}"
                    f"@{db_info['params']['host']}:{db_info['params']['port']}"
                    f"/{db_info['params']['database']}"
                )
        except Exception as e:
            print(f"Failed to create database engine: {e}")
            raise

    async def auto_synchronous(self, knowledge_name, sync_info):
        try:
            if sync_info["data_type"] in ("mysql", "starrocks"):
                engine = await self.get_db_engine(sync_info)
                table_schema = pd.read_sql(f"SHOW FULL COLUMNS FROM {sync_info['table_name']};", con=engine)[
                    ["Field", "Comment"]]
                table_schema = table_schema.set_index('Field')['Comment'].to_dict()
                df = pd.read_sql(sync_info["sql"], con=engine)
                df.rename(table_schema, axis=1, inplace=True)
            elif sync_info["data_type"] == "csv":
                csv_path = os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, 'synchronous', str(sync_info["doc_name"]))
                df = pd.read_csv(csv_path)
            else:
                print(f"Unsupported data type: {sync_info['data_type']}")

            # 将DataFrame转换为CSV格式的字节流
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            # 准备文件上传
            files = {
                'file': (f"{knowledge_name}.csv", csv_buffer.getvalue().encode('utf-8'), 'text/csv')
            }

            # 准备其他数据
            data = {
                'knowledge_name': knowledge_name,
                'doc_name': sync_info['doc_name'],
                'data_type': sync_info['data_type']
            }

            # 使用post_file方法上传
            res = await self.client.post_file(
                "/knowledge/documents/auto_synchronous",
                data=data,
                files=files
            )
            res = res.json()["data"]
            return res
        except Exception as e:
            print(e)

    async def run_sync(self):
        """Main synchronization entry point"""
        self.client = Client(api_key=os.getenv("DBGPT_API_KEY", "dbgpt"))
        sync_configs = self.config_manager.get_sync_configs()
        for knowledge_name, sync_info in sync_configs.items():
            try:
                await self.auto_synchronous(knowledge_name, sync_info)
            except Exception as e:
                print(f"Synchronization failed for {knowledge_name}: {e}")
                continue
        await self.client.aclose()
        print("Client connection closed")

    def execute_sync_task(self):
        """执行同步任务的同步方法"""
        try:
            asyncio.run(self.run_sync())
        except Exception as e:
            print(f"同步任务执行失败: {str(e)}")

    def after_start(self):
        """在系统启动后注册定时任务"""
        schedule.every().day.at("00:00").do(
            lambda: self.execute_sync_task()
        )

if __name__ == "__main__":
    client = KnowledgeSyncSchedule()
    client.execute_sync_task()