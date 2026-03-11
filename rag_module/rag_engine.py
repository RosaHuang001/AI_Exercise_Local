import os
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

class ACSMRagEngine:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ 警告：找不到 OPENAI_API_KEY")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        persist_dir = os.path.join(CURRENT_DIR, "vector_db")
        
        self.vector_db = Chroma(
            persist_directory=persist_dir, 
            embedding_function=self.embeddings,
            collection_name="acsm_hf_guidelines"
        )

    def retrieve_rules(self, query_text: str, k: int = 5):
        """
        將 k 預設提高至 5，確保大型表格資訊能完整呈現 [cite: 17, 84]
        """
        try:
            # 使用相似度搜尋
            results = self.vector_db.similarity_search(query_text, k=k)
            
            formatted_rules = []
            for doc in results:
                formatted_rules.append({
                    "rule": doc.page_content,
                    "topic": doc.metadata.get("category", "General"),
                    "source": "ACSM HF Guideline"
                })
            return formatted_rules
        except Exception as e:
            print(f"❌ 檢索錯誤: {e}")
            return []