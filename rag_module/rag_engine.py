import os
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

class ACSMRagEngine:
    def __init__(self):
        model_name = "intfloat/multilingual-e5-small"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
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
            return []