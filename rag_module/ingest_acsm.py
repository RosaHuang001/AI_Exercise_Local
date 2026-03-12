import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# 建議安裝 pip install langchain-chroma
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 1. 讀取環境變數
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

def build_acsm_vector_db():
    # 請確認您的 PDF 檔名與路徑
    pdf_path = os.path.join(BASE_DIR, "data", "ACSM心臟衰竭病患運動指引處方.pdf")
    
    if not os.path.exists(pdf_path):
        return
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 3. 調整切片大小：放大 chunk_size 確保標題與表格數據鎖在一起 [cite: 17, 31]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    docs = text_splitter.split_documents(pages)

    # 4. 優化 Metadata 分類
    for doc in docs:
        content = doc.page_content
        #標註 Safety 與 FITT 等分類，讓RAG在搜尋時可以根據情況「挑重點看」
        if any(key in content for key in ["FITT", "頻率", "強度", "1-RM", "HRR", "RPE"]):
            doc.metadata["category"] = "FITT"
        elif any(key in content for key in ["特殊考量", "LVAD", "安全", "禁忌", "MAP", "疲勞"]):
            doc.metadata["category"] = "Safety"
        else:
            doc.metadata["category"] = "General"

    # 5. 執行向量化並儲存
    try:
        #針對多語言優化的 E5 模型，它能理解「心衰竭」與「Cardiac Failure」在數學空間中是同一個意思
        model_name = "intfloat/multilingual-e5-small" 
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        persist_dir = os.path.join(BASE_DIR, "vector_db")
        
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="acsm_hf_guidelines"
        )
    except Exception as e:
        return

if __name__ == "__main__":
    build_acsm_vector_db()