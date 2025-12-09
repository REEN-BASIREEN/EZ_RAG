from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import OLLAMA_EMBED_MODEL, CHROMA_PATH


def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL) # ใช้โมเดลจาก config
    texts = [chunk.page_content for chunk in chunks] # ดึงข้อความจาก chunks
    metadata = [chunk.metadata for chunk in chunks] # ดึง metadata จาก chunks
    # สร้างหรือโหลด vector store ด้วย Chroma
    vector_store = Chroma.from_documents(
        texts,
        metadata = metadata,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return vector_store