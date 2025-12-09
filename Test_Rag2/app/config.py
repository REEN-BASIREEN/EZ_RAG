OLLAMA_EMBED_MODEL = "bge-m3"
OLLAMA_LLM_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct:latest"
CHROMA_PATH = "./chroma_db"
DOCS_PATH = "./docs_th"


CHUNK_SIZE = 400 # ปรับหัวข้อความที่ตัดเป็นชิ้น
CHUNK_OVERLAP = 200 # ปรับหัวข้อความที่ตัดเป็นชิ้น
TOP_K = 10 # จำนวนเอกสารที่ดึงมาใช้ในการตอบคำถาม