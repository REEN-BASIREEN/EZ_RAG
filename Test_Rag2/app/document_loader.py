import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from config import DOCS_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_document():
    docs = []
    
    for fname in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, fname)
        # อ่านไฟล์ .txt 
        if fname.endswith('.txt'):
            loader = TextLoader(path, encoding='utf-8')
            docs.extend(loader.load())
            print(f"โหลดไฟล์ .txt: {fname}")
        # PDF loading can be added here if needed
        elif fname.endswith('.pdf'):
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
            print(f"โหลดไฟล์ .pdf: {fname}")
                # Placeholder for PDF loading logic

    # โหลดเอกสารจากไฟล์ข้อความ
    loader = TextLoader(os.path.join(DOCS_PATH, "sample_th.txt"), encoding="utf-8")
    documents = loader.load()
    print(f"โหลดเอกสารทั้งหมด: {len(documents)} ไฟล์")
    
    # แบ่งเอกสารเป็นชิ้นๆ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"แบ่งเอกสารเป็น {len(chunks)} ชิ้น")
    
    
    return chunks
