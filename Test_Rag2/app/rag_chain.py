from langchain_ollama import ChatOllama
from langchain import PromptTemplate, LLMChain
from config import OLLAMA_LLM_MODEL, TOP_K
import os

def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    # Try preferred model; allow optional override via OLLAMA_LLM_MODEL env var
    model_name = os.getenv("OLLAMA_LLM_MODEL", OLLAMA_LLM_MODEL)
    try:
        llm = ChatOllama(model=model_name, temperature=0.0)
    except Exception:
        # Fallback to a common alias if fully-qualified tag not found locally
        alt = "llama3.1-typhoon2-8b-instruct"
        llm = ChatOllama(model=alt, temperature=0.0)
    
    prompt_template = """
    คุณเป็นผู้ช่วยที่มีความรู้และเชี่ยวชาญในการให้ข้อมูลที่ถูกต้องและเป็นประโยชน์
    โดยใช้ข้อมูลจากเอกสารที่ให้มาในการตอบคำถามของผู้ใช้
    โดยจะต้อง:
    1. อ่านและทำความเข้าใจเอกสารที่ให้มาอย่างละเอียด
    2. ตอบคำถามของผู้ใช้โดยอิงจากเอกสารเท่านั้น
    3. หากเอกสารไม่เกี่ยวข้องหรือไม่เพียงพอ ให้ตอบว่า "ขออภัย ฉันไม่มีข้อมูลที่เกี่ยวข้องกับคำถามนี้"
    4. ใช้ภาษาที่สุภาพและเป็นมิตรในการตอบคำถาม
    {context}
    
    Question: {question}
    ตอบคำถามผู้ใช้งานเป็นภาษาไทยอย่างชัดเจน กระชับ และตรงประเด็นตามเอกสารที่ให้มา พร้อมทั้งอ้างอิงจากไฟล์เอกสาร {เช่น (doc1.txt) }
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = LLMChain(llm=llm, prompt=PROMPT)
    
    return retriever, chain

def rag_response(chain, retriever, question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([f"{doc.page_content} ({doc.metadata.get('source', 'doc')})" for doc in docs])
    
    response = chain.run(context=context, question=question)
    return response