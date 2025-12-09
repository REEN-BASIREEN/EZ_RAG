from document_loader import load_and_split_document
from embedding_store import create_vector_store
from rag_chain import build_chain, rag_chain_with_meta


def main():
    print("Loading document from doc_th...")
    chunks = load_and_split_document()
    print(f"Loaded {len(chunks)} chunks")
    
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    print("Vector store and chain created.")

    print("setting RAG_Chain...")
    retriever, chain = build_chain(vector_store)
    
    print("Ready to answer questions.")
    print("If you want to exit the program can do Ctrl+C")
    
    while True:
        question = input("\nEnter your question: ")
        result = rag_chain_with_meta(chain, retriever, question)
        print(f"Answer: {result['answer']}")
        
    if __name__ == "__main__":
        main()