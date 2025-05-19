import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === Step 1: Load and Split PDF Documents ===
def load_all_pdfs(pdf_paths):
    all_documents = []
    for pdf in pdf_paths:
        print(f"Loading {pdf}...")
        loader = PyPDFLoader(pdf)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

# === Step 2: Chunk Text ===
def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# === Step 3: Create Embeddings ===
def create_embeddings():
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Step 4: Build and Save FAISS Vector Store ===
def build_and_save_vdb(chunks, output_dir="new_faiss_book_db"):
    print("Creating FAISS index...")
    embeddings = create_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    print(f"Saving vector database to {output_dir}...")
    db.save_local(output_dir)
    print("Vector database saved successfully.")
    return db

# === Main Execution ===
if __name__ == "__main__":
    # List of your PDF files
    pdf_files = [
        "new_books\Concrete Mathematics.pdf",
        "new_books\math1.pdf",
        "new_books\Mathematics_ Its Content, Methods and Meaning - PDF Room.pdf"  # Replace with your actual third file name
    ]

    # Check if all files exist
    for pdf in pdf_files:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"File '{pdf}' not found.")

    # Load all PDF content
    docs = load_all_pdfs(pdf_files)

    # Split into chunks
    chunks = chunk_text(docs)

    # Build and save FAISS vector store
    build_and_save_vdb(chunks)
