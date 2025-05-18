import os
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# === STEP 1: Load text from TXT file ===
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

# === STEP 2: Hybrid chunking ===
def hybrid_chunk_text(text, big_chunk_size=200, overlap=40, min_paragraph_len=100):
    big_chunks = []
    for i in range(0, len(text), big_chunk_size - overlap):
        big_chunks.append(text[i:i + big_chunk_size])

    final_chunks = []
    for chunk in big_chunks:
        paragraphs = [p.strip() for p in chunk.split('\n\n') if len(p.strip()) >= min_paragraph_len]
        if not paragraphs:
            paragraphs = [p.strip() for p in chunk.split('\n') if len(p.strip()) >= min_paragraph_len]
        final_chunks.extend(paragraphs)

    return [Document(page_content=p) for p in final_chunks]

# === STEP 3: Process TXT file and return chunks ===
def process_txt(txt_path):
    text = extract_text_from_txt(txt_path)
    return hybrid_chunk_text(text)

# === STEP 4: Build a new VDB ===
def build_new_vdb_from_txt(txt_folder, save_path="new_arabic_vdb"):
    print(f"📁 Creating new VDB at: {save_path}")
    all_chunks = []
    
    for file in os.listdir(txt_folder):
        if file.endswith(".txt"):
            print(f"📄 Processing {file}")
            chunks = process_txt(os.path.join(txt_folder, file))
            for chunk in chunks:
                chunk.metadata["source"] = file
            all_chunks.extend(chunks)

    print(f"📦 Total chunks: {len(all_chunks)}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"✅ VDB saved to: {save_path}")

# === RUN ===
if __name__ == "__main__":
    build_new_vdb_from_txt("arabic_books")  # Folder containing your .txt files
