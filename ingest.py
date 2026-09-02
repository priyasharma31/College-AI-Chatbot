from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle


# --------------------------------
# 1. Load knowledge base
# --------------------------------

file_path = Path("data/college_info.txt")

text = file_path.read_text(encoding="utf-8")

print("Knowledge base loaded!")


# --------------------------------
# 2. Create chunks
# --------------------------------

chunks = []

# Split according to sections
sections = text.split("\n\n")

for section in sections:
    section = section.strip()

    if section:
        chunks.append(section)


print(f"Created {len(chunks)} chunks")


# --------------------------------
# 3. Create embeddings
# --------------------------------

print("Creating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embeddings created!")


# --------------------------------
# 4. Create FAISS index
# --------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# --------------------------------
# 5. Save FAISS index
# --------------------------------

faiss.write_index(index, "college.index")


# Save chunks
with open("chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)


print("FAISS index saved!")
print("Chunks saved!")

print("\nYour RAG knowledge base is ready!")