import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/college_info.txt"
INDEX_FILE = "college.index"
CHUNKS_FILE = "chunks.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Load document
with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read()

print("Knowledge base loaded!")


# -----------------------------------
# Better chunking
# -----------------------------------

sections = text.split("==================================================")

chunks = []

for section in sections:

    section = section.strip()

    if not section:
        continue

    # Keep reasonably sized sections together
    words = section.split()

    # If section is small, keep it as one chunk
    if len(words) <= 250:
        chunks.append(section)

    else:
        # Split large sections into overlapping chunks
        chunk_size = 180
        overlap = 40

        start = 0

        while start < len(words):

            chunk_words = words[start:start + chunk_size]

            chunk = " ".join(chunk_words)

            chunks.append(chunk)

            start += chunk_size - overlap


print(f"Created {len(chunks)} chunks")


# -----------------------------------
# Create embeddings
# -----------------------------------

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Creating embeddings...")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# -----------------------------------
# Create FAISS index
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings.astype("float32"))


# -----------------------------------
# Save files
# -----------------------------------

faiss.write_index(index, INDEX_FILE)

with open(CHUNKS_FILE, "wb") as file:
    pickle.dump(chunks, file)


print()
print("================================")
print("Ingestion completed successfully!")
print("================================")
print(f"Saved index  : {INDEX_FILE}")
print(f"Saved chunks : {CHUNKS_FILE}")