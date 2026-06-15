import fitz
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer(
    "intfloat/e5-small-v2"
)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    return EMBED_MODEL.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()


def embed_query(query):
    return EMBED_MODEL.encode(
        query,
        normalize_embeddings=True
    ).tolist()