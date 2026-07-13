"""Inspect the raw payload structure vs what LangChain returns."""
from dotenv import load_dotenv
load_dotenv('.env')
import os

url = os.environ.get('QDRANT_URL')
key = os.environ.get('QDRANT_API_KEY')
col = os.environ.get('QDRANT_COLLECTION_NAME', 'device_manuals')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

print('Loading embeddings...')
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

client = QdrantClient(url=url, api_key=key)

# First inspect 1 raw Qdrant point to see exact payload structure
print('\n--- RAW QDRANT PAYLOAD (first point) ---')
pts = client.scroll(col, limit=1, with_payload=True, with_vectors=False)[0]
if pts:
    p = pts[0]
    print('All payload keys:', list(p.payload.keys()))
    print('Full payload:', p.payload)

# Now check what LangChain builds as Document
print('\n--- LangChain Document (from similarity_search) ---')
vector_store = QdrantVectorStore(client=client, collection_name=col, embedding=embeddings)
results = vector_store.similarity_search_with_score('refrigerator cooling problem', k=1)
if results:
    doc, score = results[0]
    print('page_content[:100]:', doc.page_content[:100])
    print('metadata:', doc.metadata)
    print('All metadata keys:', list(doc.metadata.keys()))
