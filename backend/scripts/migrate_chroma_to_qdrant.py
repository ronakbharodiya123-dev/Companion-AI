"""
migrate_chroma_to_qdrant.py
───────────────────────────
One-time migration script: copies every document that currently lives in the
local ChromaDB collection into your Qdrant Cloud cluster.

Usage (from the backend/ directory, with the venv activated):

    python scripts/migrate_chroma_to_qdrant.py

The script expects the environment variables in .env to be set correctly
(QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, EMBEDDING_DIMENSION).
It reads ChromaDB from ./chroma_data (the default persist directory).
"""

import os
import sys
from pathlib import Path

# Make sure we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import chromadb
from chromadb.config import Settings as ChromaSettings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parents[1] / "chroma_data")
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION_NAME", "device_manuals")

QDRANT_URL         = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION_NAME", "device_manuals")
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIMENSION", "384"))
BATCH_SIZE         = 100
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=== ChromaDB → Qdrant Cloud Migration ===\n")

    # ── Source: ChromaDB ──────────────────────────────────────────────────────
    print(f"[1/4] Connecting to ChromaDB at {CHROMA_PERSIST_DIR} …")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_col = chroma_client.get_collection(CHROMA_COLLECTION)
        total = chroma_col.count()
        print(f"      Found {total} vectors in '{CHROMA_COLLECTION}'")
    except Exception as e:
        print(f"ERROR connecting to ChromaDB: {e}")
        sys.exit(1)

    if total == 0:
        print("      Nothing to migrate. Exiting.")
        return

    # ── Destination: Qdrant Cloud ─────────────────────────────────────────────
    print(f"\n[2/4] Connecting to Qdrant Cloud at {QDRANT_URL} …")
    try:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # create collection if needed
        existing = [c.name for c in qdrant.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            print(f"      Creating collection '{QDRANT_COLLECTION}' …")
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        else:
            print(f"      Collection '{QDRANT_COLLECTION}' already exists – appending.")
    except Exception as e:
        print(f"ERROR connecting to Qdrant: {e}")
        sys.exit(1)

    # ── Migrate in batches ────────────────────────────────────────────────────
    print(f"\n[3/4] Migrating {total} vectors in batches of {BATCH_SIZE} …")
    migrated = 0
    offset   = 0

    while offset < total:
        result = chroma_col.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )

        ids        = result["ids"]
        embeddings = result["embeddings"]
        documents  = result.get("documents") or [""] * len(ids)
        metadatas  = result.get("metadatas") or [{}] * len(ids)

        if not ids:
            break

        points = []
        for i, (chroma_id, vector, doc, meta) in enumerate(
            zip(ids, embeddings, documents, metadatas)
        ):
            # Qdrant numeric IDs only – use sequential int as id,
            # store the original chroma id inside payload.
            payload = dict(meta or {})
            payload["page_content"] = doc or ""
            payload["chroma_id"]    = chroma_id

            points.append(
                PointStruct(
                    id=offset + i,
                    vector=list(vector),
                    payload=payload,
                )
            )

        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        migrated += len(points)
        offset   += BATCH_SIZE
        print(f"      Migrated {migrated}/{total} …", end="\r")

    print(f"\n      Done – {migrated} vectors uploaded.")

    # ── Verify ────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Verifying …")
    info = qdrant.get_collection(QDRANT_COLLECTION)
    print(f"      Qdrant collection '{QDRANT_COLLECTION}' now has "
          f"{info.points_count} points.")
    print("\n✅  Migration complete!\n")


if __name__ == "__main__":
    main()
