"""
Discover what is actually indexed in Qdrant.
Prints unique device_type / brand / model values and a content sample.
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")  # .env lives in backend/

from qdrant_client import QdrantClient

url = os.environ["QDRANT_URL"]
key = os.environ["QDRANT_API_KEY"]
col = os.environ.get("QDRANT_COLLECTION_NAME", "device_manuals")

client = QdrantClient(url=url, api_key=key, timeout=60)

# Count total points
info = client.get_collection(col)
total = info.points_count
print(f"\nCollection: '{col}'  |  Total points: {total}\n")

# Scroll through all points to collect unique combos + content samples
seen = {}          # key = (device_type, brand, model)
offset = None
batch  = 200
fetched = 0

while True:
    pts, offset = client.scroll(
        col, limit=batch, offset=offset,
        with_payload=True, with_vectors=False
    )
    for p in pts:
        pl = p.payload or {}
        # Handle both flat (Schema A) and nested (Schema B)
        meta = pl.get("metadata") or {}

        def get(key):
            return pl.get(key) or meta.get(key) or ""

        dt    = (get("device_type") or "").strip()
        brand = (get("brand")       or "").strip()
        model = (get("model")       or "").strip()
        text  = pl.get("page_content") or meta.get("page_content") or ""

        combo = (dt, brand, model)
        if combo not in seen:
            seen[combo] = text[:300]   # save a short content sample

    fetched += len(pts)
    if offset is None or len(pts) < batch:
        break

print(f"Unique (device_type, brand, model) combos found: {len(seen)}\n")
print(f"{'device_type':<25} {'brand':<20} {'model':<30}")
print("-" * 78)
for (dt, brand, model), sample in sorted(seen.items()):
    print(f"{(dt or '<none>'):<25} {(brand or '<none>'):<20} {(model or '<none>'):<30}")

# Save as JSON for the evaluator to use
out = Path(__file__).parent / "indexed_catalog.json"
catalog = [
    {"device_type": dt, "brand": brand, "model": model, "sample": sample}
    for (dt, brand, model), sample in seen.items()
]
with open(out, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2)

print(f"\nCatalog saved → {out}")
