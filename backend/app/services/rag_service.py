"""
RAG (Retrieval Augmented Generation) service for document-based Q&A.
Backed by Qdrant Cloud vector store.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio
import time
from functools import partial

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based document retrieval and question answering."""

    def __init__(self):
        """Initialize RAG service with embeddings and Qdrant vector store."""
        self.embeddings = None
        self.vector_store = None
        self.qdrant_client = None
        self.llm = None
        self.groq_llm = None
        self.text_splitter = None
        self.initialized = False
        try:
            self._initialize()
            self.initialized = True
        except Exception as e:
            logger.error(
                f"RAG service failed to initialise (auth/chat may still work): {e}"
            )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self):
        """Initialize all components."""
        try:
            # Embedding model
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            # Qdrant Cloud client
            # timeout=120: the default (~5s) was too short — the upsert HTTP
            # request was being killed before Qdrant could respond.
            logger.info(f"Connecting to Qdrant Cloud: {settings.qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=120,
            )

            # Ensure collection exists
            self._ensure_collection()

            # LangChain Qdrant vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.qdrant_collection_name,
                embedding=self.embeddings,
            )

            # LLM - primary (Gemini)
            if settings.llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=settings.llm_model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    openai_api_key=settings.openai_api_key,
                )
            elif settings.llm_provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    google_api_key=settings.google_api_key,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

            # LLM - secondary (Groq)
            if settings.groq_api_key:
                try:
                    self.groq_llm = ChatGroq(
                        model=settings.groq_model,
                        temperature=settings.llm_temperature,
                        groq_api_key=settings.groq_api_key,
                    )
                    logger.info(f"Groq LLM initialised with model: {settings.groq_model}")
                except Exception as groq_err:
                    logger.warning(f"Groq LLM failed to initialise: {groq_err}")
            else:
                logger.info("GROQ_API_KEY not set — Groq model unavailable")

            logger.info("RAG service initialised successfully with Qdrant Cloud")

        except Exception as e:
            logger.error(f"Failed to initialise RAG service: {e}")
            raise  # Re-raised so __init__ can catch and mark initialized=False

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist yet, and ensure
        keyword payload indexes exist on all filterable metadata fields."""
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if settings.qdrant_collection_name not in collections:
            logger.info(
                f"Creating Qdrant collection: {settings.qdrant_collection_name}"
            )
            self.qdrant_client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(
                f"Qdrant collection '{settings.qdrant_collection_name}' already exists"
            )

        # Ensure keyword indexes exist for all filterable metadata fields.
        # Two schemas coexist in the collection:
        #   Schema A (PDF uploads)   : flat top-level keys  e.g.  device_type
        #   Schema B (ChromaDB mig.) : nested under metadata e.g.  metadata.device_type
        # We index BOTH paths so Qdrant can filter efficiently on either.
        # create_payload_index is idempotent — safe to call every startup.
        _filter_fields = [
            "device_type",
            "brand",
            "model",
            "document_id",
            # nested paths for ChromaDB-migrated points
            "metadata.device_type",
            "metadata.brand",
            "metadata.model",
            "metadata.document_id",
        ]
        for field in _filter_fields:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=settings.qdrant_collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Payload index ensured for field: {field}")
            except Exception as idx_err:
                # Index may already exist with different schema — log and continue.
                logger.warning(f"Could not create payload index for '{field}': {idx_err}")

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    def create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for troubleshooting."""
        template = (
            "You are a friendly and knowledgeable device assistant helping a user with their product.\n\n"
            "Answer based ONLY on the manual excerpts below. Follow these rules:\n"
            "- Write in a natural, conversational tone — like a helpful friend, not a technical document.\n"
            "- Be concise. Summarise the manual content in plain language instead of quoting it verbatim.\n"
            "- If the answer involves steps, use a clean numbered list.\n"
            "- Do NOT put source references mid-sentence. Add one clean '📖 Source:' line at the very end.\n"
            "- If an important safety tip is mentioned in the excerpts, include it naturally in your answer.\n"
            "- If the excerpts don't contain enough information, say: "
            "'I don\\'t have enough detail in the available manuals for that. "
            "Please check the full manual or contact support.'\n"
            "- Never invent information not present in the excerpts.\n\n"
            "MANUAL EXCERPTS:\n{context}\n\n"
            "USER QUESTION:\n{question}\n\n"
            "YOUR ANSWER:"
        )

        return PromptTemplate(template=template, input_variables=["context", "question"])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve_relevant_chunks(
        self,
        query: str,
        device_type: Optional[str] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query using Qdrant filters.

        Fetches 2x candidates then deduplicates by content similarity so the
        final top_k chunks are diverse — preventing 5 near-identical chunks
        from the same manual page from dominating the context window.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        # Fetch extra candidates so dedup doesn't leave us short
        fetch_k = top_k * 2

        try:
            # Build Qdrant filter from optional metadata fields
            qdrant_filter = self._build_filter(device_type=device_type, brand=brand, model=model)

            # Similarity search (run blocking call in executor)
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                partial(
                    self.vector_store.similarity_search_with_score,
                    query,
                    k=fetch_k,
                    filter=qdrant_filter,
                ),
            )

            # ------------------------------------------------------------------
            # Deduplicate: drop chunks whose first 150 chars overlap heavily
            # with an already-accepted chunk.  This prevents 5 consecutive
            # overlapping chunks from the same page filling the entire context.
            # ------------------------------------------------------------------
            seen_prefixes: list[str] = []
            chunks = []

            for doc, score in results:
                relevance_score = float(score)
                if relevance_score < settings.relevance_threshold:
                    continue

                # Use the first 150 chars as a near-duplicate fingerprint
                prefix = doc.page_content[:150].lower().strip()
                is_dup = any(
                    self._overlap_ratio(prefix, s) > 0.7
                    for s in seen_prefixes
                )
                if is_dup:
                    continue

                seen_prefixes.append(prefix)
                chunks.append(
                    {
                        "content": doc.page_content,
                        "source_file": self._get_meta(doc.metadata, "source_file", "Unknown"),
                        "page_number": self._get_meta(doc.metadata, "page_number"),
                        "section_name": self._get_meta(doc.metadata, "section_name"),
                        "relevance_score": round(relevance_score, 3),
                        "device_type": self._get_meta(doc.metadata, "device_type"),
                        "brand": self._get_meta(doc.metadata, "brand"),
                        "model": self._get_meta(doc.metadata, "model"),
                    }
                )

                if len(chunks) >= top_k:
                    break

            logger.info(f"Retrieved {len(chunks)} diverse chunks for query: {query[:50]}...")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    # ------------------------------------------------------------------
    # Title generation
    # ------------------------------------------------------------------

    async def generate_title(self, first_message: str, ai_model: str = "gemini") -> str:
        """Generate a short, descriptive chat title from the first user message.

        Returns a 4-6 word title, or a truncated fallback if the LLM fails.
        """
        try:
            prompt = (
                "Generate a short, descriptive title (4-6 words max) for a chat session "
                "that starts with this user message. Reply with ONLY the title — no quotes, "
                "no punctuation at the end, no explanation.\n\n"
                f"User message: {first_message[:300]}"
            )

            active_llm = (
                self.groq_llm
                if ai_model == "groq" and self.groq_llm is not None
                else self.llm
            )

            if active_llm is None:
                raise ValueError("No LLM available")

            response = await active_llm.ainvoke(prompt)
            title = (response.content if hasattr(response, "content") else str(response)).strip()

            # Strip surrounding quotes the LLM sometimes adds
            title = title.strip('"\'')

            # Hard cap at 60 chars
            return title[:60] if title else first_message[:40]

        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            # Graceful fallback: first 40 chars of the message
            return first_message[:40].strip()

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    async def generate_answer(
        self,
        query: str,
        device_type: Optional[str] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
        ai_model: Optional[str] = "gemini",
    ) -> Dict[str, Any]:
        """Generate an answer using RAG."""
        try:
            chunks = await self.retrieve_relevant_chunks(
                query=query,
                device_type=device_type,
                brand=brand,
                model=model,
            )

            if not chunks:
                return {
                    "answer": (
                        "I don't have specific information about this issue in the available manuals. "
                        "I recommend checking the device's official manual or contacting customer support."
                    ),
                    "sources": [],
                }

            # Build context — include section name and relevance so the LLM
            # can see how well each chunk matches and prioritise accordingly.
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                section = chunk.get("section_name") or "General"
                context_parts.append(
                    f"[Excerpt {i} | Source: {chunk['source_file']} | "
                    f"Page: {chunk.get('page_number', 'N/A')} | "
                    f"Section: {section} | Relevance: {chunk['relevance_score']:.0%}]\n"
                    f"{chunk['content']}"
                )
            context = "\n\n---\n\n".join(context_parts)

            prompt_template = self.create_prompt_template()
            prompt = prompt_template.format(context=context, question=query)

            # Select LLM based on ai_model param
            if ai_model == "groq" and self.groq_llm is not None:
                active_llm = self.groq_llm
                logger.info("Using Groq LLM for this request")
            else:
                active_llm = self.llm
                if ai_model == "groq":
                    logger.warning("Groq LLM not available, falling back to primary LLM")

            response = await active_llm.ainvoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            sources = [
                {
                    "content": chunk["content"][:200] + "..."
                    if len(chunk["content"]) > 200
                    else chunk["content"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk.get("page_number"),
                    "section_name": chunk.get("section_name"),
                    "relevance_score": chunk["relevance_score"],
                }
                for chunk in chunks[:5]
            ]

            return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 25,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> int:
        """Add documents to the Qdrant vector store in small batches with retries.

        Two-phase upload per batch:
          1. Embed texts on CPU  (no Qdrant connection held open – no timeout risk)
          2. Upsert pre-computed vectors to Qdrant  (fast network call)

        If the upsert step times out, it is retried up to `max_retries` times
        with a `retry_delay`-second pause.  Embeddings are cached so we never
        recompute them on a retry.
        """
        # Safety net: ensure collection exists even if init failed earlier
        if self.qdrant_client is not None:
            try:
                self._ensure_collection()
            except Exception as ec:
                logger.warning(f"ensure_collection in add_documents failed: {ec}")

        total = len(texts)
        added = 0
        loop = asyncio.get_running_loop()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_metas = metadatas[start:end]
            batch_num = start // batch_size + 1

            # ── Step 1: Embed texts (CPU-bound; runs once per batch) ───────
            logger.info(f"Embedding batch {batch_num} ({start}-{end-1} of {total})…")
            embeddings_list: List[List[float]] = await loop.run_in_executor(
                None,
                partial(self.embeddings.embed_documents, batch_texts),
            )

            # ── Step 2: Build PointStructs ─────────────────────────────
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "page_content": text,
                        "metadata": meta,
                    },
                )
                for text, meta, embedding in zip(
                    batch_texts, batch_metas, embeddings_list
                )
            ]

            # ── Step 3: Upload to Qdrant (with retry on timeout) ─────────
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    await loop.run_in_executor(
                        None,
                        partial(
                            self.qdrant_client.upsert,
                            collection_name=settings.qdrant_collection_name,
                            points=points,
                        ),
                    )
                    added += len(batch_texts)
                    logger.info(
                        f"Batch {batch_num} uploaded ✓  "
                        f"({added}/{total} chunks done)"
                        + (f"  [attempt {attempt}]" if attempt > 1 else "")
                    )
                    last_error = None
                    break  # success — move to next batch
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Batch {batch_num} attempt {attempt} failed: {e}. "
                            f"Retrying in {retry_delay}s…"
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Batch {batch_num} failed after {max_retries} attempts: {e}"
                        )

            if last_error is not None:
                raise last_error  # propagate only after all retries exhausted

        logger.info(f"Successfully stored all {added} chunks in Qdrant")
        return added

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> bool:
        """Delete all vectors belonging to a document from Qdrant.

        Handles both payload schemas:
          Schema A (PDF uploads)   – document_id at top level
          Schema B (ChromaDB mig.) – document_id nested under metadata
        """
        try:
            # Safety net: ensure collection exists before trying to delete from it
            if self.qdrant_client is not None:
                self._ensure_collection()
            self.qdrant_client.delete(
                collection_name=settings.qdrant_collection_name,
                points_selector=Filter(
                    should=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        ),
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        ),
                    ]
                ),
            )
            logger.info(f"Deleted document {document_id} from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_meta(self, metadata: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Read a metadata field from either the flat schema (PDF uploads) or the
        nested schema (ChromaDB-migrated points where fields live under 'metadata').
        Flat schema takes priority if both are present.
        """
        # Flat schema (Schema A – PDF uploads)
        if key in metadata and metadata[key] is not None:
            return metadata[key]
        # Nested schema (Schema B – ChromaDB migration)
        nested = metadata.get("metadata") or {}
        if isinstance(nested, dict) and key in nested and nested[key] is not None:
            return nested[key]
        return default

    @staticmethod
    def _overlap_ratio(a: str, b: str) -> float:
        """Estimate character-level overlap between two short strings."""
        if not a or not b:
            return 0.0
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a:
            return 0.0
        return len(set_a & set_b) / len(set_a)

    def _build_filter(
        self,
        device_type: Optional[str],
        brand: Optional[str],
        model: Optional[str],
    ) -> Optional[Filter]:
        """Build a Qdrant Filter from optional metadata fields.

        Two payload schemas coexist in the collection:
          Schema A (PDF uploads)   – flat keys:            device_type, brand, model
          Schema B (ChromaDB mig.) – nested under metadata: metadata.device_type, …

        For each provided filter value we create a 'should' (OR) sub-filter that
        matches either schema, then wrap all sub-filters in a 'must' (AND) block
        so that multiple filters are applied together.
        """
        must_conditions = []

        if device_type:
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="device_type",          match=MatchValue(value=device_type)),
                    FieldCondition(key="metadata.device_type", match=MatchValue(value=device_type)),
                ])
            )
        if brand:
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="brand",          match=MatchValue(value=brand)),
                    FieldCondition(key="metadata.brand", match=MatchValue(value=brand)),
                ])
            )
        if model:
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="model",          match=MatchValue(value=model)),
                    FieldCondition(key="metadata.model", match=MatchValue(value=model)),
                ])
            )

        return Filter(must=must_conditions) if must_conditions else None


# Global RAG service instance
rag_service = RAGService()
