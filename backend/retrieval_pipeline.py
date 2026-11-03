#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings via OpenRouter + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

Usage:
    from retrieval_pipeline import RetrievalPipeline
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import os
import re
import requests
import numpy as np
from typing import Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()

# Must match the collection name used in upload_to_qdrant.py
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "siggraph2025_papers"


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """
    
    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        """
        Initialize the embedder.

        Args:
            api_key: OpenRouter API key
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": text
        }

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        response_data = response.json()
        if "error" in response_data:
            raise RuntimeError(f"OpenRouter API error: {response_data['error']}")
            
        embedding = response_data["data"][0]["embedding"]
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("OpenRouter returned an invalid embedding vector")
        return vector


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches that semantic search might miss.
    """
    
    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries from chunks.json
        """
        self.chunks = chunks
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs) if any(self.tokenized_docs) else None
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase and extract alphanumeric words.

        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase word tokens
        """
        return re.findall(r'\w+', text.lower())
    
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Search for query and return top-k results.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        words = self._tokenize(query)
        if not words or top_k <= 0 or self.bm25 is None:
            return []
            
        bm25_scores = np.array(self.bm25.get_scores(words))
        indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = [
            (int(idx), float(bm25_scores[idx]))
            for idx in indices
            if bm25_scores[idx] > 0
        ]
        return results


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """
    
    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize all components of the retrieval pipeline.

        Args:
            config: Optional configuration. If None, loads from environment variables.
        """
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY") or None,
                chunks_path=os.getenv("CHUNKS_PATH") or "./chunks.json",
                embedding_model=os.getenv("EMBEDDING_MODEL") or "baai/bge-large-en-v1.5",
            ) 

        missing_fields = [
            f"config.{field}" for field in ["qdrant_url", "qdrant_api_key", "openrouter_api_key"]
            if not getattr(config, field)
        ]

        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")

        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        self.qdrant = self.client

        self.embedder = OpenRouterEmbedder(
            api_key=config.openrouter_api_key,
            model=config.embedding_model
        )

        # Handle chunks path (check direct path, or relative to backend directory)
        chunks_path = config.chunks_path or "./chunks.json"
        if not os.path.isfile(chunks_path):
            alt_path = os.path.join(os.path.dirname(__file__), chunks_path)
            if os.path.isfile(alt_path):
                chunks_path = alt_path
            else:
                alt_path2 = os.path.join(os.path.dirname(__file__), os.path.basename(chunks_path))
                if os.path.isfile(alt_path2):
                    chunks_path = alt_path2

        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data["chunks"] if isinstance(data, dict) and "chunks" in data else data

        self.bm25_index = BM25Index(self.chunks)
        print("BM25 index built")

        self.config = config
    
    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.

        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        if not query.strip() or top_k <= 0:
            return []
        query_embedding = self.embedder.embed_query(query)

        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        ).points

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    
    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        results = self.bm25_index.search(query, top_k)
        
        return [
            {
                "chunk_id": self.chunks[idx]["chunk_id"],
                "score": score,
                "payload": self.chunks[idx]
            }
            for idx, score in results
        ]
    
    def hybrid_search(self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.

        Args:
            query: Search query
            semantic_top_k: Max results from semantic search
            bm25_top_k: Max results from BM25 search
            
        Returns:
            Combined and sorted list of results
        """
        # 1. Get results from both search methods
        semantic_results = self.semantic_search(query, semantic_top_k)
        bm25_results = self.bm25_search(query, bm25_top_k)
        
        # 2. Normalize semantic scores (divide by max score)
        if semantic_results:
            max_semantic = max(r["score"] for r in semantic_results)
            for r in semantic_results:
                r["normalized_score"] = r["score"] / max_semantic if max_semantic > 0 else 0.0

        # 3. Normalize BM25 scores the same way
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["normalized_score"] = r["score"] / max_bm25 if max_bm25 > 0 else 0.0

        # 4. Combine results into a single dict keyed by chunk_id
        combined = {}

        # Semantic results
        for r in semantic_results:
            cid = r["chunk_id"]
            sem_score = r.get("normalized_score", 0.0)
            combined[cid] = {
                "chunk_id": cid,
                "payload": r["payload"],
                "semantic_score": sem_score,
                "bm25_score": 0.0,
                "combined_score": self.config.semantic_weight * sem_score
            }

        # BM25 results
        for r in bm25_results:
            cid = r["chunk_id"]
            bm_score = r.get("normalized_score", 0.0)
            if cid in combined:
                combined[cid]["bm25_score"] = bm_score
                combined[cid]["combined_score"] += self.config.bm25_weight * bm_score
            else:
                combined[cid] = {
                    "chunk_id": cid,
                    "payload": r["payload"],
                    "semantic_score": 0.0,
                    "bm25_score": bm_score,
                    "combined_score": self.config.bm25_weight * bm_score
                }

        # 5. Sort by combined_score descending
        results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        return results
    
    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).
        
        Args:
            query: Original query
            results: Results from hybrid_search
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of results
        """
        # 1. If no Cohere API key or no results, return results[:top_k]
        if not query.strip() or top_k <= 0:
            return []
        if not self.config.use_reranker or not self.config.cohere_api_key or not results:
            return results[:top_k]
        
        # 2. Extract texts for reranking
        texts = [r["payload"]["text"] for r in results]

        # 3. Call Cohere Rerank API
        body = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": texts,
            "top_n": min(top_k, len(texts))
        }
        headers = {
            "Authorization": f"Bearer {self.config.cohere_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers=headers,
                json=body,
                timeout=30
            )

            if response.status_code != 200:
                print(f"Cohere rerank returned status {response.status_code}: {response.text}")
                return results[:top_k]

            data = response.json()
            
            # 4. Parse response and reorder results based on Cohere's ranking
            reranked_results = []
            seen = set()
            for item in data.get("results", []):
                idx = item.get("index")
                if type(idx) is int and 0 <= idx < len(results) and idx not in seen:
                    score = float(item["relevance_score"])
                    if not np.isfinite(score):
                        continue
                    res = dict(results[idx])
                    res["rerank_score"] = score
                    reranked_results.append(res)
                    seen.add(idx)

            # A partial/malformed response must not silently drop candidates.
            if len(reranked_results) < min(top_k, len(results)):
                return results[:top_k]
            return reranked_results[:top_k]

        except Exception as e:
            print(f"Cohere rerank failed: {e}. Falling back to hybrid search results.")
            return results[:top_k]
    
    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!
        
        Args:
            query: User's search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects ready for RAG generation
        """
        if not query.strip() or top_k <= 0:
            return []

        # Retrieve enough candidates even when the caller requests more than 30.
        candidate_count = max(30, top_k * 2)
        candidates = self.hybrid_search(
            query, semantic_top_k=candidate_count, bm25_top_k=candidate_count
        )

        # 2. Rerank if enabled and key is available
        if self.config.use_reranker and self.config.cohere_api_key:
            reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        else:
            reranked = candidates
    
        # 3. Take top_k results
        final = reranked[:top_k]
    
        return self.to_results(final)

    @staticmethod
    def to_results(final: list[dict]) -> list[RetrievalResult]:
        """Convert ranked candidates without performing another search."""
        return [
            RetrievalResult(
                chunk_id=r["payload"]["chunk_id"],
                paper_id=r["payload"]["paper_id"],
                title=r["payload"]["title"],
                authors=r["payload"]["authors"],
                text=r["payload"]["text"],
                score=r.get("rerank_score", r.get("combined_score", r.get("score", 0.0))),
                chunk_type=r["payload"].get("chunk_type", ""),
                chunk_section=r["payload"].get("chunk_section", ""),
                pdf_url=r["payload"].get("pdf_url"),
                github_link=r["payload"].get("github_link"),
                video_link=r["payload"].get("video_link"),
                acm_url=r["payload"].get("acm_url"),
                abstract_url=r["payload"].get("abstract_url"),
            )
            for r in final
        ]


# For testing this file directly
if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"
    
    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()
