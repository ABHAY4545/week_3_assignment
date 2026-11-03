#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM via OpenRouter API.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT
    
    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenerationConfig:
    """Configuration for the RAG generator."""
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free")
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 2000  # Max length of generated answer
    openrouter_api_key: Optional[str] = None  # Will load from env if not set
    refine_query: bool = True  # Whether to refine queries before retrieval
    refinement_model: str = field(default_factory=lambda: os.getenv("REFINEMENT_MODEL") or os.getenv("LLM_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free")
    retrieval_top_k: int = 8  # Number of chunks to retrieve


# =============================================================================
# RAG GENERATOR CLASS
# =============================================================================
class RAGGenerator:
    """
    Main RAG class - this is what api_server.py uses!
    
    Flow:
    1. Refine the user's query (optional)
    2. Retrieve relevant chunks using the retrieval pipeline
    3. Format chunks into context
    4. Generate answer using LLM
    5. Return answer with source metadata
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        """
        Initialize the RAG generator.
        
        Args:
            config: Optional configuration object
            retrieval_pipeline: Optional pre-initialized retrieval pipeline
        """
        self.config = config or GenerationConfig()
        self.openrouter_api_key = self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.retrieval = retrieval_pipeline if retrieval_pipeline is not None else RetrievalPipeline()
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
    
    def refine_query(self, query: str) -> str:
        """
        Use LLM to improve the search query (optional but helps retrieval).
        
        Args:
            query: Original user query
            
        Returns:
            Refined query (or original if refinement disabled/fails)
        """
        if not self.config.refine_query:
            return query

        try:
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.refinement_model,
                    "messages": [{
                        "role": "user",
                        "content": QUERY_REFINEMENT_PROMPT.format(query=query),
                    }],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                timeout=30,
            )
            response.raise_for_status()
            refined = response.json()["choices"][0]["message"]["content"]
            if not isinstance(refined, str):
                return query
            return refined.strip().strip("\"'").strip() or query
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            return query
    
    def _format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            results: List of RetrievalResult objects
            
        Returns:
            Formatted context string
        """
        return "\n\n".join(
            f"--- Source {i} ---\n"
            f"Title: {result.title}\n"
            f"Authors: {result.authors}\n"
            f"Section: {result.chunk_section}\n\n"
            f"Content:\n{result.text}"
            for i, result in enumerate(results, start=1)
        )
    
    def _build_sources_metadata(self, results: list[RetrievalResult]) -> list[dict]:
        """
        Build list of unique source papers for citations.
        The frontend displays these as clickable source links.
        
        Args:
            results: List of RetrievalResult objects
            
        Returns:
            List of unique source metadata dicts
        """
        seen = {}
        for number, result in enumerate(results, start=1):
            if result.title not in seen:
                seen[result.title] = {
                    "excerpts": [],
                    "title": result.title,
                    "authors": result.authors,
                    "pdf_url": result.pdf_url,
                    "github_link": result.github_link,
                    "video_link": result.video_link,
                    "acm_url": result.acm_url,
                    "abstract_url": result.abstract_url,
                }
            # "Source N" in the answer is the Nth excerpt; record which paper supplied it.
            seen[result.title]["excerpts"].append(number)
        return list(seen.values())
    
    def _call_llm(self, query: str, context: str) -> str:
        """
        Call OpenRouter API to generate an answer.
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            Generated answer string
        """
        user_message = (
            "Based on the following research paper excerpts, answer this question.\n\n"
            f"Question: {query}\n\n"
            f"Research Paper Excerpts:\n{context}\n\n"
            "Remember to cite papers using [Paper Title] format."
        )
        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        response_data = response.json()
        if "error" in response_data:
            raise RuntimeError(f"OpenRouter API error: {response_data['error']}")
        answer = response_data["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("OpenRouter returned an empty or invalid answer")
        return answer
    
    def stream_answer(self, query: str, context: str):
        """Yield answer tokens as OpenRouter sends them."""
        user_message = (
            "Based on the following research paper excerpts, answer this question.\n\n"
            f"Question: {query}\n\nResearch Paper Excerpts:\n{context}\n\n"
            "Remember to cite papers using [Paper Title] format."
        )
        import json
        with requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"},
            json={"model": self.config.llm_model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ], "temperature": self.config.temperature, "max_tokens": self.config.max_tokens, "stream": True},
            stream=True, timeout=(10, 120),
        ) as response:
            if not response.ok:
                detail = response.text[:300].strip()
                raise RuntimeError(f"OpenRouter returned HTTP {response.status_code} for {self.config.llm_model}: {detail}")
            received = False
            complete = False
            for line in response.iter_lines(chunk_size=1):
                if not line or not line.startswith(b"data: "):
                    continue
                raw = line[6:]
                if raw == b"[DONE]":
                    complete = True
                    break
                data = json.loads(raw)
                if "error" in data:
                    error = data["error"]
                    detail = error.get("message") if isinstance(error, dict) else str(error)
                    raise RuntimeError(f"The model provider interrupted answer generation: {detail}")
                for choice in data.get("choices", []):
                    if choice.get("finish_reason"):
                        complete = True
                    content = choice.get("delta", {}).get("content")
                    if content:
                        received = True
                        yield content
            if not received or not complete:
                raise RuntimeError("The model provider returned an empty or incomplete answer.")

    def generate(self, query: str, top_k: Optional[int] = None, return_sources: bool = True) -> dict:
        """
        Full RAG pipeline - retrieve relevant chunks and generate an answer.
        THIS IS THE MAIN METHOD THAT api_server.py CALLS!
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (uses config default if None)
            return_sources: Whether to include source metadata
            
        Returns:
            Dict with query, refined_query, answer, and sources
        """
        refined = self.refine_query(query)
        results = self.retrieval.retrieve(
            refined, top_k=self.config.retrieval_top_k if top_k is None else top_k
        )
        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
            }

        context = self._format_context(results)
        answer = self._call_llm(query, context)
        return {
            "query": query,
            "refined_query": refined,
            "answer": answer,
            "sources": self._build_sources_metadata(results) if return_sources else [],
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================
if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"
    
    print("Initializing RAG Generator...")
    generator = RAGGenerator()
    
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    result = generator.generate(query)
    
    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result['answer'])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get('sources', []):
        print(f"  - {source['title']}")
