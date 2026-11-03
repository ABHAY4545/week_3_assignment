"""Controlled reranking demo: one refined query and one shared candidate pool."""
import traceback
from time import perf_counter


def stream_comparison(generator, query: str, top_k: int, use_reranker: bool = True):
    retrieval = generator.retrieval
    if not retrieval.config.cohere_api_key:
        raise ValueError("Comparison requires COHERE_API_KEY. Configure it and restart the backend.")

    started = perf_counter()
    yield {"type": "progress", "message": "Preparing the shared search query…"}
    refined = generator.refine_query(query)
    candidate_count = max(30, top_k * 2)
    yield {"type": "progress", "message": "Retrieving a shared pool of hybrid search candidates…"}
    search_started = perf_counter()
    candidates = retrieval.hybrid_search(
        refined, semantic_top_k=candidate_count, bm25_top_k=candidate_count
    )
    search_seconds = perf_counter() - search_started
    if not candidates:
        raise ValueError("No matching excerpts found. Try a more specific research question.")

    yield {"type": "progress", "message": "Reranking the same candidates with Cohere…"}
    rerank_started = perf_counter()
    reranked = retrieval.rerank(refined, candidates, top_k=top_k)
    rerank_seconds = perf_counter() - rerank_started
    # The normal search gracefully falls back; a comparison must reveal that failure.
    if len(reranked) != min(top_k, len(candidates)) or any("rerank_score" not in r for r in reranked):
        raise RuntimeError("Cohere reranking was unavailable. No comparison was produced; try again or check the Cohere configuration.")

    baseline = candidates[:top_k]
    original_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(candidates)}

    def variant(ranked, score_key, retrieval_seconds):
        results = retrieval.to_results(ranked)
        return {
            "answer": "",
            "answer_error": None,
            "answer_status": "pending",
            "sources": generator._build_sources_metadata(results),
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": 0,
            "ranking": [{
                "rank": i + 1,
                "hybrid_rank": original_ranks[r["chunk_id"]],
                "chunk_id": r["chunk_id"],
                "title": r["payload"]["title"],
                "section": r["payload"].get("chunk_section", ""),
                "excerpt": r["payload"]["text"][:500],
                "score": r[score_key],
            } for i, r in enumerate(ranked)],
        }

    without = variant(baseline, "combined_score", search_seconds)
    with_reranking = variant(reranked, "rerank_score", search_seconds + rerank_seconds)
    baseline_ids = {r["chunk_id"] for r in baseline}
    reranked_ids = {r["chunk_id"] for r in reranked}
    result = {
        "query": query,
        "search_query": refined,
        "model": generator.config.llm_model,
        "candidate_count": len(candidates),
        "excerpt_count": len(baseline),
        "shared_excerpts": len(baseline_ids & reranked_ids),
        "changed_positions": sum(a["chunk_id"] != b["chunk_id"] for a, b in zip(baseline, reranked)),
        "rerank_seconds": rerank_seconds,
        "total_seconds": perf_counter() - started,
        "without_reranking": without,
        "with_reranking": with_reranking,
    }

    yield {"type": "rankings", "result": result}
    variants = (("without_reranking", baseline), ("with_reranking", reranked))
    if use_reranker:
        variants = tuple(reversed(variants))
    for name, ranked in variants:
        current = result[name]
        current["answer_status"] = "streaming"
        yield {"type": "answer_start", "variant": name}
        generation_started = perf_counter()
        try:
            context = generator._format_context(retrieval.to_results(ranked))
            for content in generator.stream_answer(query, context):
                current["answer"] += content
                yield {"type": "answer_chunk", "variant": name, "content": content}
            current["answer_status"] = "complete"
        except Exception as error:
            traceback.print_exc()
            current["answer_status"] = "error"
            current["answer_error"] = f"Answer generation failed: {error}"
        current["generation_seconds"] = perf_counter() - generation_started
        yield {"type": "answer_done", "variant": name, "generation_seconds": current["generation_seconds"], "answer_error": current["answer_error"], "answer_status": current["answer_status"]}
    result["total_seconds"] = perf_counter() - started
    yield {"type": "complete", "total_seconds": result["total_seconds"]}


def compare_reranking(generator, query: str, top_k: int, use_reranker: bool = True) -> dict:
    """Nonstreaming interface for scripts; the UI uses stream_comparison."""
    result = None
    for event in stream_comparison(generator, query, top_k, use_reranker):
        if event["type"] == "rankings":
            result = event["result"]
    return result
