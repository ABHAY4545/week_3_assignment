"""Checks for a controlled comparison without making provider calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from comparison import compare_reranking, stream_comparison
from retrieval_pipeline import RetrievalPipeline


def candidate(index):
    return {"chunk_id": str(index), "combined_score": 1 / index, "payload": {
        "chunk_id": str(index), "paper_id": str(index), "title": f"Paper {index}",
        "authors": "Author", "text": f"Evidence {index}", "chunk_section": "Method",
    }}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [candidate(i) for i in range(1, 4)]
        reranked = [{**self.candidates[i], "rerank_score": score} for i, score in [(2, .9), (0, .8)]]
        self.retrieval = SimpleNamespace(
            config=SimpleNamespace(cohere_api_key="test"),
            hybrid_search=Mock(return_value=self.candidates),
            rerank=Mock(return_value=reranked), to_results=RetrievalPipeline.to_results,
        )
        self.generator = SimpleNamespace(
            retrieval=self.retrieval, config=SimpleNamespace(llm_model="test-model"),
            refine_query=Mock(return_value="shared refined query"),
            _format_context=lambda results: ",".join(r.title for r in results),
            stream_answer=Mock(side_effect=lambda query, context: iter([context])),
            _build_sources_metadata=lambda results: [{"title": r.title} for r in results],
        )

    def test_same_candidates_and_query_with_distinct_contexts(self):
        result = compare_reranking(self.generator, "question", 2)
        self.generator.refine_query.assert_called_once_with("question")
        self.retrieval.hybrid_search.assert_called_once_with("shared refined query", semantic_top_k=30, bm25_top_k=30)
        self.assertIs(self.retrieval.rerank.call_args.args[1], self.candidates)
        self.assertEqual(result["without_reranking"]["answer"], "Paper 1,Paper 2")
        self.assertEqual(result["with_reranking"]["answer"], "Paper 3,Paper 1")
        self.assertEqual(result["shared_excerpts"], 1)
        self.assertEqual(result["changed_positions"], 2)
        self.assertEqual([r["hybrid_rank"] for r in result["with_reranking"]["ranking"]], [3, 1])

    def test_fallback_is_not_misrepresented_as_reranking(self):
        self.retrieval.rerank.return_value = self.candidates[:2]
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            compare_reranking(self.generator, "question", 2)
        self.generator.stream_answer.assert_not_called()

    def test_missing_key_stops_before_search(self):
        self.retrieval.config.cohere_api_key = None
        with self.assertRaisesRegex(ValueError, "COHERE_API_KEY"):
            compare_reranking(self.generator, "question", 2)
        self.retrieval.hybrid_search.assert_not_called()

    def test_failed_answer_keeps_rankings_and_other_answer(self):
        self.generator.stream_answer.side_effect = [RuntimeError("provider down"), iter(["second ", "answer"])]
        result = compare_reranking(self.generator, "question", 2)
        self.assertTrue(result["without_reranking"]["answer_error"])
        self.assertEqual(len(result["without_reranking"]["ranking"]), 2)
        self.assertEqual(result["with_reranking"]["answer"], "second answer")

    def test_rankings_and_chunks_arrive_before_completion(self):
        events = stream_comparison(self.generator, "question", 2)
        while True:
            event = next(events)
            if event['type'] == 'rankings':
                break
        self.generator.stream_answer.assert_not_called()
        self.assertEqual(event['result']['without_reranking']['answer'], '')
        self.assertEqual(next(events)['type'], 'answer_start')
        chunk = next(events)
        self.assertEqual(chunk['type'], 'answer_chunk')
        self.assertEqual(chunk['variant'], 'without_reranking')
        self.assertEqual(chunk['content'], 'Paper 1,Paper 2')
        self.assertEqual(self.generator.stream_answer.call_count, 1)
        remaining = list(events)
        self.assertEqual(remaining[-1]['type'], 'complete')
        self.assertTrue(any(e['type'] == 'answer_chunk' and e['variant'] == 'with_reranking' for e in remaining))

    def test_endpoint_validates_bounds_and_forces_controlled_settings(self):
        from fastapi.testclient import TestClient
        import api_server
        client = TestClient(api_server.app)
        self.assertEqual(client.post('/api/compare', json={"query":"x", "top_k":0}).status_code, 422)
        self.assertEqual(client.post('/api/compare', json={"query":" "}).status_code, 503)
        with patch.object(api_server, 'rag_generator', object()), patch.object(api_server, 'configured_generator', return_value=self.generator) as configure:
            self.assertEqual(client.post('/api/compare', json={"query":" "}).status_code, 422)
            response = client.post('/api/compare', json={"query":"question", "top_k":2})
            self.assertEqual(response.status_code, 200)
            configure.assert_called_once_with(True, True, temperature=0)
            self.assertEqual(response.json()['candidate_count'], 3)
            streaming = client.post('/api/compare/stream', json={"query":"question", "top_k":2})
            self.assertEqual(streaming.status_code, 200)
            self.assertIn('text/event-stream', streaming.headers['content-type'])
            self.assertLess(streaming.text.index('"type": "rankings"'), streaming.text.index('"type": "answer_chunk"'))
            self.assertIn('"type": "complete"', streaming.text)


class AnswerStreamingTests(unittest.TestCase):
    def test_provider_chunks_are_yielded_incrementally(self):
        from rag_generate import RAGGenerator
        generator = object.__new__(RAGGenerator)
        generator.config = SimpleNamespace(llm_model="free-test", temperature=0, max_tokens=2000)
        generator.openrouter_api_key = "test"
        generator.openrouter_base_url = "https://example.invalid"
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_lines.return_value = iter([
            b'data: {"choices":[{"delta":{"content":"first "}}]}',
            b'data: {"choices":[{"delta":{"content":"second"},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ])
        with patch('rag_generate.requests.post', return_value=response) as post:
            tokens = generator.stream_answer('question', 'context')
            self.assertEqual(next(tokens), 'first ')
            self.assertEqual(list(tokens), ['second'])
            self.assertTrue(post.call_args.kwargs['stream'])
            self.assertTrue(post.call_args.kwargs['json']['stream'])
            response.iter_lines.assert_called_once_with(chunk_size=1)



if __name__ == '__main__':
    unittest.main()
