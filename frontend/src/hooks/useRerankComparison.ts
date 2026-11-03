'use client'
import { useState } from 'react'
import type { Source } from '@/components/rag/ReferencesSection'

export interface RankedExcerpt {
  rank: number
  hybrid_rank: number
  chunk_id: string
  title: string
  section: string
  excerpt: string
  score: number
}
export interface ComparisonVariant {
  answer: string
  answer_error: string | null
  answer_status: 'pending' | 'streaming' | 'complete' | 'error'
  sources: Source[]
  retrieval_seconds: number
  generation_seconds: number
  ranking: RankedExcerpt[]
}
export interface RerankComparison {
  query: string
  search_query: string
  model: string
  candidate_count: number
  excerpt_count: number
  shared_excerpts: number
  changed_positions: number
  rerank_seconds: number
  total_seconds: number
  without_reranking: ComparisonVariant
  with_reranking: ComparisonVariant
}
type VariantName = 'without_reranking' | 'with_reranking'
interface ComparisonEvent {
  type: string
  result?: RerankComparison
  variant?: VariantName
  content?: string
  message?: string
  generation_seconds?: number
  total_seconds?: number
  answer_error?: string | null
  answer_status?: ComparisonVariant['answer_status']
}
export function useRerankComparison() {
  const [result, setResult] = useState<RerankComparison | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [activeVariant, setActiveVariant] = useState<VariantName>('without_reranking')
  const reset = () => { setResult(null); setError(''); setMessage(''); setActiveVariant('without_reranking') }
  const run = async (query: string, topK: number, refineQuery: boolean, useReranker: boolean) => {
    reset()
    const preferredVariant = useReranker ? 'with_reranking' : 'without_reranking'
    setActiveVariant(preferredVariant)
    setIsLoading(true)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 360000)
    try {
      const apiUrl = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8082').replace(/\/$/, '')
      const response = await fetch(`${apiUrl}/api/compare/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({query, top_k: topK, refine_query: refineQuery, use_reranker: useReranker}), signal: controller.signal,
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(typeof body?.detail === 'string' ? body.detail : `Comparison failed (HTTP ${response.status}).`)
      }
      const reader = response.body?.getReader()
      if (!reader) throw new Error('The response could not be streamed. Please retry.')
      const decoder = new TextDecoder()
      let buffer = ''
      let completed = false
      const handleEvent = (event: ComparisonEvent) => {
        if (event.type === 'error') throw new Error(event.message || 'Comparison failed.')
        if (event.type === 'progress') setMessage(event.message || 'Comparing…')
        if (event.type === 'rankings' && event.result) setResult(event.result)
        if (event.type === 'answer_start' && event.variant) {
          setMessage(event.variant === 'without_reranking' ? 'Streaming answer without reranking…' : 'Streaming answer with reranking…')
        }
        if (event.variant) {
          const name = event.variant
          setResult(previous => {
            if (!previous) return previous
            const variant = previous[name]
            return { ...previous, [name]: {
              ...variant,
              answer: event.type === 'answer_chunk' ? variant.answer + (event.content || '') : variant.answer,
              answer_status: event.type === 'answer_start' ? 'streaming' : event.answer_status || variant.answer_status,
              answer_error: event.answer_error ?? variant.answer_error,
              generation_seconds: event.generation_seconds ?? variant.generation_seconds,
            } }
          })
        }
        if (event.type === 'complete') {
          completed = true
          setResult(previous => previous ? {...previous, total_seconds: event.total_seconds ?? previous.total_seconds} : previous)
          setMessage('')
        }
      }
      try {
        while (true) {
          const {done, value} = await reader.read()
          buffer += done ? decoder.decode() : decoder.decode(value, {stream:true})
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          if (done && buffer) { lines.push(buffer); buffer = '' }
          for (const line of lines) {
            if (line.startsWith('data: ')) handleEvent(JSON.parse(line.slice(6)))
          }
          if (done) break
        }
        if (!completed) throw new Error('The comparison stream ended early. Retry to complete both answers.')
      } finally {
        await reader.cancel().catch(() => {})
        reader.releaseLock()
      }
    } catch (error) {
      setError(error instanceof Error && error.name === 'AbortError'
        ? 'The comparison took too long. Try fewer excerpts or retry.'
        : error instanceof TypeError ? 'Cannot connect to the research service. Start the backend and retry.'
        : error instanceof Error ? error.message : 'Comparison failed. Please retry.')
    } finally {
      clearTimeout(timeout)
      setIsLoading(false)
    }
  }
  return { result, isLoading, error, message, activeVariant, run, reset }
}
