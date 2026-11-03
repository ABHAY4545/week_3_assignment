'use client'

import { useState, useEffect, useRef, FormEvent } from 'react'

interface QueryFormProps {
  onSubmit: (query: string, topK: number, refineQuery: boolean, useReranker: boolean, compare: boolean) => void
  isLoading: boolean
  suggestion?: { text: string; id: number }
  compact?: boolean
}

export function QueryForm({ onSubmit, isLoading, suggestion, compact }: QueryFormProps) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(8)
  const [refineQuery, setRefineQuery] = useState(true)
  const [useReranker, setUseReranker] = useState(true)
  const [compare, setCompare] = useState(false)
  const input = useRef<HTMLTextAreaElement>(null)

  const resize = (el: HTMLTextAreaElement | null) => {
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`
  }

  useEffect(() => {
    if (suggestion?.text) {
      setQuery(suggestion.text)
      input.current?.focus()
      requestAnimationFrame(() => resize(input.current))
    }
  }, [suggestion])

  useEffect(() => { resize(input.current) }, [compact])

  const submit = () => {
    if (query.trim() && !isLoading) onSubmit(query.trim(), topK, refineQuery, useReranker, compare)
  }
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    submit()
  }

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <div className="composer">
        <label className="sr-only" htmlFor="research-query">Ask about SIGGRAPH 2025 papers</label>
        <textarea
          ref={input}
          id="research-query"
          rows={2}
          value={query}
          onChange={event => { setQuery(event.target.value); resize(event.target) }}
          placeholder="Ask about rendering, geometry, simulation, animation…"
          disabled={isLoading}
          className="research-input"
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault()
              submit()
            }
          }}
        />
        <div className="composer-footer">
          <fieldset className="search-options" disabled={isLoading}>
            <legend className="sr-only">Search options</legend>
            <label title="Rewrite the question into a precise search query first">
              <input type="checkbox" checked={refineQuery} onChange={event => setRefineQuery(event.target.checked)} /> Refine question
            </label>
            <label title="Reorder retrieved excerpts with a reranking model">
              <input type="checkbox" checked={useReranker} onChange={event => setUseReranker(event.target.checked)} /> Rerank
            </label>
            <label title="Number of excerpts given to the model">
              Excerpts
              <select value={topK} onChange={event => setTopK(Number(event.target.value))}>
                {[5, 8, 10, 15].map(count => <option key={count} value={count}>{count}</option>)}
              </select>
            </label>
            <label title="Show rankings and answers with and without reranking">
              <input type="checkbox" checked={compare} onChange={event => setCompare(event.target.checked)} /> Compare rankings
            </label>
          </fieldset>
          <button
            type="submit"
            className="search-submit"
            aria-label={isLoading ? 'Searching papers' : 'Search papers'}
            disabled={isLoading || !query.trim()}
          >
            {isLoading ? 'Searching…' : 'Search'}
          </button>
        </div>
      </div>
    </form>
  )
}
