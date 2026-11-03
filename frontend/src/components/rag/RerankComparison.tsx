'use client'
import { AnswerSection } from './AnswerSection'
import type { RerankComparison as Comparison, RankedExcerpt } from '@/hooks/useRerankComparison'

export type VariantName = 'without_reranking' | 'with_reranking'

function Cell({ item, reranked }: { item?: RankedExcerpt; reranked?: boolean }) {
  if (!item) return <div className="ranking-cell" />
  const moved = reranked && item.hybrid_rank !== item.rank
  return (
    <div className="ranking-cell">
      <details>
        <summary title={item.title}>{item.title}</summary>
        <p>{item.excerpt}{item.excerpt.length === 500 ? '…' : ''}</p>
      </details>
      <div className="ranking-meta">
        <span>{item.score.toFixed(3)}</span>
        {reranked && <span className={moved ? 'ranking-moved' : ''}>{moved ? `was ${item.hybrid_rank}` : 'same'}</span>}
      </div>
    </div>
  )
}

interface RerankComparisonProps {
  result: Comparison
  selected: VariantName
  onSelect: (variant: VariantName) => void
}

export function RerankComparison({ result, selected, onSelect }: RerankComparisonProps) {
  const variant = result[selected]
  const before = result.without_reranking.ranking
  const after = result.with_reranking.ranking
  const rows = Array.from({ length: Math.max(before.length, after.length) }, (_, i) => i)
  const stats = [
    { value: `${result.changed_positions}/${result.excerpt_count}`, label: 'positions changed' },
    { value: String(result.shared_excerpts), label: 'excerpts in both' },
    { value: String(result.candidate_count), label: 'candidates' },
    { value: `${result.rerank_seconds.toFixed(2)}s`, label: 'rerank time' },
  ]

  return (
    <section className="comparison-results" aria-label="Reranking comparison">
      <details className="ranking-comparison" open>
        <summary><h2 className="eyebrow">Rankings</h2></summary>
        <dl className="comparison-stats">
          {stats.map(stat => <div key={stat.label}><dt>{stat.label}</dt><dd>{stat.value}</dd></div>)}
        </dl>
        <p className="comparison-query" title={`Model: ${result.model}`}>Search query <q>{result.search_query}</q></p>

        <div className="ranking-table" role="table" aria-label="Rankings with and without reranking">
          <div className="ranking-row ranking-head" role="row">
            <span role="columnheader" className="ranking-position">#</span>
            <div role="columnheader" className="ranking-cell"><span>Hybrid search</span><span>{result.without_reranking.retrieval_seconds.toFixed(2)}s</span></div>
            <div role="columnheader" className="ranking-cell"><span>Reranked</span><span>{result.with_reranking.retrieval_seconds.toFixed(2)}s</span></div>
          </div>
          {rows.map(i => (
            <div className="ranking-row" role="row" key={i}>
              <span role="cell" className="ranking-position">{i + 1}</span>
              <div role="cell"><Cell item={before[i]} /></div>
              <div role="cell"><Cell item={after[i]} reranked /></div>
            </div>
          ))}
        </div>
      </details>

      <div className="research-results">
        <div className="answers-heading">
          <h2 className="eyebrow">Answers</h2>
          <div className="answer-variant-tabs" role="group" aria-label="Choose answer">
            <button type="button" aria-pressed={selected === 'without_reranking'} onClick={() => onSelect('without_reranking')}>Without reranking</button>
            <button type="button" aria-pressed={selected === 'with_reranking'} onClick={() => onSelect('with_reranking')}>With reranking</button>
          </div>
        </div>
        {variant.answer_error && <p role="alert" className="variant-status is-error">{variant.answer_error}</p>}
        {variant.answer_status === 'pending' && <p className="variant-status">Preparing this answer…</p>}
        {variant.answer_status === 'streaming' && !variant.answer && <p role="status" className="variant-status">Writing answer…</p>}
        <AnswerSection
          answer={variant.answer}
          sources={variant.sources}
          processingTime={variant.answer_status === 'complete' ? variant.generation_seconds : undefined}
          isVisible={!!variant.answer}
        />
      </div>
    </section>
  )
}
