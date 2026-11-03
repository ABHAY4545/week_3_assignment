'use client'
import { useState } from 'react'

export interface Source { title: string; authors?: string; pdf_url?: string; github_link?: string; video_link?: string; acm_url?: string }

export function uniqueSources(sources: Source[]) {
  return sources.filter((source, index, list) => list.findIndex(s => s.title === source.title) === index)
}

interface ReferencesSectionProps { answer: string; allSources: Source[]; isVisible: boolean }

export function ReferencesSection({ allSources, isVisible }: ReferencesSectionProps) {
  const [open, setOpen] = useState(false)
  if (!isVisible || !allSources.length) return null
  const sources = uniqueSources(allSources)

  return (
    <aside className={`sources-sidebar ${open ? 'sources-open' : ''}`} aria-label="Source papers">
      <button
        type="button"
        className="sources-toggle"
        aria-expanded={open}
        aria-controls="source-list"
        onClick={() => setOpen(!open)}
      >
        {open ? 'Close sources' : `Sources (${sources.length})`}
      </button>
      <div id="source-list" className="references-panel">
        <div className="result-heading">
          <h2 className="eyebrow">Sources</h2>
          <span>{sources.length} {sources.length === 1 ? 'paper' : 'papers'}</span>
        </div>
        {sources.map((source, index) => {
          const links = [
            { url: source.pdf_url, label: 'Paper' },
            { url: source.github_link, label: 'Code' },
            { url: source.video_link, label: 'Video' },
            { url: source.acm_url, label: 'ACM' },
          ].filter(link => link.url)
          return (
            <article id={`source-${index + 1}`} tabIndex={-1} className="reference-item" key={source.title}>
              <span className="reference-number">{index + 1}</span>
              <div>
                <h3>{source.title}</h3>
                {source.authors && <p title={source.authors}>{source.authors}</p>}
                {links.length > 0 && (
                  <div className="reference-links">
                    {links.map(link => (
                      <a key={link.label} href={link.url} target="_blank" rel="noopener noreferrer">{link.label}</a>
                    ))}
                  </div>
                )}
              </div>
            </article>
          )
        })}
      </div>
    </aside>
  )
}
