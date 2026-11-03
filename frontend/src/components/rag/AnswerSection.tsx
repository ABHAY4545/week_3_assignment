'use client'

import { useEffect, useRef } from 'react'
import { marked } from 'marked'
import katex from 'katex'
import { uniqueSources, type Source } from './ReferencesSection'
import hljs from 'highlight.js/lib/core'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import bash from 'highlight.js/lib/languages/bash'
import 'highlight.js/styles/github-dark.min.css'
import 'katex/dist/katex.min.css'

// Register languages
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('bash', bash)

interface AnswerSectionProps {
  answer: string
  sources: Source[]
  processingTime?: number
  isVisible: boolean
}

type Math = { tex: string; display: boolean }

/** Pull LaTeX out before markdown runs (it would mangle _, ^ and \), leaving placeholders. */
function extractMath(text: string): { text: string; math: Math[] } {
  const math: Math[] = []
  const stash = (tex: string, display: boolean) => {
    math.push({ tex: tex.trim(), display })
    return `@@MATH${math.length - 1}@@`
  }
  // Even indexes are prose; odd indexes are code, which must stay untouched.
  const parts = text.split(/(```[\s\S]*?```|`[^`\n]*`)/g)
  const out = parts
    .map((part, i) => {
      if (i % 2 === 1) return part
      return part
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash(t, true))
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => stash(t, true))
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => stash(t, false))
        // $x$ but not currency like "$5 and $10": no space inside the dollar signs
        .replace(/(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g, (_, t) => stash(t, false))
    })
    .join('')
  return { text: out, math }
}

function renderMath(html: string, math: Math[]) {
  return html.replace(/@@MATH(\d+)@@/g, (whole, n) => {
    const item = math[Number(n)]
    if (!item) return whole
    try {
      return katex.renderToString(item.tex, { displayMode: item.display, throwOnError: false })
    } catch {
      return item.tex
    }
  })
}

const normalize = (title: string) => title.trim().toLowerCase().replace(/\s+/g, ' ')

/** Resolve a bracket label like "Paper Title", "1", "Source 1, Source 3" to source-list indexes. */
function resolveLabel(label: string, papers: Source[]): number[] | null {
  const byToken = (token: string): number | null => {
    const t = normalize(token)
    const byTitle = papers.findIndex(paper => normalize(paper.title) === t)
    if (byTitle >= 0) return byTitle
    const match = t.match(/^(sources?|excerpts?|papers?)?\s*(\d+)$/)
    if (!match) return null
    const n = Number(match[2])
    if (match[1]) {
      // "Source 3" is the model's excerpt number; each paper lists the excerpts it supplied.
      const index = papers.findIndex(paper => paper.excerpts?.includes(n))
      return index >= 0 ? index : null
    }
    return n >= 1 && n <= papers.length ? n - 1 : null
  }
  for (const separator of [/\s*;\s*/, /\s*(?:[;,&]|\band\b)\s*/]) {
    const indexes = label.split(separator).map(byToken)
    if (indexes.every(index => index !== null)) return indexes as number[]
  }
  return null
}

export function AnswerSection({ answer, sources, processingTime, isVisible }: AnswerSectionProps) {
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    if (contentRef.current && answer) {
      const renderMarkdown = async () => {
        try {
          const { text, math } = extractMath(answer)
          const html = renderMath((await marked.parse(text)) as string, math)
          if (contentRef.current && !cancelled) {
            contentRef.current.innerHTML = html
            const papers = uniqueSources(sources)
            const walker = document.createTreeWalker(contentRef.current, NodeFilter.SHOW_TEXT)
            const nodes: Text[] = []
            while (walker.nextNode()) {
              const node = walker.currentNode as Text
              if (!node.parentElement?.closest('a, code, pre, .katex')) nodes.push(node)
            }
            for (const node of nodes) {
              const text = node.textContent || ''
              const fragment = document.createDocumentFragment()
              let position = 0
              let matched = false
              for (const match of text.matchAll(/\[([^\]]+)\]/g)) {
                const indices = resolveLabel(match[1], papers)
                if (!indices) continue
                fragment.append(text.slice(position, match.index))
                for (const index of Array.from(new Set(indices))) {
                  const link = document.createElement('a')
                  link.href = `#source-${index + 1}`
                  link.className = 'inline-citation'
                  link.textContent = `[${index + 1}]`
                  link.title = papers[index].title
                  link.setAttribute('aria-label', `Source ${index + 1}: ${papers[index].title}`)
                  link.addEventListener('click', () => {
                    const sidebar = document.querySelector('.sources-sidebar')
                    const toggle = sidebar?.querySelector<HTMLButtonElement>('.sources-toggle')
                    if (toggle && getComputedStyle(toggle).display !== 'none' && toggle.getAttribute('aria-expanded') === 'false') toggle.click()
                    requestAnimationFrame(() => document.getElementById(`source-${index + 1}`)?.focus())
                  })
                  fragment.append(link)
                }
                position = match.index! + match[0].length
                matched = true
              }
              if (matched) { fragment.append(text.slice(position)); node.replaceWith(fragment) }
            }

            // Highlight code blocks after a brief delay to ensure DOM is updated
            setTimeout(() => {
              if (contentRef.current) {
                contentRef.current.querySelectorAll('pre code').forEach((block) => {
                  hljs.highlightElement(block as HTMLElement)
                })
              }
            }, 0)
          }
        } catch (error) {
          console.error('Error parsing markdown:', error)
          if (contentRef.current && !cancelled) {
            contentRef.current.textContent = answer
          }
        }
      }

      renderMarkdown()
    }
    return () => { cancelled = true }
  }, [answer, sources])

  if (!isVisible) return null

  return (
    <section className="answer-panel" aria-label="Research answer">
      <div className="result-heading"><h2 className="eyebrow">Answer</h2>{processingTime !== undefined && <span>{processingTime.toFixed(1)}s</span>}</div>
      <div ref={contentRef} className="prose" />
    </section>
  )
}
