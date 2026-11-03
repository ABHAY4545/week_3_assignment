'use client'

import { useEffect, useRef } from 'react'
import { marked } from 'marked'
import { uniqueSources, type Source } from './ReferencesSection'
import hljs from 'highlight.js/lib/core'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import bash from 'highlight.js/lib/languages/bash'
import 'highlight.js/styles/github-dark.min.css'

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

export function AnswerSection({ answer, sources, processingTime, isVisible }: AnswerSectionProps) {
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    if (contentRef.current && answer) {
      // Render markdown (marked.parse can be sync or async depending on version)
      const renderMarkdown = async () => {
        try {
          const html = await marked.parse(answer)
          if (contentRef.current && !cancelled) {
            contentRef.current.innerHTML = html as string
            const papers = uniqueSources(sources)
            const normalize = (title: string) => title.trim().toLowerCase().replace(/\s+/g, ' ')
            const walker = document.createTreeWalker(contentRef.current, NodeFilter.SHOW_TEXT)
            const nodes: Text[] = []
            while (walker.nextNode()) {
              const node = walker.currentNode as Text
              if (!node.parentElement?.closest('a, code, pre')) nodes.push(node)
            }
            for (const node of nodes) {
              const text = node.textContent || ''
              const fragment = document.createDocumentFragment()
              let position = 0
              let matched = false
              for (const match of text.matchAll(/\[([^\]]+)\]/g)) {
                const labels = match[1].split(';').map(normalize)
                const indices = labels.map(label => papers.findIndex((paper, i) => normalize(paper.title) === label || String(i + 1) === label))
                if (indices.some(index => index < 0)) continue
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
