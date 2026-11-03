"use client";

import { useState } from "react";
import { QueryForm } from "@/components/rag/QueryForm";
import { AnswerSection } from "@/components/rag/AnswerSection";
import { ReferencesSection } from "@/components/rag/ReferencesSection";
import { StatusLine } from "@/components/rag/StatusLine";
import { RerankComparison, type VariantName } from "@/components/rag/RerankComparison";
import { useRerankComparison } from "@/hooks/useRerankComparison";
import { useRAGStream } from "@/hooks/useRAGStream";

const EXAMPLES = [
  "How does 3D Gaussian splatting work?",
  "What’s new in neural rendering this year?",
  "Which papers improve physically based cloth simulation?",
];

function Pipeline() {
  return (
    <section className="pipeline" aria-label="How answers are produced">
      <span className="eyebrow">How it works</span>
      <ol>
        <li><strong>Refine</strong><span>Your question is rewritten into a precise search query.</span></li>
        <li><strong>Retrieve</strong><span>Hybrid semantic and keyword search over SIGGRAPH 2025 paper excerpts.</span></li>
        <li><strong>Rerank</strong><span>A reranking model reorders the candidates by relevance.</span></li>
        <li><strong>Answer</strong><span>The model writes an answer and cites the papers it used.</span></li>
      </ol>
    </section>
  );
}

export default function Home() {
  const rag = useRAGStream();
  const comparison = useRerankComparison();
  const [isComparison, setIsComparison] = useState(false);
  const [refineUsed, setRefineUsed] = useState(true);
  const [variant, setVariant] = useState<VariantName>("with_reranking");
  const [question, setQuestion] = useState("");
  const [suggestion, setSuggestion] = useState({ text: "", id: 0 });
  const [formKey, setFormKey] = useState(0);

  const busy = rag.isLoading || comparison.isLoading;
  const hasResults = !!question;

  const newSearch = () => {
    rag.reset();
    comparison.reset();
    setIsComparison(false);
    setQuestion("");
    setSuggestion({ text: "", id: 0 });
    setFormKey((k) => k + 1);
  };

  const sidebarSources = isComparison
    ? comparison.result?.[variant].sources ?? []
    : rag.sources;
  const sidebarAnswer = isComparison ? comparison.result?.[variant].answer ?? "" : rag.answer;
  const showRagStatus = rag.isActive || (!!rag.statusMessage && rag.currentStage !== "complete");

  return (
    <div className="research-app">
      <header className="site-header">
        <div className="shell">
          <a href="/" className="brand" aria-label="SIGGRAPH 2025 research companion, home">
            <span className="brand-mark">SIGGRAPH <em>2025</em></span>
          </a>
          {hasResults && (
            <button type="button" className="header-action" onClick={newSearch} disabled={busy}>
              New search
            </button>
          )}
        </div>
      </header>

      <main id="main-content" className={`workspace shell ${hasResults ? "with-results" : ""}`}>
        <div className="layout">
          <div className="main-col">
            {!hasResults && (
              <header className="intro">
                <h1>Explore the research.</h1>
              </header>
            )}

            <QueryForm
              key={formKey}
              suggestion={suggestion}
              isLoading={busy}
              compact={hasResults}
              onSubmit={(query, topK, refine, rerank, compare) => {
                setQuestion(query);
                setIsComparison(compare);
                setRefineUsed(refine);
                setVariant(rerank ? "with_reranking" : "without_reranking");
                rag.reset();
                comparison.reset();
                if (compare) comparison.run(query, topK, refine, rerank);
                else rag.streamQuery(query, topK, refine, rerank);
              }}
            />

            {!hasResults && (
              <nav className="example-queries" aria-label="Example questions">
                <span className="eyebrow">Try asking</span>
                <ul>
                  {EXAMPLES.map((text) => (
                    <li key={text}>
                      <button type="button" onClick={() => setSuggestion((p) => ({ text, id: p.id + 1 }))}>
                        {text}
                      </button>
                    </li>
                  ))}
                </ul>
              </nav>
            )}

            {hasResults && !isComparison && (
              <div className="research-results" aria-busy={rag.isLoading}>
                {showRagStatus && (
                  <StatusLine
                    message={rag.statusMessage}
                    isActive={rag.isActive}
                    currentStage={rag.currentStage}
                    showSteps
                    includeRefine={refineUsed}
                  />
                )}
                <AnswerSection
                  sources={rag.sources}
                  answer={rag.answer}
                  processingTime={rag.processingTime ?? undefined}
                  isVisible={!!rag.answer}
                />
              </div>
            )}

            {isComparison && (
              <div className="research-results" aria-busy={comparison.isLoading}>
                {comparison.isLoading && (
                  <StatusLine message={comparison.message || "Preparing comparison…"} isActive />
                )}
                {comparison.error && (
                  <StatusLine message={comparison.error} isActive={false} currentStage="error" />
                )}
              </div>
            )}
            {isComparison && comparison.result && (
              <RerankComparison result={comparison.result} selected={variant} onSelect={setVariant} />
            )}
          </div>

          <div className={`side-col ${hasResults ? "" : "landing-only"}`}>
            {!hasResults && <Pipeline />}
            {hasResults && (
              <ReferencesSection answer={sidebarAnswer} allSources={sidebarSources} isVisible={sidebarSources.length > 0} />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
