"use client";

interface StatusLineProps {
  message: string;
  isActive: boolean;
  currentStage?: string;
  /** Show the pipeline step trail (single-answer mode). */
  showSteps?: boolean;
  includeRefine?: boolean;
}

const STAGE_INDEX: Record<string, number> = {
  initializing: 0,
  refining: 0,
  refined: 1,
  searching: 1,
  found: 2,
  generating: 2,
  writing: 3,
  complete: 4,
};

export function StatusLine({ message, isActive, currentStage = "", showSteps, includeRefine = true }: StatusLineProps) {
  const isError = currentStage === "error";
  const steps = includeRefine ? ["Refine", "Retrieve", "Generate", "Write"] : ["Retrieve", "Generate", "Write"];
  let current = STAGE_INDEX[currentStage] ?? 0;
  if (!includeRefine) current = Math.max(0, current - 1);

  const text = message || (isError ? "Search failed. Please try again." : "Finding relevant research…");

  return (
    <div
      role={isError ? "alert" : "status"}
      aria-live="polite"
      className={`status ${isError ? "is-error" : isActive ? "is-active" : "is-idle"}`}
    >
      {showSteps && !isError && (
        <ol className="status-steps" aria-hidden="true">
          {steps.map((step, i) => (
            <li key={step} data-state={i < current ? "done" : i === current && isActive ? "current" : "todo"}>
              {step}
            </li>
          ))}
        </ol>
      )}
      {!showSteps && isActive && (
        <ol className="status-steps" aria-hidden="true"><li data-state="current">Working</li></ol>
      )}
      <p className="status-message">{text}</p>
    </div>
  );
}
