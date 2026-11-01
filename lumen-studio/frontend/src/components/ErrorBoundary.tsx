import { Component, ErrorInfo, Fragment, ReactNode } from "react";
import { RotateCcw, XCircle } from "lucide-react";

interface Props {
  children: ReactNode;
  /** What failed, e.g. "3D viewer" or "This page". */
  label?: string;
  hint?: ReactNode;
  /** Fill the parent (absolute inset-0), for viewers inside a positioned box. */
  fill?: boolean;
  onError?: (error: Error) => void;
}

interface State {
  error: Error | null;
  attempt: number;
}

/**
 * Keeps a failing part of the UI from taking the whole app down (without it React unmounts
 * everything and the window goes black). Shows an inline card; Retry remounts the children.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, attempt: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[Lumen] ${this.props.label || "A view"} failed:`, error, info.componentStack);
    try {
      this.props.onError?.(error);
    } catch {
      /* never let the error handler fail the boundary */
    }
  }

  retry = () => this.setState((s) => ({ error: null, attempt: s.attempt + 1 }));

  render() {
    const { error, attempt } = this.state;
    if (!error) return <Fragment key={attempt}>{this.props.children}</Fragment>;
    const label = this.props.label || "This part of Lumen Studio";
    const card = (
      <div className="w-full max-w-md rounded-lg border border-[#5a2a2e] bg-[#1f1012] text-[#ffb3b3] p-4 text-[12px] leading-relaxed shadow-xl" role="alert" data-testid="error-boundary">
        <div className="flex items-start gap-2.5">
          <XCircle className="w-4 h-4 shrink-0 mt-[1px]" />
          <div className="min-w-0 flex-1">
            <div className="font-semibold">{label} stopped working</div>
            <div className="mt-1 font-mono text-[11px] text-[#ffcfcf]/90 break-words">{error.message || String(error)}</div>
            <div className="mt-2 opacity-80">
              {this.props.hint ?? "The rest of Lumen Studio keeps working. Retry reloads this part; if it keeps failing, reload the page — your photos stay on this computer."}
            </div>
            <button
              type="button"
              onClick={this.retry}
              className="mt-3 inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md bg-[#2a1517] hover:bg-[#3a1b1e] border border-[#5a2a2e] text-[#ffb3b3] text-xs font-medium"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Retry
            </button>
          </div>
        </div>
      </div>
    );
    return <div className={this.props.fill ? "absolute inset-0 flex items-center justify-center p-4 bg-[#0a0c10]" : "w-full h-full flex items-center justify-center p-6"}>{card}</div>;
  }
}

export default ErrorBoundary;
