"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { useChatActions } from "@/components/chat-actions-provider";
import { WaveformViz } from "@/components/waveform-viz";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002";

interface SamplePayload {
  id: string;
  filename: string;
  audio_url: string;
  type?: string;
  key?: string;
  bpm?: number;
}

interface PairVerdictPayload {
  sample_a: SamplePayload;
  sample_b: SamplePayload;
  pair_score: number;
  summary: string;
}

interface PairVerdictBlockProps {
  code: string;
  isIncomplete: boolean;
}

function SampleCard({ sample }: { sample: SamplePayload }) {
  const audioUrl = sample.audio_url.startsWith("http")
    ? sample.audio_url
    : `${BACKEND_URL}${sample.audio_url}`;

  const pills: string[] = [];
  if (sample.type) pills.push(sample.type);
  if (sample.key) pills.push(sample.key);
  if (sample.bpm) pills.push(`${sample.bpm} BPM`);

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-2 rounded-lg border border-border bg-muted/30 p-3">
      <div className="flex items-center gap-2">
        <span className="truncate font-medium text-sm">{sample.filename}</span>
      </div>
      {pills.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {pills.map((pill) => (
            <span
              className="rounded-full border border-border bg-background px-1.5 py-0.5 text-muted-foreground text-xs"
              key={pill}
            >
              {pill}
            </span>
          ))}
        </div>
      )}
      <WaveformViz audioUrl={audioUrl} height={36} />
    </div>
  );
}

export function PairVerdictBlock({
  code,
  isIncomplete,
}: PairVerdictBlockProps) {
  const chatActions = useChatActions();
  const [submitted, setSubmitted] = useState<"approved" | "rejected" | null>(
    null,
  );

  if (isIncomplete) {
    return (
      <div className="my-2 flex h-24 items-center justify-center rounded-lg border border-border bg-muted/30">
        <span className="text-muted-foreground text-xs">
          Loading pair evaluation...
        </span>
      </div>
    );
  }

  let payload: PairVerdictPayload;
  try {
    payload = JSON.parse(code.trim());
  } catch {
    return (
      <div className="my-2 rounded-lg border border-border bg-muted/30 p-3 text-muted-foreground text-xs">
        Failed to parse pair data.
      </div>
    );
  }

  const handleVerdict = (approved: boolean) => {
    if (submitted || !chatActions) return;
    const status = approved ? "approved" : "rejected";
    setSubmitted(status);
    chatActions.sendMessage(
      `[PAIR_VERDICT] ${status}: ${payload.sample_a.id} + ${payload.sample_b.id}`,
    );
  };

  return (
    <div className="my-3 space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row">
        <SampleCard sample={payload.sample_a} />
        <SampleCard sample={payload.sample_b} />
      </div>

      <div className="flex items-center justify-between px-1">
        <span className="text-muted-foreground text-xs">
          Compatibility: {payload.pair_score}/1.0
        </span>

        {submitted ? (
          <span
            className={`text-xs font-medium ${
              submitted === "approved"
                ? "text-green-600 dark:text-green-500"
                : "text-red-600 dark:text-red-500"
            }`}
          >
            {submitted === "approved" ? "Approved" : "Rejected"}
          </span>
        ) : (
          <div className="flex gap-2">
            <button
              className="flex items-center gap-1 rounded-md border border-green-300 px-2.5 py-1 text-green-700 text-xs transition-colors hover:bg-green-50 dark:border-green-700 dark:text-green-400 dark:hover:bg-green-950/30"
              onClick={() => handleVerdict(true)}
              type="button"
            >
              <ThumbsUp size={13} />
              Works
            </button>
            <button
              className="flex items-center gap-1 rounded-md border border-red-300 px-2.5 py-1 text-red-700 text-xs transition-colors hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-950/30"
              onClick={() => handleVerdict(false)}
              type="button"
            >
              <ThumbsDown size={13} />
              Doesn&apos;t work
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
