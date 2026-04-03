"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useCallback, useState } from "react";
import { useChatActions } from "@/components/chat-actions-provider";
import { Button } from "@/components/ui/button";
import { SampleCard, type SamplePayload } from "./sample-card";

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

export function PairVerdictBlock({
  code,
  isIncomplete,
}: PairVerdictBlockProps) {
  const chatActions = useChatActions();
  const [submitted, setSubmitted] = useState<"approved" | "rejected" | null>(
    null,
  );
  const [playingId, setPlayingId] = useState<string | null>(null);
  const handleTogglePlay = useCallback((id: string) => {
    setPlayingId((prev) => (prev === id ? null : id));
  }, []);

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
    if (!payload.sample_a?.id || !payload.sample_b?.id) {
      throw new Error("Missing sample data");
    }
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
        <SampleCard
          sample={payload.sample_a}
          isPlaying={playingId === payload.sample_a.id}
          onTogglePlay={handleTogglePlay}
        />
        <SampleCard
          sample={payload.sample_b}
          isPlaying={playingId === payload.sample_b.id}
          onTogglePlay={handleTogglePlay}
        />
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
            <Button
              className="border-green-300 text-green-700 text-xs hover:bg-green-50 dark:border-green-700 dark:text-green-400 dark:hover:bg-green-950/30"
              onClick={() => handleVerdict(true)}
              type="button"
              variant="outline"
              size="sm"
            >
              <ThumbsUp size={13} />
              Works
            </Button>
            <Button
              className="border-red-300 text-red-700 text-xs hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-950/30"
              onClick={() => handleVerdict(false)}
              type="button"
              variant="outline"
              size="sm"
            >
              <ThumbsDown size={13} />
              Doesn&apos;t work
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
