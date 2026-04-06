"use client";

import { useCallback, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { SampleCard, type SamplePayload } from "./sample-card";

interface KitSlotPayload {
  position: number;
  requested_type: string;
  sample: SamplePayload;
  compatibility_score: number;
}

export interface KitPayload {
  slots: KitSlotPayload[];
  overall_score: number;
  pairwise_scores: Array<{
    slot_a: number;
    slot_b: number;
    score: number;
    summary: string;
  }>;
  vibe?: string;
  genre?: string;
  skipped_types?: string[];
}

interface KitBlockProps {
  code: string;
  isIncomplete: boolean;
}

function ScoreBadge({ score, label }: { score: number; label?: string }) {
  if (score == null) return null;

  const color =
    score >= 0.7
      ? "text-green-600 dark:text-green-400 border-green-300 dark:border-green-700"
      : score >= 0.4
        ? "text-yellow-600 dark:text-yellow-400 border-yellow-300 dark:border-yellow-700"
        : "text-red-600 dark:text-red-400 border-red-300 dark:border-red-700";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-medium text-xs ${color}`}
    >
      {label && <span className="text-muted-foreground">{label}</span>}
      {score.toFixed(2)}
    </span>
  );
}

export function KitBlock({ code, isIncomplete }: KitBlockProps) {
  const [playingId, setPlayingId] = useState<string | null>(null);
  const handleTogglePlay = useCallback((id: string) => {
    setPlayingId((prev) => (prev === id ? null : id));
  }, []);

  if (isIncomplete) {
    return (
      <div className="my-3 space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton
            className="h-20 rounded-lg border border-border bg-muted/30"
            key={`skeleton-${i}`}
          />
        ))}
      </div>
    );
  }

  let payload: KitPayload;
  try {
    payload = JSON.parse(code.trim());
    if (!payload.slots?.length) {
      throw new Error("No slots in kit");
    }
  } catch {
    return (
      <div className="my-2 rounded-lg border border-border bg-muted/30 p-3 text-muted-foreground text-xs">
        Failed to parse kit data.
      </div>
    );
  }

  const pills: string[] = [];
  if (payload.genre) pills.push(payload.genre);
  if (payload.vibe) pills.push(payload.vibe);

  return (
    <div className="my-3 space-y-3">
      <div className="flex items-center gap-2">
        <ScoreBadge score={payload.overall_score} label="Kit score" />
        {pills.map((pill, i) => (
          <span
            className="rounded-full border border-border bg-background px-2 py-0.5 text-muted-foreground text-xs"
            key={`${pill}-${i}`}
          >
            {pill}
          </span>
        ))}
      </div>

      <div className="space-y-2">
        {payload.slots.map((slot, i) => (
          <div className="flex items-start gap-2" key={slot.sample?.id ?? i}>
            <div className="flex w-14 shrink-0 flex-col items-center gap-0.5 pt-3">
              <span className="font-medium text-xs capitalize">
                {slot.requested_type}
              </span>
              {slot.compatibility_score != null && (
                <span className="text-muted-foreground text-[10px]">
                  {slot.compatibility_score.toFixed(2)}
                </span>
              )}
            </div>
            <div className="min-w-0 flex-1">
              <SampleCard
                sample={slot.sample}
                isPlaying={playingId === slot.sample.id}
                onTogglePlay={handleTogglePlay}
              />
            </div>
          </div>
        ))}
      </div>

      {payload.skipped_types && payload.skipped_types.length > 0 && (
        <p className="text-muted-foreground text-xs">
          Could not find samples for: {payload.skipped_types.join(", ")}
        </p>
      )}
    </div>
  );
}
