"use client";

import { useCallback, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { SampleCard, type SamplePayload } from "./sample-card";

interface SampleResultPayload extends SamplePayload {
  annotation?: string;
}

export interface SampleResultsPayload {
  samples: SampleResultPayload[];
}

interface SampleResultsBlockProps {
  code: string;
  isIncomplete: boolean;
}

export function SampleResultsBlock({
  code,
  isIncomplete,
}: SampleResultsBlockProps) {
  const [playingId, setPlayingId] = useState<string | null>(null);
  const handleTogglePlay = useCallback((id: string) => {
    setPlayingId((prev) => (prev === id ? null : id));
  }, []);

  if (isIncomplete) {
    return (
      <div className="my-3 space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton
            className="h-20 rounded-lg border border-border bg-muted/30"
            key={`skeleton-${i}`}
          />
        ))}
      </div>
    );
  }

  let payload: SampleResultsPayload;
  try {
    payload = JSON.parse(code.trim());
    if (!payload.samples?.length) {
      throw new Error("No samples in results");
    }
  } catch {
    return (
      <div className="my-2 rounded-lg border border-border bg-muted/30 p-3 text-muted-foreground text-xs">
        Failed to parse sample results.
      </div>
    );
  }

  return (
    <div className="my-3 space-y-2">
      {payload.samples.map((sample) => (
        <SampleCard
          key={sample.id}
          sample={sample}
          isPlaying={playingId === sample.id}
          onTogglePlay={handleTogglePlay}
          annotation={sample.annotation}
        />
      ))}
    </div>
  );
}
