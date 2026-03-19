"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { listSamplesOptions } from "@/api/generated/@tanstack/react-query.gen";
import type { SampleSchema } from "@/api/generated/types.gen";
import { Button } from "@/components/ui/button";

const SAMPLE_TYPES = [
  "kick",
  "snare",
  "hihat",
  "clap",
  "percussion",
  "bass",
  "pad",
  "lead",
  "keys",
  "fx",
  "vocal",
] as const;

function SampleCard({ sample }: { sample: SampleSchema }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border bg-card p-3 text-sm">
      <div className="truncate font-medium" title={sample.filename}>
        {sample.filename}
      </div>
      <div className="flex flex-wrap gap-1.5 text-xs text-muted-foreground">
        {sample.sample_type && (
          <span className="rounded bg-secondary px-1.5 py-0.5">
            {sample.sample_type}
          </span>
        )}
        {sample.key && (
          <span className="rounded bg-secondary px-1.5 py-0.5">
            {sample.key}
          </span>
        )}
        {sample.bpm != null && sample.bpm > 0 && (
          <span className="rounded bg-secondary px-1.5 py-0.5">
            {sample.bpm} BPM
          </span>
        )}
        {sample.duration != null && (
          <span className="rounded bg-secondary px-1.5 py-0.5">
            {sample.duration.toFixed(1)}s
          </span>
        )}
      </div>
      <div className="mt-1 truncate text-[10px] text-muted-foreground/60">
        {sample.id}
      </div>
    </div>
  );
}

export function SampleBrowser() {
  const [activeType, setActiveType] = useState<string | null>(null);

  const { data, isLoading } = useQuery(
    listSamplesOptions({
      query: { limit: 100 },
    }),
  );

  const samples = data?.samples ?? [];
  const filteredSamples = activeType
    ? samples.filter((s) => s.sample_type === activeType)
    : samples;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b px-4 py-3">
        <h2 className="text-sm font-semibold">Sample Library</h2>
        <p className="text-xs text-muted-foreground">
          {filteredSamples.length} samples
          {activeType ? ` (${activeType})` : ""}
        </p>
      </div>

      {/* Type filters */}
      <div className="flex flex-wrap gap-1 border-b px-4 py-2">
        <Button
          className="h-6 text-xs"
          onClick={() => setActiveType(null)}
          size="sm"
          variant={activeType === null ? "default" : "ghost"}
        >
          All
        </Button>
        {SAMPLE_TYPES.map((type) => (
          <Button
            className="h-6 text-xs"
            key={type}
            onClick={() => setActiveType(activeType === type ? null : type)}
            size="sm"
            variant={activeType === type ? "default" : "ghost"}
          >
            {type}
          </Button>
        ))}
      </div>

      {/* Sample grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground text-sm">
            Loading samples...
          </div>
        ) : filteredSamples.length === 0 ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground text-sm">
            No samples found
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-2">
            {filteredSamples.map((sample) => (
              <SampleCard key={sample.id} sample={sample} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
