"use client";

import { useQuery } from "@tanstack/react-query";
import { Pause, Play } from "lucide-react";
import { useCallback, useState } from "react";
import { listSamplesOptions } from "@/api/generated/@tanstack/react-query.gen";
import type { SampleSchema } from "@/api/generated/types.gen";
import { Button } from "@/components/ui/button";
import { WaveformViz } from "@/components/waveform-viz";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002";

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

function SampleCard({
  sample,
  isPlaying,
  onTogglePlay,
  onPlaybackEnd,
}: {
  sample: SampleSchema;
  isPlaying: boolean;
  onTogglePlay: (sample: SampleSchema) => void;
  onPlaybackEnd: () => void;
}) {
  return (
    <div className="rounded-lg border bg-card p-2 text-sm">
      <div className="flex items-center gap-2">
        <button
          className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary/90"
          onClick={() => onTogglePlay(sample)}
          type="button"
        >
          {isPlaying ? <Pause size={12} /> : <Play size={12} />}
        </button>

        <div className="min-w-0 flex-1">
          <div className="truncate font-medium text-xs" title={sample.filename}>
            {sample.filename}
          </div>
          <div className="flex flex-wrap gap-1 mt-0.5">
            {sample.sample_type && (
              <span className="rounded bg-secondary px-1 py-0.5 text-[10px] text-muted-foreground">
                {sample.sample_type}
              </span>
            )}
            {sample.key && (
              <span className="rounded bg-secondary px-1 py-0.5 text-[10px] text-muted-foreground">
                {sample.key}
              </span>
            )}
            {sample.bpm != null && sample.bpm > 0 && (
              <span className="rounded bg-secondary px-1 py-0.5 text-[10px] text-muted-foreground">
                {sample.bpm} BPM
              </span>
            )}
          </div>
        </div>
      </div>

      {isPlaying && (
        <div className="mt-2">
          <WaveformViz
            audioUrl={`${BACKEND_URL}/api/samples/${sample.id}/audio`}
            height={40}
            autoplay
            onFinish={onPlaybackEnd}
          />
        </div>
      )}
    </div>
  );
}

export function SampleBrowser() {
  const [activeType, setActiveType] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);

  const { data, isLoading } = useQuery(
    listSamplesOptions({
      query: { limit: 100 },
    }),
  );

  const samples = data?.samples ?? [];
  const filteredSamples = activeType
    ? samples.filter((s) => s.sample_type === activeType)
    : samples;

  const handleTogglePlay = useCallback((sample: SampleSchema) => {
    setPlayingId((prev) => (prev === sample.id ? null : sample.id));
  }, []);

  const handlePlaybackEnd = useCallback(() => {
    setPlayingId(null);
  }, []);

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

      {/* Sample list */}
      <div className="flex-1 overflow-y-auto p-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground text-sm">
            Loading samples...
          </div>
        ) : filteredSamples.length === 0 ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground text-sm">
            No samples found
          </div>
        ) : (
          <div className="flex flex-col gap-1.5">
            {filteredSamples.map((sample) => (
              <SampleCard
                isPlaying={playingId === sample.id}
                key={sample.id}
                onPlaybackEnd={handlePlaybackEnd}
                onTogglePlay={handleTogglePlay}
                sample={sample}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
