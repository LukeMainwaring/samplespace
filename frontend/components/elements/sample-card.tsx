"use client";

import { Pause, Play } from "lucide-react";
import { WaveformViz } from "@/components/waveform-viz";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002";

export interface SamplePayload {
  id: string;
  filename: string;
  audio_url: string;
  type?: string;
  key?: string;
  bpm?: number;
}

interface SampleCardProps {
  sample: SamplePayload;
  isPlaying?: boolean;
  onTogglePlay?: (id: string) => void;
}

export function SampleCard({
  sample,
  isPlaying,
  onTogglePlay,
}: SampleCardProps) {
  const rawUrl = sample.audio_url || `/api/samples/${sample.id}/audio`;
  const audioUrl = rawUrl.startsWith("http")
    ? rawUrl
    : `${BACKEND_URL}${rawUrl}`;

  const pills: string[] = [];
  if (sample.type) pills.push(sample.type);
  if (sample.key) pills.push(sample.key);
  if (sample.bpm) pills.push(`${sample.bpm} BPM`);

  return (
    <div
      className={`flex min-w-0 flex-1 flex-col gap-2 rounded-lg border p-3 ${
        isPlaying
          ? "border-primary/50 bg-primary/5"
          : "border-border bg-muted/30"
      }`}
    >
      <div className="flex items-center gap-2">
        {onTogglePlay && (
          <button
            className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary/90"
            onClick={() => onTogglePlay(sample.id)}
            type="button"
          >
            {isPlaying ? <Pause size={11} /> : <Play size={11} />}
          </button>
        )}
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
      <WaveformViz
        audioUrl={audioUrl}
        height={36}
        playing={isPlaying}
        onPlay={() => onTogglePlay?.(sample.id)}
        onFinish={() => {
          if (isPlaying) onTogglePlay?.(sample.id);
        }}
      />
    </div>
  );
}
