"use client";

import { Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { WaveformViz } from "@/components/waveform-viz";
import { BACKEND_URL } from "@/lib/constants";

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
  annotation?: string;
}

export function SampleCard({
  sample,
  isPlaying,
  onTogglePlay,
  annotation,
}: SampleCardProps) {
  const rawUrl = sample.audio_url || `/api/samples/${sample.id}/audio`;
  const audioUrl = rawUrl.startsWith("http")
    ? rawUrl
    : `${BACKEND_URL}${rawUrl}`;

  const pills: string[] = [];
  if (sample.type) pills.push(sample.type);
  if (sample.key) pills.push(sample.key);
  if (sample.bpm) pills.push(`${sample.bpm} BPM`);
  if (annotation) pills.push(annotation);

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
          <Button
            className="size-7 shrink-0 rounded-full"
            onClick={() => onTogglePlay(sample.id)}
            type="button"
            size="icon"
          >
            {isPlaying ? <Pause size={11} /> : <Play size={11} />}
          </Button>
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
        onFinish={() => {
          if (isPlaying) onTogglePlay?.(sample.id);
        }}
      />
    </div>
  );
}
