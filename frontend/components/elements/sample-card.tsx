"use client";

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

export function SampleCard({ sample }: { sample: SamplePayload }) {
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
