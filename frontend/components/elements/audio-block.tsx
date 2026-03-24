"use client";

import { WaveformViz } from "@/components/waveform-viz";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002";

interface AudioBlockProps {
  code: string;
  isIncomplete: boolean;
}

export function AudioBlock({ code, isIncomplete }: AudioBlockProps) {
  if (isIncomplete) {
    return (
      <div className="my-2 flex h-12 items-center justify-center rounded-lg border border-border bg-muted/30">
        <span className="text-muted-foreground text-xs">Loading audio...</span>
      </div>
    );
  }

  const audioPath = code.trim();
  const audioUrl = audioPath.startsWith("http")
    ? audioPath
    : `${BACKEND_URL}${audioPath}`;

  return (
    <div className="my-2 rounded-lg border border-border bg-muted/30 p-3">
      <WaveformViz audioUrl={audioUrl} height={40} />
    </div>
  );
}
