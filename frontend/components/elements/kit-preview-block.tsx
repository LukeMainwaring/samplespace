"use client";

import { Layers, Pause, Play } from "lucide-react";
import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { WaveformViz } from "@/components/waveform-viz";
import { BACKEND_URL } from "@/lib/constants";

interface KitPreviewPayload {
  audio_url: string;
  target_key?: string;
  target_bpm?: number;
}

interface KitPreviewBlockProps {
  code: string;
  isIncomplete: boolean;
}

export function KitPreviewBlock({ code, isIncomplete }: KitPreviewBlockProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const handleToggle = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  if (isIncomplete) {
    return (
      <div className="my-2 flex h-12 items-center justify-center rounded-lg border border-border bg-muted/30">
        <span className="text-muted-foreground text-xs">
          Mixing kit preview...
        </span>
      </div>
    );
  }

  let payload: KitPreviewPayload;
  try {
    payload = JSON.parse(code.trim());
  } catch {
    return (
      <div className="my-2 rounded-lg border border-border bg-muted/30 p-3 text-muted-foreground text-xs">
        Failed to parse kit preview data.
      </div>
    );
  }

  const audioUrl = payload.audio_url.startsWith("http")
    ? payload.audio_url
    : `${BACKEND_URL}${payload.audio_url}`;

  const pills: string[] = [];
  if (payload.target_key) pills.push(payload.target_key);
  if (payload.target_bpm) pills.push(`${payload.target_bpm} BPM`);

  return (
    <div className="my-3 space-y-2">
      <Separator />

      <div className="flex items-center gap-2">
        <Layers size={14} className="text-muted-foreground" />
        <span className="font-medium text-sm">Kit Preview</span>
        {pills.map((pill) => (
          <span
            className="rounded-full border border-border bg-background px-2 py-0.5 text-muted-foreground text-xs"
            key={pill}
          >
            {pill}
          </span>
        ))}
      </div>

      <div
        className={`rounded-lg border p-3 ${
          isPlaying
            ? "border-primary/50 bg-primary/5"
            : "border-border bg-muted/30"
        }`}
      >
        <div className="flex items-center gap-3">
          <Button
            className="size-8 shrink-0 rounded-full"
            onClick={handleToggle}
            type="button"
            size="icon"
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
          </Button>
          <div className="min-w-0 flex-1">
            <WaveformViz
              audioUrl={audioUrl}
              height={40}
              playing={isPlaying}
              onFinish={() => setIsPlaying(false)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
