"use client";

import { Pause, Play } from "lucide-react";
import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { WaveformViz } from "@/components/waveform-viz";
import { BACKEND_URL } from "@/lib/constants";

interface AudioBlockProps {
  code: string;
  isIncomplete: boolean;
}

export function AudioBlock({ code, isIncomplete }: AudioBlockProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const handleToggle = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

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
    <div
      className={`my-2 rounded-lg border p-3 ${
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
  );
}
