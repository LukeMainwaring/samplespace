"use client";

import WavesurferPlayer from "@wavesurfer/react";
import { useCallback, useState } from "react";
import type WaveSurfer from "wavesurfer.js";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function getThemeColors() {
  const style = getComputedStyle(document.documentElement);
  const primary = style.getPropertyValue("--primary").trim();
  const muted = style.getPropertyValue("--muted-foreground").trim();
  return {
    progressColor: primary || "#3b82f6",
    waveColor: muted || "#a1a1aa",
  };
}

interface WaveformVizProps {
  audioUrl: string;
  height?: number;
  autoplay?: boolean;
  onFinish?: () => void;
}

export function WaveformViz({
  audioUrl,
  height = 40,
  autoplay = false,
  onFinish,
}: WaveformVizProps) {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [colors] = useState(() =>
    typeof document !== "undefined"
      ? getThemeColors()
      : { progressColor: "#3b82f6", waveColor: "#a1a1aa" },
  );

  const handleReady = useCallback(
    (ws: WaveSurfer) => {
      setDuration(ws.getDuration());
      if (autoplay) {
        ws.play();
      }
    },
    [autoplay],
  );

  const handleTimeupdate = useCallback((ws: WaveSurfer) => {
    setCurrentTime(ws.getCurrentTime());
  }, []);

  return (
    <div className="w-full">
      <WavesurferPlayer
        url={audioUrl}
        height={height}
        waveColor={colors.waveColor}
        progressColor={colors.progressColor}
        barWidth={2}
        barGap={1}
        barRadius={2}
        cursorWidth={0}
        interact={true}
        onReady={handleReady}
        onTimeupdate={handleTimeupdate}
        onFinish={onFinish}
      />
      <div className="flex justify-between text-[10px] text-muted-foreground mt-0.5">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
