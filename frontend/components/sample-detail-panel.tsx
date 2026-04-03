"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Pause, Play } from "lucide-react";
import Image from "next/image";
import { useCallback, useState } from "react";
import {
  getSampleOptions,
  getSimilarSamplesOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import type { SimilarSampleSchema } from "@/api/generated/types.gen";
import { Button } from "@/components/ui/button";
import { WaveformViz } from "@/components/waveform-viz";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002";

interface SampleDetailPanelProps {
  sampleId: string;
  onClose: () => void;
}

function MetadataItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </dt>
      <dd className="text-sm font-medium">{value}</dd>
    </div>
  );
}

function SimilarSampleRow({
  item,
  isPlaying,
  onTogglePlay,
  onPlaybackEnd,
}: {
  item: SimilarSampleSchema;
  isPlaying: boolean;
  onTogglePlay: (id: string) => void;
  onPlaybackEnd: () => void;
}) {
  const similarity = Math.max(0, Math.round((1 - item.distance) * 100));
  const { sample } = item;

  return (
    <div className="rounded-lg border border-border bg-muted/30 p-2 text-sm">
      <div className="flex w-full items-center gap-2">
        <Button
          className="size-7 shrink-0 rounded-full"
          onClick={() => onTogglePlay(sample.id)}
          size="icon"
        >
          {isPlaying ? <Pause size={11} /> : <Play size={11} />}
        </Button>

        <div className="min-w-0 flex-1">
          <div className="truncate font-medium text-xs" title={sample.filename}>
            {sample.filename}
          </div>
          <div className="mt-0.5 flex flex-wrap gap-1">
            {sample.sample_type && (
              <span className="rounded bg-secondary px-1 py-0.5 text-[10px] text-muted-foreground">
                {sample.sample_type}
              </span>
            )}
            <span className="rounded bg-secondary px-1 py-0.5 text-[10px] text-muted-foreground">
              {sample.is_loop ? "loop" : "one-shot"}
            </span>
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

        <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
          {similarity}%
        </span>
      </div>

      {isPlaying && (
        <div className="mt-2">
          <WaveformViz
            audioUrl={`${BACKEND_URL}/api/samples/${sample.id}/audio`}
            autoplay
            height={36}
            onFinish={onPlaybackEnd}
          />
        </div>
      )}
    </div>
  );
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function SampleDetailPanel({
  sampleId,
  onClose,
}: SampleDetailPanelProps) {
  const [spectrogramMode, setSpectrogramMode] = useState<"full" | "cnn">(
    "full",
  );
  const [spectrogramLoaded, setSpectrogramLoaded] = useState(false);
  const [spectrogramError, setSpectrogramError] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);

  const { data: sample, isLoading: sampleLoading } = useQuery(
    getSampleOptions({ path: { sample_id: sampleId } }),
  );

  const { data: similarSamples, isLoading: similarLoading } = useQuery(
    getSimilarSamplesOptions({
      path: { sample_id: sampleId },
      query: { limit: 8 },
    }),
  );

  const handleSpectrogramModeChange = useCallback(
    (mode: "full" | "cnn") => {
      if (mode !== spectrogramMode) {
        setSpectrogramLoaded(false);
        setSpectrogramError(false);
        setSpectrogramMode(mode);
      }
    },
    [spectrogramMode],
  );

  const handleTogglePlay = useCallback((id: string) => {
    setPlayingId((prev) => (prev === id ? null : id));
  }, []);

  const handlePlaybackEnd = useCallback(() => {
    setPlayingId(null);
  }, []);

  if (sampleLoading || !sample) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Loading...
      </div>
    );
  }

  const isPlaying = playingId === sampleId;
  const audioUrl = `${BACKEND_URL}/api/samples/${sampleId}/audio`;
  const spectrogramUrl = `${BACKEND_URL}/api/samples/${sampleId}/spectrogram?mode=${spectrogramMode}`;

  return (
    <div className="flex h-full min-w-0 flex-col overflow-y-auto overflow-x-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 border-b px-4 py-3">
        <Button
          className="size-7 shrink-0"
          onClick={onClose}
          size="icon"
          variant="ghost"
        >
          <ArrowLeft size={16} />
        </Button>
        <h2 className="truncate text-sm font-semibold" title={sample.filename}>
          {sample.filename}
        </h2>
      </div>

      <div className="flex min-w-0 w-full flex-col gap-5 p-4">
        {/* Waveform */}
        <div>
          <div className="mb-2 flex items-center gap-2">
            <Button
              className="size-8 shrink-0 rounded-full"
              onClick={() => handleTogglePlay(sampleId)}
              size="icon"
            >
              {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            </Button>
            <span className="text-xs text-muted-foreground">
              {sample.duration ? formatDuration(sample.duration) : "—"}
            </span>
          </div>
          <WaveformViz
            audioUrl={audioUrl}
            height={64}
            onFinish={handlePlaybackEnd}
            playing={isPlaying}
          />
        </div>

        {/* Metadata */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          {sample.sample_type && (
            <MetadataItem label="Type" value={sample.sample_type} />
          )}
          <MetadataItem
            label="Category"
            value={sample.is_loop ? "Loop" : "One-shot"}
          />
          {sample.key && <MetadataItem label="Key" value={sample.key} />}
          {sample.bpm != null && sample.bpm > 0 && (
            <MetadataItem label="BPM" value={String(sample.bpm)} />
          )}
          {sample.duration != null && (
            <MetadataItem
              label="Duration"
              value={`${sample.duration.toFixed(2)}s`}
            />
          )}
          {sample.pack_name && (
            <MetadataItem label="Pack" value={sample.pack_name} />
          )}
        </div>

        {/* Spectrogram */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Spectrogram
            </h3>
            <div className="flex gap-1">
              <Button
                className="h-5 text-[10px]"
                onClick={() => handleSpectrogramModeChange("full")}
                size="sm"
                variant={spectrogramMode === "full" ? "default" : "ghost"}
              >
                Full
              </Button>
              <Button
                className="h-5 text-[10px]"
                onClick={() => handleSpectrogramModeChange("cnn")}
                size="sm"
                variant={spectrogramMode === "cnn" ? "default" : "ghost"}
              >
                CNN View
              </Button>
            </div>
          </div>
          <div className="relative overflow-hidden rounded-md border bg-black/20">
            {!spectrogramLoaded && !spectrogramError && (
              <div className="flex h-[100px] items-center justify-center text-xs text-muted-foreground">
                Generating spectrogram...
              </div>
            )}
            {spectrogramError && (
              <div className="flex h-[100px] items-center justify-center text-xs text-muted-foreground">
                Failed to load spectrogram
              </div>
            )}
            <Image
              alt={`${spectrogramMode === "cnn" ? "CNN input" : "Full"} mel spectrogram`}
              className={spectrogramLoaded ? "block w-full h-auto" : "hidden"}
              height={0}
              key={spectrogramUrl}
              onError={() => setSpectrogramError(true)}
              onLoad={() => setSpectrogramLoaded(true)}
              src={spectrogramUrl}
              style={{ width: "100%", height: "auto" }}
              unoptimized
              width={0}
            />
          </div>
          {spectrogramMode === "cnn" && (
            <p className="mt-1 text-[10px] text-muted-foreground">
              128 mel bins, 2s fixed — what the CNN model sees during inference
            </p>
          )}
        </div>

        {/* Similar Samples */}
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Similar Samples
          </h3>
          {similarLoading ? (
            <div className="py-4 text-center text-xs text-muted-foreground">
              Finding similar samples...
            </div>
          ) : !similarSamples || similarSamples.length === 0 ? (
            <div className="py-4 text-center text-xs text-muted-foreground">
              No similar samples found
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {similarSamples.map((item) => (
                <SimilarSampleRow
                  isPlaying={playingId === item.sample.id}
                  item={item}
                  key={item.sample.id}
                  onPlaybackEnd={handlePlaybackEnd}
                  onTogglePlay={handleTogglePlay}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
