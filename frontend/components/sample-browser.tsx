"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Pause, Play, Search } from "lucide-react";
import { useCallback, useState } from "react";
import { listSamplesOptions } from "@/api/generated/@tanstack/react-query.gen";
import type { SampleSchema, SampleType } from "@/api/generated/types.gen";
import { SampleDetailPanel } from "@/components/sample-detail-panel";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { WaveformViz } from "@/components/waveform-viz";
import { BACKEND_URL, SAMPLE_TYPES } from "@/lib/constants";
import { cn } from "@/lib/utils";

const ITEMS_PER_PAGE = 50;

function SampleCard({
  sample,
  isPlaying,
  isSelected,
  onTogglePlay,
  onPlaybackEnd,
  onSelect,
}: {
  sample: SampleSchema;
  isPlaying: boolean;
  isSelected: boolean;
  onTogglePlay: (sample: SampleSchema) => void;
  onPlaybackEnd: () => void;
  onSelect: (sample: SampleSchema) => void;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border bg-card p-2 text-sm",
        isSelected && "ring-2 ring-primary",
      )}
    >
      <div className="flex items-center gap-2">
        <Button
          className="size-8 shrink-0 rounded-full"
          onClick={() => onTogglePlay(sample)}
          size="icon"
        >
          {isPlaying ? <Pause size={12} /> : <Play size={12} />}
        </Button>

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

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                className="size-6 shrink-0"
                onClick={() => onSelect(sample)}
                size="icon"
                variant="ghost"
              >
                <Search size={12} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Show similar sounds</TooltipContent>
          </Tooltip>
        </TooltipProvider>
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
  const [activeCategory, setActiveCategory] = useState<
    "all" | "one-shot" | "loop"
  >("all");
  const [activeType, setActiveType] = useState<SampleType | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  const isLoopParam =
    activeCategory === "loop"
      ? true
      : activeCategory === "one-shot"
        ? false
        : undefined;

  const { data, isLoading } = useQuery({
    ...listSamplesOptions({
      query: {
        limit: ITEMS_PER_PAGE,
        offset: (page - 1) * ITEMS_PER_PAGE,
        sample_type: activeType ?? undefined,
        is_loop: isLoopParam,
        source: "library",
      },
    }),
    placeholderData: keepPreviousData,
  });

  const samples = data?.samples ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);

  const handleCategoryChange = useCallback(
    (cat: "all" | "one-shot" | "loop") => {
      setActiveCategory(cat);
      setPage(1);
      setPlayingId(null);
    },
    [],
  );

  const handleTypeChange = useCallback((type: SampleType | null) => {
    setActiveType(type);
    setPage(1);
    setPlayingId(null);
  }, []);

  const handleTogglePlay = useCallback((sample: SampleSchema) => {
    setPlayingId((prev) => (prev === sample.id ? null : sample.id));
  }, []);

  const handlePlaybackEnd = useCallback(() => {
    setPlayingId(null);
  }, []);

  const handleSelectSample = useCallback((sample: SampleSchema) => {
    setSelectedSampleId(sample.id);
    setPlayingId(null);
  }, []);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sample list */}
      <div
        className={cn(
          "flex h-full min-w-0 flex-col transition-[width] duration-200",
          selectedSampleId ? "w-2/5 border-r" : "w-full",
        )}
      >
        <div className="border-b px-4 py-3">
          <h2 className="text-sm font-semibold">Sample Library</h2>
          <p className="text-xs text-muted-foreground">
            {total.toLocaleString()} samples
            {activeType ? ` (${activeType})` : ""}
          </p>
        </div>

        <div className="flex gap-1 border-b px-4 py-2">
          {(["all", "one-shot", "loop"] as const).map((cat) => (
            <Button
              className="h-6 text-xs"
              key={cat}
              onClick={() => handleCategoryChange(cat)}
              size="sm"
              variant={activeCategory === cat ? "default" : "ghost"}
            >
              {cat === "all"
                ? "All"
                : cat === "one-shot"
                  ? "One-shots"
                  : "Loops"}
            </Button>
          ))}
        </div>

        <div className="flex flex-wrap gap-1 border-b px-4 py-2">
          <Button
            className="h-6 text-xs"
            onClick={() => handleTypeChange(null)}
            size="sm"
            variant={activeType === null ? "default" : "ghost"}
          >
            All
          </Button>
          {SAMPLE_TYPES.map((type) => (
            <Button
              className="h-6 text-xs"
              key={type}
              onClick={() =>
                handleTypeChange(activeType === type ? null : type)
              }
              size="sm"
              variant={activeType === type ? "default" : "ghost"}
            >
              {type}
            </Button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-10 text-muted-foreground text-sm">
              Loading samples...
            </div>
          ) : samples.length === 0 ? (
            <div className="flex items-center justify-center py-10 text-muted-foreground text-sm">
              No samples found
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {samples.map((sample) => (
                <SampleCard
                  isPlaying={playingId === sample.id}
                  isSelected={selectedSampleId === sample.id}
                  key={sample.id}
                  onPlaybackEnd={handlePlaybackEnd}
                  onSelect={handleSelectSample}
                  onTogglePlay={handleTogglePlay}
                  sample={sample}
                />
              ))}
            </div>
          )}
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t px-4 py-2">
            <Button
              className="h-7 text-xs"
              disabled={page <= 1}
              onClick={() => {
                setPage((p) => p - 1);
                setPlayingId(null);
              }}
              size="sm"
              variant="ghost"
            >
              <ChevronLeft size={14} />
              Previous
            </Button>
            <span className="text-xs text-muted-foreground">
              {page} / {totalPages}
            </span>
            <Button
              className="h-7 text-xs"
              disabled={page >= totalPages}
              onClick={() => {
                setPage((p) => p + 1);
                setPlayingId(null);
              }}
              size="sm"
              variant="ghost"
            >
              Next
              <ChevronRight size={14} />
            </Button>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedSampleId && (
        <div className="flex min-w-0 w-3/5 flex-col">
          <SampleDetailPanel
            key={selectedSampleId}
            onClose={() => setSelectedSampleId(null)}
            sampleId={selectedSampleId}
          />
        </div>
      )}
    </div>
  );
}
