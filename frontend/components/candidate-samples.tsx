"use client";

import { useQuery } from "@tanstack/react-query";
import { Loader2, Pause, Play, Trash2, Upload } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { listSamplesOptions } from "@/api/generated/@tanstack/react-query.gen";
import type { SampleSchema } from "@/api/generated/types.gen";
import { useDeleteSample, useUploadSample } from "@/api/hooks/uploads";
import { SampleMetadataDialog } from "@/components/sample-metadata-dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { WaveformViz } from "@/components/waveform-viz";
import { BACKEND_URL, MAX_UPLOAD_SIZE_MB } from "@/lib/constants";

function CandidateCard({
  sample,
  isPlaying,
  onTogglePlay,
  onPlaybackEnd,
  onDelete,
}: {
  sample: SampleSchema;
  isPlaying: boolean;
  onTogglePlay: (sample: SampleSchema) => void;
  onPlaybackEnd: () => void;
  onDelete: (sample: SampleSchema) => void;
}) {
  return (
    <div className="group rounded-lg border bg-card p-2 text-sm">
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
            {sample.duration != null && (
              <span className="rounded bg-secondary px-1 py-0.5 text-[10px] text-muted-foreground">
                {sample.duration.toFixed(1)}s
              </span>
            )}
          </div>
        </div>

        <AlertDialog>
          <Tooltip>
            <TooltipTrigger asChild>
              <AlertDialogTrigger asChild>
                <Button
                  className="size-6 shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                  variant="ghost"
                  size="icon"
                >
                  <Trash2 size={12} className="text-muted-foreground" />
                </Button>
              </AlertDialogTrigger>
            </TooltipTrigger>
            <TooltipContent side="left">
              <p className="text-xs">Delete sample</p>
            </TooltipContent>
          </Tooltip>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete sample?</AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently remove{" "}
                <span className="font-medium">{sample.filename}</span> and any
                associated data.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={() => onDelete(sample)}>
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
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

export function CandidateSamples() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [editingSample, setEditingSample] = useState<SampleSchema | null>(null);

  const { data, isLoading } = useQuery(
    listSamplesOptions({
      query: { limit: 100, source: "upload" },
    }),
  );

  const uploadMutation = useUploadSample();
  const deleteMutation = useDeleteSample();

  const samples = data?.samples ?? [];

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      e.target.value = "";

      if (!file.name.toLowerCase().endsWith(".wav")) {
        toast.error("Only WAV files are supported");
        return;
      }

      if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
        toast.error(`File exceeds ${MAX_UPLOAD_SIZE_MB}MB limit`);
        return;
      }

      uploadMutation.mutate(
        { body: { file } },
        {
          onSuccess: (data) => {
            toast.success(`Uploaded ${data.filename}`);
            setEditingSample(data);
          },
          onError: (error) => {
            const detail = error.response?.data?.detail ?? "Upload failed";
            toast.error(typeof detail === "string" ? detail : "Upload failed");
          },
        },
      );
    },
    [uploadMutation],
  );

  const handleDelete = useCallback(
    (sample: SampleSchema) => {
      deleteMutation.mutate(
        { path: { sample_id: sample.id } },
        {
          onSuccess: () => {
            toast.success(`Deleted ${sample.filename}`);
            if (playingId === sample.id) setPlayingId(null);
          },
          onError: () => {
            toast.error("Failed to delete sample");
          },
        },
      );
    },
    [deleteMutation, playingId],
  );

  const handleTogglePlay = useCallback((sample: SampleSchema) => {
    setPlayingId((prev) => (prev === sample.id ? null : sample.id));
  }, []);

  const handlePlaybackEnd = useCallback(() => {
    setPlayingId(null);
  }, []);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Candidate Samples</h2>
          <p className="text-xs text-muted-foreground">
            {samples.length} uploaded sample{samples.length !== 1 ? "s" : ""}
          </p>
        </div>
        <div>
          <input
            accept=".wav,audio/wav"
            className="hidden"
            onChange={handleFileChange}
            ref={fileInputRef}
            type="file"
          />
          <Button
            className="h-7 gap-1.5 text-xs"
            disabled={uploadMutation.isPending}
            onClick={() => fileInputRef.current?.click()}
            size="sm"
          >
            {uploadMutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Upload className="size-3" />
            )}
            {uploadMutation.isPending ? "Processing..." : "Upload WAV"}
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-10 text-sm text-muted-foreground">
            Loading...
          </div>
        ) : samples.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
            <Upload className="size-8 opacity-50" />
            <p>No uploads yet</p>
            <p className="text-xs">
              Upload a song or snippet to find similar samples in the library
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5">
            {samples.map((sample) => (
              <CandidateCard
                isPlaying={playingId === sample.id}
                key={sample.id}
                onDelete={handleDelete}
                onPlaybackEnd={handlePlaybackEnd}
                onTogglePlay={handleTogglePlay}
                sample={sample}
              />
            ))}
          </div>
        )}
      </div>

      <SampleMetadataDialog
        sample={editingSample}
        open={editingSample !== null}
        onOpenChange={(open) => {
          if (!open) setEditingSample(null);
        }}
      />
    </div>
  );
}
