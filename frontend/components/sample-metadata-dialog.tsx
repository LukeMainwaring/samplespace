"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import type {
  SampleSchema,
  SampleUpdateSchema,
} from "@/api/generated/types.gen";
import { useUpdateSample } from "@/api/hooks/uploads";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function SampleMetadataDialog({
  sample,
  open,
  onOpenChange,
}: {
  sample: SampleSchema | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const updateMutation = useUpdateSample();

  const [key, setKey] = useState("");
  const [bpm, setBpm] = useState("");
  const [isLoop, setIsLoop] = useState(false);

  useEffect(() => {
    if (sample) {
      setKey(sample.key ?? "");
      setBpm(sample.bpm != null && sample.bpm > 0 ? String(sample.bpm) : "");
      setIsLoop(sample.is_loop ?? false);
    }
  }, [sample]);

  const handleSave = useCallback(() => {
    if (!sample) return;

    const body: Partial<SampleUpdateSchema> = {};
    const trimmedKey = key.trim();
    if (trimmedKey !== (sample.key ?? "")) {
      body.key = trimmedKey || null;
    }
    const parsed = parseInt(bpm, 10);
    const parsedBpm = bpm && !Number.isNaN(parsed) ? parsed : null;
    const originalBpm =
      sample.bpm != null && sample.bpm > 0 ? sample.bpm : null;
    if (parsedBpm !== originalBpm) {
      body.bpm = parsedBpm;
    }
    if (isLoop !== (sample.is_loop ?? false)) {
      body.is_loop = isLoop;
    }

    if (Object.keys(body).length === 0) {
      onOpenChange(false);
      return;
    }

    updateMutation.mutate(
      {
        path: { sample_id: sample.id },
        body,
      },
      {
        onSuccess: () => {
          toast.success("Metadata updated");
          onOpenChange(false);
        },
        onError: () => {
          toast.error("Failed to update metadata");
        },
      },
    );
  }, [sample, key, bpm, isLoop, updateMutation, onOpenChange]);

  if (!sample) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-sm">Edit Sample Metadata</DialogTitle>
          <DialogDescription className="truncate text-xs">
            {sample.filename}
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 py-2">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="sample-key" className="text-xs">
                Key
              </Label>
              <Input
                id="sample-key"
                className="h-8 text-xs"
                placeholder="e.g. C major"
                value={key}
                onChange={(e) => setKey(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="sample-bpm" className="text-xs">
                BPM
              </Label>
              <Input
                id="sample-bpm"
                className="h-8 text-xs"
                type="number"
                min={1}
                max={300}
                placeholder="e.g. 120"
                value={bpm}
                onChange={(e) => setBpm(e.target.value)}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">Classification</Label>
            <div className="flex gap-1.5">
              <Button
                className="h-7 flex-1 text-xs"
                variant={isLoop ? "default" : "outline"}
                size="sm"
                onClick={() => setIsLoop(true)}
              >
                Loop
              </Button>
              <Button
                className="h-7 flex-1 text-xs"
                variant={!isLoop ? "default" : "outline"}
                size="sm"
                onClick={() => setIsLoop(false)}
              >
                One-shot
              </Button>
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={() => onOpenChange(false)}
          >
            Skip
          </Button>
          <Button
            size="sm"
            className="h-7 text-xs"
            onClick={handleSave}
            disabled={updateMutation.isPending}
          >
            {updateMutation.isPending ? "Saving..." : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
