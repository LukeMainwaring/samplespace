"use client";

import { FileAudio, Loader2, X } from "lucide-react";
import type { SampleSchema } from "@/api/generated/types.gen";

export type Attachment = {
  file: File;
  sample?: SampleSchema;
  isUploading: boolean;
};

export function PreviewAttachment({
  attachment,
  onRemove,
}: {
  attachment: Attachment;
  onRemove: () => void;
}) {
  return (
    <div className="relative flex items-center gap-1.5 rounded-lg border bg-muted/50 px-2 py-1.5 text-xs">
      {attachment.isUploading ? (
        <Loader2 className="size-3.5 animate-spin text-muted-foreground" />
      ) : (
        <FileAudio className="size-3.5 text-muted-foreground" />
      )}

      <span className="max-w-[200px] truncate font-medium">
        {attachment.file.name}
      </span>

      {attachment.sample && (
        <span className="text-muted-foreground">
          {[
            attachment.sample.sample_type,
            attachment.sample.key,
            attachment.sample.bpm ? `${attachment.sample.bpm} BPM` : null,
          ]
            .filter(Boolean)
            .join(" / ") || "analyzed"}
        </span>
      )}

      {attachment.isUploading ? (
        <span className="text-muted-foreground">Processing...</span>
      ) : null}

      <button
        className="ml-0.5 rounded p-0.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        onClick={onRemove}
        type="button"
      >
        <X className="size-3" />
      </button>
    </div>
  );
}
