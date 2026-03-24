"use client";

import type { SongContext } from "@/api/generated/types.gen";

export function SongContextBadge({
  songContext,
}: {
  songContext: SongContext | null | undefined;
}) {
  if (!songContext) return null;

  const pills: string[] = [];
  if (songContext.key) pills.push(songContext.key);
  if (songContext.bpm) pills.push(`${songContext.bpm} BPM`);
  if (songContext.genre) pills.push(songContext.genre);
  if (songContext.vibe) pills.push(songContext.vibe);

  if (pills.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 overflow-x-auto">
      {pills.map((pill) => (
        <span
          className="shrink-0 rounded-full border border-border bg-muted/50 px-2 py-0.5 text-muted-foreground text-xs"
          key={pill}
        >
          {pill}
        </span>
      ))}
    </div>
  );
}
