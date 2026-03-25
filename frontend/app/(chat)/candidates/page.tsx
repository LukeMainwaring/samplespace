"use client";

import { CandidateSamples } from "@/components/candidate-samples";

export default function CandidatesPage() {
  return (
    <div className="flex h-dvh min-w-0 flex-col overflow-y-auto bg-background">
      <CandidateSamples />
    </div>
  );
}
