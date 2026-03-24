"use client";

import { SampleBrowser } from "@/components/sample-browser";

export default function SamplesPage() {
  return (
    <div className="flex h-dvh min-w-0 flex-col overflow-y-auto bg-background">
      <SampleBrowser />
    </div>
  );
}
