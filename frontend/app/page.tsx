"use client";

import { ChatPanel } from "@/components/chat-panel";
import { SampleBrowser } from "@/components/sample-browser";

export default function HomePage() {
  return (
    <main className="flex h-dvh">
      {/* Chat panel — primary interaction */}
      <div className="flex-1 border-r">
        <ChatPanel />
      </div>

      {/* Sample browser — sidebar */}
      <div className="hidden w-80 lg:block xl:w-96">
        <SampleBrowser />
      </div>
    </main>
  );
}
