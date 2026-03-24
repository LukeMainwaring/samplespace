import { headers } from "next/headers";
import { Suspense } from "react";
import { ChatPanel } from "@/components/chat-panel";

export default function Page() {
  return (
    <Suspense fallback={<div className="flex h-dvh" />}>
      <NewChatPage />
    </Suspense>
  );
}

async function NewChatPage() {
  await headers(); // opt into dynamic rendering
  const id = crypto.randomUUID();
  return <ChatPanel id={id} initialMessages={[]} key={id} />;
}
