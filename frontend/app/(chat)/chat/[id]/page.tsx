"use client";

import { useParams } from "next/navigation";
import { useThreadMessages } from "@/api/hooks/threads";
import { ChatPanel } from "@/components/chat-panel";

export default function Page() {
  const { id } = useParams<{ id: string }>();
  const { data: initialMessages, isLoading } = useThreadMessages(id);

  if (isLoading) {
    return <div className="flex h-dvh" />;
  }

  return <ChatPanel id={id} initialMessages={initialMessages ?? []} />;
}
