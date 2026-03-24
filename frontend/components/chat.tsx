"use client";

import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  getThreadMessagesQueryKey,
  listThreadsQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { useThreadSongContext } from "@/api/hooks/threads";
import { useAutoResume } from "@/hooks/use-auto-resume";
import type { ChatMessage } from "@/lib/types";
import { ChatHeader } from "./chat-header";
import { useDataStream } from "./data-stream-provider";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";

export function Chat({
  id,
  initialMessages,
  autoResume = false,
}: {
  id: string;
  initialMessages: ChatMessage[];
  autoResume?: boolean;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { data: songContext } = useThreadSongContext(id);
  const { setDataStream } = useDataStream();

  const [input, setInput] = useState("");

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );

  const { messages, setMessages, sendMessage, status, stop, resumeStream } =
    useChat<ChatMessage>({
      id,
      transport,
      messages: initialMessages,
      experimental_throttle: 100,
      onData: (dataPart) => {
        setDataStream((ds) => (ds ? [...ds, dataPart] : []));
      },
      onFinish: () => {
        queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
        queryClient.invalidateQueries({
          queryKey: getThreadMessagesQueryKey({
            path: { thread_id: id },
          }),
        });
      },
      onError: (error) => {
        console.error("Chat error:", error);
      },
    });

  useAutoResume({
    autoResume,
    initialMessages,
    resumeStream,
    setMessages,
  });

  // Handle browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      router.refresh();
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [router]);

  return (
    <div className="flex h-dvh min-w-0 flex-col bg-background">
      <ChatHeader songContext={songContext} />

      <Messages messages={messages} setMessages={setMessages} status={status} />

      <MultimodalInput
        chatId={id}
        input={input}
        messages={messages}
        sendMessage={sendMessage}
        setInput={setInput}
        setMessages={setMessages}
        status={status}
        stop={stop}
      />
    </div>
  );
}
