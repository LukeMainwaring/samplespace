"use client";

import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  getThreadMessagesQueryKey,
  listThreadsQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { useThreadSongContext } from "@/api/hooks/threads";
import { useUploadSample } from "@/api/hooks/uploads";
import { useAutoResume } from "@/hooks/use-auto-resume";
import type { ChatMessage } from "@/lib/types";
import { ChatActionsProvider } from "./chat-actions-provider";
import { ChatHeader } from "./chat-header";
import { useDataStream } from "./data-stream-provider";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";
import type { Attachment } from "./preview-attachment";

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
  const [threadExists, setThreadExists] = useState(initialMessages.length > 0);
  const { data: songContext } = useThreadSongContext(id, threadExists);
  const { setDataStream } = useDataStream();

  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const uploadMutation = useUploadSample();

  const handleUpload = useCallback(
    (file: File) => {
      const attachment: Attachment = {
        id: crypto.randomUUID(),
        file,
        isUploading: true,
      };
      setAttachments((prev) => [...prev, attachment]);

      uploadMutation.mutate(
        { body: { file } },
        {
          onSuccess: (data) => {
            setAttachments((prev) =>
              prev.map((a) =>
                a.file === file
                  ? { ...a, sample: data, isUploading: false }
                  : a,
              ),
            );
          },
          onError: () => {
            setAttachments((prev) => prev.filter((a) => a.file !== file));
            toast.error(`Failed to upload ${file.name}`);
          },
        },
      );
    },
    [uploadMutation],
  );

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
        setThreadExists(true);
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

  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;

  const sendChatMessage = useCallback((text: string) => {
    sendMessageRef.current({ text });
  }, []);

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

      <ChatActionsProvider sendMessage={sendChatMessage}>
        <Messages
          messages={messages}
          setMessages={setMessages}
          status={status}
        />
      </ChatActionsProvider>

      <MultimodalInput
        attachments={attachments}
        chatId={id}
        input={input}
        messages={messages}
        onUpload={handleUpload}
        sendMessage={sendMessage}
        setAttachments={setAttachments}
        setInput={setInput}
        setMessages={setMessages}
        status={status}
        stop={stop}
      />
    </div>
  );
}
