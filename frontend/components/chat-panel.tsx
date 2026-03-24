"use client";

import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport, isToolUIPart, type UIMessage } from "ai";
import equal from "fast-deep-equal";
import { ArrowUp, Square } from "lucide-react";
import Image from "next/image";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { listThreadsQueryKey } from "@/api/generated/@tanstack/react-query.gen";
import { cn, sanitizeText } from "@/lib/utils";
import { BouncingDots } from "./bouncing-dots";
import { ChatHeader } from "./chat-header";
import {
  PromptInput,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
} from "./elements/prompt-input";
import { MessageActions } from "./message-actions";
import { Response } from "./response";
import { ToolCall } from "./tool-call";
import { Button } from "./ui/button";

const PureMessage = memo(
  ({ message, isLoading }: { message: UIMessage; isLoading: boolean }) => {
    const hasTextParts = message.parts?.some(
      (p) => p.type === "text" && p.text?.trim(),
    );
    const hasToolParts = message.parts?.some((p) => isToolUIPart(p));
    const hasVisibleContent = hasTextParts || hasToolParts;

    return (
      <div
        className="group/message w-full animate-in fade-in duration-200"
        data-role={message.role}
      >
        <div
          className={cn("flex w-full items-start gap-2 md:gap-3", {
            "justify-end": message.role === "user",
            "justify-start": message.role === "assistant",
          })}
        >
          {message.role === "assistant" && (
            <div
              className={cn(
                "-mt-1 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full bg-background ring-1 ring-border",
                isLoading && "animate-pulse",
              )}
            >
              <Image
                alt="SampleSpace"
                className="size-full object-cover"
                height={32}
                src="/images/samplespace-logo.png"
                width={32}
              />
            </div>
          )}

          <div
            className={cn("flex flex-col", {
              "gap-2 md:gap-4": hasVisibleContent || isLoading,
              "w-full":
                message.role === "assistant" &&
                (hasVisibleContent || isLoading),
              "max-w-[calc(100%-2.5rem)] sm:max-w-[min(fit-content,80%)]":
                message.role === "user",
            })}
          >
            {!hasVisibleContent &&
              isLoading &&
              message.role === "assistant" && (
                <div className="flex items-center gap-1 p-0 text-muted-foreground text-sm">
                  <span>Thinking</span>
                  <BouncingDots />
                </div>
              )}

            {message.parts?.map((part, index) => {
              const key = `message-${message.id}-part-${index}`;
              if (part.type === "text") {
                return (
                  <div key={key}>
                    <div
                      className={cn(
                        "flex flex-col gap-2 overflow-hidden rounded-lg px-4 py-3 text-foreground text-sm",
                        {
                          "wrap-break-word w-fit rounded-2xl px-3 py-2 text-right text-white":
                            message.role === "user",
                          "bg-transparent px-0 py-0 text-left":
                            message.role === "assistant",
                        },
                      )}
                      style={
                        message.role === "user"
                          ? { backgroundColor: "#006cff" }
                          : undefined
                      }
                    >
                      <Response>{sanitizeText(part.text)}</Response>
                    </div>
                  </div>
                );
              }
              if (isToolUIPart(part)) {
                return (
                  <ToolCall
                    isStreaming={isLoading && !hasTextParts}
                    key={key}
                    part={part}
                  />
                );
              }
              return null;
            })}

            <MessageActions isLoading={isLoading} message={message} />
          </div>
        </div>
      </div>
    );
  },
  (prevProps, nextProps) => {
    if (nextProps.isLoading) return false;
    if (
      prevProps.isLoading === nextProps.isLoading &&
      prevProps.message.id === nextProps.message.id &&
      equal(prevProps.message.parts, nextProps.message.parts)
    ) {
      return true;
    }
    return false;
  },
);

PureMessage.displayName = "PureMessage";

export function ChatPanel({
  id,
  initialMessages,
}: {
  id: string;
  initialMessages: UIMessage[];
}) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const queryClient = useQueryClient();
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );

  const { messages, setMessages, sendMessage, status, stop } = useChat({
    id,
    transport,
    messages: initialMessages,
    experimental_throttle: 100,
    onFinish: () => {
      queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
    },
    onError: (error) => {
      console.error("Chat error:", error);
    },
  });

  const isStreaming = status === "streaming" || status === "submitted";

  // Auto-scroll to bottom when new messages arrive
  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on message changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submitForm = useCallback(() => {
    if (!input.trim() || status !== "ready") {
      if (status !== "ready") {
        toast.error("Please wait for the model to finish its response!");
      }
      return;
    }

    window.history.replaceState({}, "", `/chat/${id}`);

    sendMessage({ text: input });
    setInput("");
  }, [input, status, id, sendMessage]);

  return (
    <div className="flex h-dvh min-w-0 flex-col bg-background">
      <ChatHeader />

      {/* Messages */}
      <div className="relative flex-1">
        <div
          className="absolute inset-0 touch-pan-y overflow-y-auto"
          ref={messagesContainerRef}
        >
          <div className="mx-auto flex min-w-0 max-w-4xl flex-col gap-4 px-2 py-4 md:gap-6 md:px-4">
            {messages.length === 0 && (
              <div className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center px-4 md:mt-16 md:px-8">
                <div className="flex flex-col items-center gap-2 text-center text-muted-foreground">
                  <Image
                    alt="SampleSpace"
                    className="size-20 rounded-full"
                    height={64}
                    src="/images/samplespace-logo.png"
                    width={64}
                  />
                  <h2 className="mt-2 text-lg font-semibold text-foreground">
                    SampleSpace
                  </h2>
                  <p className="max-w-md text-sm">
                    Ask me to find samples, check key compatibility, or suggest
                    complementary sounds for your production.
                  </p>
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <PureMessage
                isLoading={isStreaming && index === messages.length - 1}
                key={message.id}
                message={message}
              />
            ))}

            <div
              className="min-h-[24px] min-w-[24px] shrink-0"
              ref={messagesEndRef}
            />
          </div>
        </div>
      </div>

      {/* Input */}
      <div className="sticky bottom-0 z-[1] mx-auto flex w-full max-w-4xl gap-2 border-t-0 bg-background px-2 pb-3 md:px-4 md:pb-4">
        <div className="relative flex w-full flex-col gap-4">
          <PromptInput
            className="rounded-xl border border-border bg-background p-3 shadow-xs transition-all duration-200 focus-within:border-border hover:border-muted-foreground/50"
            onSubmit={(event) => {
              event.preventDefault();
              submitForm();
            }}
          >
            <div className="flex flex-row items-start gap-1 sm:gap-2">
              <PromptInputTextarea
                className="grow resize-none border-0! border-none! bg-transparent p-2 text-base outline-none ring-0 [-ms-overflow-style:none] [scrollbar-width:none] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 [&::-webkit-scrollbar]:hidden"
                disableAutoResize={true}
                maxHeight={200}
                minHeight={44}
                onChange={(e) => setInput(e.target.value)}
                ref={textareaRef}
                rows={1}
                value={input}
              />
            </div>
            <PromptInputToolbar className="border-top-0! border-t-0! p-0 shadow-none dark:border-0 dark:border-transparent!">
              {status === "submitted" ? (
                <Button
                  className="size-7 rounded-full bg-foreground p-1 text-background transition-colors duration-200 hover:bg-foreground/90 disabled:bg-muted disabled:text-muted-foreground"
                  onClick={(event) => {
                    event.preventDefault();
                    stop();
                    setMessages((msgs) => msgs);
                  }}
                >
                  <Square className="size-4" />
                </Button>
              ) : (
                <PromptInputSubmit
                  className="size-8 rounded-full bg-primary text-primary-foreground transition-colors duration-200 hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground"
                  disabled={!input.trim()}
                  status={status}
                >
                  <ArrowUp size={14} />
                </PromptInputSubmit>
              )}
            </PromptInputToolbar>
          </PromptInput>
        </div>
      </div>
    </div>
  );
}
