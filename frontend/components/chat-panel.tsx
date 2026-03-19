"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, isToolUIPart, type UIMessage } from "ai";
import equal from "fast-deep-equal";
import { ArrowUp, Square } from "lucide-react";
import { memo, useMemo, useRef, useState } from "react";
import { useStickToBottom } from "use-stick-to-bottom";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { BouncingDots } from "./bouncing-dots";
import { Response } from "./response";
import { ToolCall } from "./tool-call";

const CHAT_ID = "samplespace-chat";

function sanitizeText(text: string) {
  return text.replace("<has_function_call>", "");
}

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
          className={cn("flex w-full items-start gap-3", {
            "justify-end": message.role === "user",
            "justify-start": message.role === "assistant",
          })}
        >
          {message.role === "assistant" && (
            <div
              className={cn(
                "-mt-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold ring-1 ring-border",
                isLoading && "animate-pulse",
              )}
            >
              SS
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
                      className={cn("overflow-hidden rounded-lg text-sm", {
                        "w-fit rounded-2xl bg-primary px-3 py-2 text-right text-primary-foreground":
                          message.role === "user",
                        "bg-transparent px-0 py-0 text-left":
                          message.role === "assistant",
                      })}
                    >
                      {message.role === "assistant" ? (
                        <Response>{sanitizeText(part.text)}</Response>
                      ) : (
                        part.text
                      )}
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

export function ChatPanel() {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );

  const { messages, sendMessage, status, stop } = useChat({
    id: CHAT_ID,
    transport,
    experimental_throttle: 100,
    onError: (error) => {
      console.error("Chat error:", error);
    },
  });

  const { scrollRef, contentRef } = useStickToBottom();

  const isStreaming = status === "streaming" || status === "submitted";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || status !== "ready") return;

    sendMessage({
      role: "user" as const,
      parts: [{ type: "text" as const, text: input }],
    });
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto" ref={scrollRef}>
        <div
          className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6"
          ref={contentRef}
        >
          {messages.length === 0 && (
            <div className="flex flex-1 flex-col items-center justify-center gap-2 py-20 text-center text-muted-foreground">
              <div className="flex size-12 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold">
                SS
              </div>
              <h2 className="mt-2 text-lg font-semibold text-foreground">
                SampleSpace
              </h2>
              <p className="max-w-md text-sm">
                Ask me to find samples, check key compatibility, or suggest
                complementary sounds for your production.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <PureMessage
              isLoading={isStreaming && index === messages.length - 1}
              key={message.id}
              message={message}
            />
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="border-t bg-background p-4">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2"
          onSubmit={handleSubmit}
        >
          <textarea
            className="flex-1 resize-none rounded-xl border bg-background px-4 py-3 text-sm outline-none ring-ring placeholder:text-muted-foreground focus:ring-2"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe the sound you're looking for..."
            ref={textareaRef}
            rows={1}
            value={input}
          />
          {isStreaming ? (
            <Button
              className="shrink-0 rounded-xl"
              onClick={stop}
              size="icon"
              type="button"
              variant="outline"
            >
              <Square size={16} />
            </Button>
          ) : (
            <Button
              className="shrink-0 rounded-xl"
              disabled={!input.trim()}
              size="icon"
              type="submit"
            >
              <ArrowUp size={16} />
            </Button>
          )}
        </form>
      </div>
    </div>
  );
}
