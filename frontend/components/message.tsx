"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { type DataUIPart, isDataUIPart, isToolUIPart } from "ai";
import equal from "fast-deep-equal";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import Image from "next/image";
import { memo } from "react";
import type { ChatMessage, CustomUIDataTypes } from "@/lib/types";
import { cn, sanitizeText } from "@/lib/utils";
import { useDataStream } from "./data-stream-provider";
import { AudioBlock } from "./elements/audio-block";
import { BouncingDots } from "./elements/bouncing-dots";
import { KitBlock } from "./elements/kit-block";
import { KitPreviewBlock } from "./elements/kit-preview-block";
import { MessageContent } from "./elements/message";
import { PairVerdictBlock } from "./elements/pair-verdict-block";
import { Response } from "./elements/response";
import { SampleResultsBlock } from "./elements/sample-results-block";
import { ToolCall } from "./elements/tool-call";
import { MessageActions } from "./message-actions";

function DataPartRenderer({ part }: { part: DataUIPart<CustomUIDataTypes> }) {
  const code =
    typeof part.data === "string" ? part.data : JSON.stringify(part.data);
  switch (part.type) {
    case "data-sample-results":
      return <SampleResultsBlock code={code} isIncomplete={false} />;
    case "data-kit":
      return <KitBlock code={code} isIncomplete={false} />;
    case "data-kit-preview":
      return <KitPreviewBlock code={code} isIncomplete={false} />;
    case "data-audio":
      return <AudioBlock code={code} isIncomplete={false} />;
    case "data-pair-verdict":
      return <PairVerdictBlock code={code} isIncomplete={false} />;
    default:
      return null;
  }
}

const AssistantAvatar = ({ isLoading }: { isLoading?: boolean }) => (
  <div
    className={cn(
      "-mt-1 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full bg-background ring-1 ring-border",
      isLoading && "animate-riff",
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
);

const PurePreviewMessage = ({
  message,
  isLoading,
  setMessages: _setMessages,
  requiresScrollPadding: _requiresScrollPadding,
}: {
  message: ChatMessage;
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  requiresScrollPadding: boolean;
}) => {
  useDataStream();

  const hasTextParts = message.parts?.some(
    (p) => p.type === "text" && p.text?.trim(),
  );
  const hasToolParts = message.parts?.some((p) => isToolUIPart(p));
  const hasDataParts = message.parts?.some((p) => isDataUIPart(p));
  const hasVisibleContent = hasTextParts || hasToolParts || hasDataParts;

  return (
    <div
      className="group/message fade-in w-full animate-in duration-200"
      data-role={message.role}
    >
      <div
        className={cn("flex w-full items-start gap-2 md:gap-3", {
          "justify-end": message.role === "user",
          "justify-start": message.role === "assistant",
        })}
      >
        {message.role === "assistant" && (
          <AssistantAvatar isLoading={isLoading} />
        )}

        <div
          className={cn("flex flex-col", {
            "gap-2 md:gap-4": hasVisibleContent || isLoading,
            "w-full":
              message.role === "assistant" && (hasVisibleContent || isLoading),
            "max-w-[calc(100%-2.5rem)] sm:max-w-[min(fit-content,80%)]":
              message.role === "user",
          })}
        >
          {!hasVisibleContent && isLoading && message.role === "assistant" && (
            <div className="flex items-center gap-1 p-0 text-muted-foreground text-sm">
              <span className="animate-shimmer">Riffing</span>
              <BouncingDots />
            </div>
          )}

          {message.parts?.map((part, index) => {
            const key = `message-${message.id}-part-${index}`;
            if (part.type === "text") {
              // Render verdict messages as compact pills
              if (
                message.role === "user" &&
                part.text.startsWith("[PAIR_VERDICT]")
              ) {
                const isApproved = part.text.includes("Works |");
                return (
                  <div key={key} className="flex justify-end">
                    <div
                      className={cn(
                        "flex w-fit items-center gap-1.5 rounded-2xl px-3 py-1.5 text-xs font-medium",
                        isApproved
                          ? "bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400"
                          : "bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400",
                      )}
                    >
                      {isApproved ? (
                        <ThumbsUp size={12} />
                      ) : (
                        <ThumbsDown size={12} />
                      )}
                      {isApproved ? "Works" : "Doesn't work"}
                    </div>
                  </div>
                );
              }
              return (
                <div key={key}>
                  <MessageContent
                    className={cn({
                      "wrap-break-word w-fit rounded-2xl px-3 py-2 text-right bg-primary text-primary-foreground":
                        message.role === "user",
                      "rounded-none bg-transparent px-0 py-0 text-left":
                        message.role === "assistant",
                    })}
                  >
                    <Response>{sanitizeText(part.text)}</Response>
                  </MessageContent>
                </div>
              );
            }
            if (isToolUIPart(part)) {
              return (
                <ToolCall
                  isStreaming={isLoading && !hasTextParts && !hasDataParts}
                  key={key}
                  part={part}
                />
              );
            }
            if (isDataUIPart(part)) {
              return <DataPartRenderer key={key} part={part} />;
            }
            return null;
          })}

          <MessageActions isLoading={isLoading} message={message} />
        </div>
      </div>
    </div>
  );
};

export const PreviewMessage = memo(
  PurePreviewMessage,
  (prevProps, nextProps) => {
    if (nextProps.isLoading) {
      return false;
    }
    if (
      prevProps.isLoading === nextProps.isLoading &&
      prevProps.message.id === nextProps.message.id &&
      prevProps.requiresScrollPadding === nextProps.requiresScrollPadding &&
      equal(prevProps.message.parts, nextProps.message.parts)
    ) {
      return true;
    }
    return false;
  },
);

export const RiffingMessage = () => {
  return (
    <div
      className="group/message fade-in w-full animate-in duration-300"
      data-role="assistant"
    >
      <div className="flex items-start justify-start gap-3">
        <AssistantAvatar isLoading />

        <div className="flex w-full flex-col gap-2 md:gap-4">
          <div className="flex items-center gap-1 p-0 text-muted-foreground text-sm">
            <span className="animate-shimmer">Riffing</span>
            <BouncingDots />
          </div>
        </div>
      </div>
    </div>
  );
};
