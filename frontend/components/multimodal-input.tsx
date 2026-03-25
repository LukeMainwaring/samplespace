"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { ArrowUp, Paperclip, Square } from "lucide-react";
import {
  type Dispatch,
  memo,
  type SetStateAction,
  useCallback,
  useEffect,
  useRef,
} from "react";
import { toast } from "sonner";
import { useLocalStorage, useWindowSize } from "usehooks-ts";
import { MAX_UPLOAD_SIZE_MB } from "@/lib/constants";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  PromptInput,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
} from "./elements/prompt-input";
import { type Attachment, PreviewAttachment } from "./preview-attachment";
import { Button } from "./ui/button";

function PureMultimodalInput({
  chatId,
  input,
  setInput,
  status,
  stop,
  messages: _messages,
  setMessages,
  sendMessage,
  attachments,
  setAttachments,
  onUpload,
  className,
}: {
  chatId: string;
  input: string;
  setInput: Dispatch<SetStateAction<string>>;
  status: UseChatHelpers<ChatMessage>["status"];
  stop: () => void;
  messages: UIMessage[];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  sendMessage: UseChatHelpers<ChatMessage>["sendMessage"];
  attachments: Attachment[];
  setAttachments: Dispatch<SetStateAction<Attachment[]>>;
  onUpload: (file: File) => void;
  className?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { width } = useWindowSize();

  const adjustHeight = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "44px";
    }
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      adjustHeight();
    }
  }, [adjustHeight]);

  const hasAutoFocused = useRef(false);
  useEffect(() => {
    if (!hasAutoFocused.current && width) {
      const timer = setTimeout(() => {
        textareaRef.current?.focus();
        hasAutoFocused.current = true;
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [width]);

  const resetHeight = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "44px";
    }
  }, []);

  const [localStorageInput, setLocalStorageInput] = useLocalStorage(
    "input",
    "",
  );

  useEffect(() => {
    if (textareaRef.current) {
      const domValue = textareaRef.current.value;
      const finalValue = domValue || localStorageInput || "";
      setInput(finalValue);
      adjustHeight();
    }
    // Only run once after hydration
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adjustHeight, localStorageInput, setInput]);

  useEffect(() => {
    setLocalStorageInput(input);
  }, [input, setLocalStorageInput]);

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value);
  };

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = "";

      if (!file.name.toLowerCase().endsWith(".wav")) {
        toast.error("Only WAV files are supported");
        return;
      }
      if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
        toast.error(`File exceeds ${MAX_UPLOAD_SIZE_MB}MB limit`);
        return;
      }

      onUpload(file);
    },
    [onUpload],
  );

  const submitForm = useCallback(() => {
    window.history.replaceState({}, "", `/chat/${chatId}`);

    // Build message text with attachment references
    const completedAttachments = attachments.filter(
      (a) => !a.isUploading && a.sample,
    );
    const attachmentPrefix = completedAttachments
      .map(
        (a) => `[Uploaded sample: ${a.sample?.filename} (ID: ${a.sample?.id})]`,
      )
      .join("\n");

    const text = attachmentPrefix ? `${attachmentPrefix}\n${input}` : input;

    sendMessage({ text });

    setAttachments([]);
    setLocalStorageInput("");
    resetHeight();
    setInput("");

    if (width && width > 768) {
      textareaRef.current?.focus();
    }
  }, [
    input,
    setInput,
    sendMessage,
    setLocalStorageInput,
    width,
    chatId,
    resetHeight,
    attachments,
    setAttachments,
  ]);

  return (
    <div
      className={cn(
        "sticky bottom-0 z-[1] mx-auto flex w-full max-w-4xl gap-2 border-t-0 bg-background px-2 pb-3 md:px-4 md:pb-4",
        className,
      )}
    >
      <div className="relative flex w-full flex-col gap-4">
        <PromptInput
          className="rounded-xl border border-border bg-background p-3 shadow-xs transition-all duration-200 focus-within:border-border hover:border-muted-foreground/50"
          onSubmit={(event) => {
            event.preventDefault();
            if (status !== "ready") {
              toast.error("Please wait for the model to finish its response!");
            } else {
              submitForm();
            }
          }}
        >
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-1.5 px-2 pt-2">
              {attachments.map((attachment) => (
                <PreviewAttachment
                  attachment={attachment}
                  key={attachment.id}
                  onRemove={() =>
                    setAttachments((prev) =>
                      prev.filter((a) => a.id !== attachment.id),
                    )
                  }
                />
              ))}
            </div>
          )}
          <div className="flex flex-row items-start gap-1 sm:gap-2">
            <input
              accept=".wav,audio/wav"
              className="hidden"
              onChange={handleFileChange}
              ref={fileInputRef}
              type="file"
            />
            <button
              className="mt-2.5 ml-1 flex size-6 shrink-0 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
              onClick={() => fileInputRef.current?.click()}
              title="Attach WAV file"
              type="button"
            >
              <Paperclip size={16} />
            </button>
            <PromptInputTextarea
              className="grow resize-none border-0! border-none! bg-transparent p-2 text-base outline-none ring-0 [-ms-overflow-style:none] [scrollbar-width:none] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 [&::-webkit-scrollbar]:hidden"
              disableAutoResize={true}
              maxHeight={200}
              minHeight={44}
              onChange={handleInput}
              ref={textareaRef}
              rows={1}
              value={input}
            />
          </div>
          <PromptInputToolbar className="border-top-0! border-t-0! p-0 shadow-none dark:border-0 dark:border-transparent!">
            {status === "submitted" ? (
              <StopButton setMessages={setMessages} stop={stop} />
            ) : (
              <PromptInputSubmit
                className="size-8 rounded-full bg-primary text-primary-foreground transition-colors duration-200 hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground"
                disabled={
                  !input.trim() &&
                  !attachments.some((a) => !a.isUploading && a.sample)
                }
                status={status}
              >
                <ArrowUp size={14} />
              </PromptInputSubmit>
            )}
          </PromptInputToolbar>
        </PromptInput>
      </div>
    </div>
  );
}

export const MultimodalInput = memo(
  PureMultimodalInput,
  (prevProps, nextProps) => {
    if (prevProps.input !== nextProps.input) {
      return false;
    }
    if (prevProps.status !== nextProps.status) {
      return false;
    }
    if (prevProps.attachments !== nextProps.attachments) {
      return false;
    }
    return true;
  },
);

function PureStopButton({
  stop,
  setMessages,
}: {
  stop: () => void;
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
}) {
  return (
    <Button
      className="size-7 rounded-full bg-foreground p-1 text-background transition-colors duration-200 hover:bg-foreground/90 disabled:bg-muted disabled:text-muted-foreground"
      onClick={(event) => {
        event.preventDefault();
        stop();
        setMessages((messages) => messages);
      }}
    >
      <Square size={14} />
    </Button>
  );
}

const StopButton = memo(PureStopButton);
