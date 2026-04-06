import type { UIMessage } from "ai";

export type CustomUIDataTypes = {
  "chat-title": string;
  "sample-results": unknown;
  kit: unknown;
  "kit-preview": unknown;
  "pair-verdict": unknown;
  audio: unknown;
};

export type ChatMessage = UIMessage<Record<string, never>, CustomUIDataTypes>;

export type Chat = {
  id: string;
  title: string | null;
  createdAt: Date;
};
