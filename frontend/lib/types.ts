import type { UIMessage } from "ai";

export type ChatMessage = UIMessage;

export type Chat = {
  id: string;
  title: string | null;
  createdAt: Date;
};
