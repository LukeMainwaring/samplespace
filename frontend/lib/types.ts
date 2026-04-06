import type { UIMessage } from "ai";
import type { KitPayload } from "@/components/elements/kit-block";
import type { KitPreviewPayload } from "@/components/elements/kit-preview-block";
import type { PairVerdictPayload } from "@/components/elements/pair-verdict-block";
import type { SampleResultsPayload } from "@/components/elements/sample-results-block";

export type CustomUIDataTypes = {
  "chat-title": string;
  "sample-results": SampleResultsPayload;
  kit: KitPayload;
  "kit-preview": KitPreviewPayload;
  "pair-verdict": PairVerdictPayload;
  audio: string;
};

export type ChatMessage = UIMessage<Record<string, never>, CustomUIDataTypes>;

export type Chat = {
  id: string;
  title: string | null;
  createdAt: Date;
};
