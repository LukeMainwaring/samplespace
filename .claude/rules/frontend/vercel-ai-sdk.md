---
paths:
  - "frontend/components/**/*.tsx"
  - "frontend/app/(chat)/**/*.{ts,tsx}"
  - "frontend/hooks/**/*.{ts,tsx}"
  - "frontend/api/hooks/**/*.{ts,tsx}"
  - "frontend/lib/**/*.ts"
---

# Vercel AI SDK Rules

## Docs are split between two places

The Vercel AI SDK's documentation lives in two places with **different content**:

1. **`docs/vercel-ai-sdk-ui.txt`** — local pinned reference, **UI surface only**
   (`useChat`, message parts, chat transports, the SSE stream protocol,
   tool-call rendering). Refresh via the `updating-deps` skill — its Phase 4
   docs-fetch step is the source of truth for which upstream pages we mirror.

2. **`https://ai-sdk.dev/`** — everything else. AI SDK Core (`generateText`,
   server-side `streamText`, model providers), guides, framework examples,
   cookbook. Not cached locally; fetch ad-hoc with WebFetch.

**Rule of thumb:** if you're grepping `vercel-ai-sdk-ui.txt` for anything
outside the UI slice and finding nothing, it hasn't been deleted — it's on
the web docs. Use WebFetch on `https://ai-sdk.dev/docs/...` before giving up.

## The chat surface in this project

- **AI SDK UI only — no Core server-side.** `app/(chat)/api/chat/route.ts` is
  a thin proxy to the FastAPI backend's `POST /api/agent/chat`; Pydantic AI
  emits the SSE stream. When the docs show `streamText` /
  `toUIMessageStreamResponse`, that's a **spec for what the backend must emit**,
  not code to write here.
- **`useChat` is on the v5/v6 transport API.** `chat.tsx` constructs a
  `DefaultChatTransport({ api: "/api/chat" })`, passes it as `transport`, and
  destructures `{ messages, setMessages, sendMessage, status, stop, resumeStream }`
  from `useChat<ChatMessage>(...)`. There is no `handleInputChange` /
  `handleSubmit` / `input` field from `useChat` — the input `useState` in
  `chat.tsx` is a local component state passed down to `MultimodalInput`, not
  the AI SDK's input. New chat code should follow the same shape.
- **Keep `ChatMessage` threaded.** `useChat` is parameterized by the custom
  `ChatMessage` type in `frontend/lib/types.ts`
  (`UIMessage<Record<string, never>, CustomUIDataTypes>` — `Record<string, never>`
  is the message-metadata slot, intentionally empty rather than `never`).
  Carry that generic through every `UseChatHelpers<ChatMessage>` site —
  falling back to plain `UIMessage` loses the project's custom data-part typing.
- **Renderable interactive blocks come in on `data-<name>` parts.** Five custom
  blocks render inline in the message stream: `data-sample-results`, `data-kit`,
  `data-kit-preview`, `data-audio`, `data-pair-verdict`. Adding a new renderable
  block is **two places**:
  1. `frontend/lib/types.ts` — extend `CustomUIDataTypes`.
  2. `frontend/components/message.tsx::DataPartRenderer` — add a
     `case "data-<name>":` branch returning the block component (sibling to
     `SampleResultsBlock`, `KitBlock`, `KitPreviewBlock`, `AudioBlock`,
     `PairVerdictBlock`).
  Non-rendering data parts (e.g. `chat-title`, which triggers a sidebar
  refresh) only need step 1 plus a handler in
  `components/data-stream-handler.tsx`. Generic tool-call transparency
  (call/result for arbitrary tools) is handled separately by
  `components/elements/tool-call.tsx`.
- **`ChatActionsProvider` carries `sendMessage` to descendants.** Children that
  need to inject a user turn (verdict buttons, kit actions) consume the context
  rather than receiving `sendMessage` as a prop drilled through every layer.
