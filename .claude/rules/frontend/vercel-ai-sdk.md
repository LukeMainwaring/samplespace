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
  a thin proxy to the FastAPI backend's `POST /agent/chat`; Pydantic AI emits
  the SSE stream. When the docs show `streamText` / `toUIMessageStreamResponse`,
  that's a **spec for what the backend must emit**, not code to write here.
- **Keep `ChatMessage` threaded.** `useChat` is parameterized by the custom
  `ChatMessage` type in `frontend/lib/types.ts` (`UIMessage<never, CustomUIDataTypes>`).
  Carry that generic through every `UseChatHelpers<ChatMessage>` site — falling
  back to plain `UIMessage` loses the project's custom data-part typing.
- **Interactive blocks render via `data-<name>` parts, not tool parts.** The
  agent emits structured data parts; the frontend renders them as code-fence-style
  blocks. New block types are added in two places:
  1. `frontend/lib/types.ts` — extend `CustomUIDataTypes`.
  2. `frontend/components/message.tsx::DataPartRenderer` — add a `case "data-<name>":`
     branch returning the block component (sibling to `SampleResultsBlock`,
     `KitBlock`, `KitPreviewBlock`, `AudioBlock`, `PairVerdictBlock`).
  Generic tool-call transparency (call/result for arbitrary tools) is handled
  separately by `components/elements/tool-call.tsx`.
- **`ChatActionsProvider` carries `sendMessage` to descendants.** Children that
  need to inject a user turn (verdict buttons, kit actions) consume the context
  rather than receiving `sendMessage` as a prop drilled through every layer.
- **Heads-up: the chat surface still uses the pre-v5 `useChat` shape**
  (`handleInputChange` / `handleSubmit` / `input` / `content`). The refreshed
  `vercel-ai-sdk-ui.txt` documents the v5/v6 transport API (`DefaultChatTransport`,
  `sendMessage`, `parts`-based messages). Migrating is its own PR — Vercel ships
  `pnpm dlx @ai-sdk/codemod v6` for the bulk rewrite. Don't pattern-match new
  code off the local docs without checking what `chat.tsx` actually does.
