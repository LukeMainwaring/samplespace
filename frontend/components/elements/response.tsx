"use client";

import { type ComponentProps, memo } from "react";
import { type PluginConfig, Streamdown } from "streamdown";
import { cn } from "@/lib/utils";
import { AudioBlock } from "./audio-block";
import { KitBlock } from "./kit-block";
import { KitPreviewBlock } from "./kit-preview-block";
import { PairVerdictBlock } from "./pair-verdict-block";
import { SampleResultsBlock } from "./sample-results-block";

const plugins: PluginConfig = {
  renderers: [
    { language: "audio", component: AudioBlock },
    { language: "kit", component: KitBlock },
    { language: "kit-preview", component: KitPreviewBlock },
    { language: "pair-verdict", component: PairVerdictBlock },
    { language: "sample-results", component: SampleResultsBlock },
  ],
};

type ResponseProps = ComponentProps<typeof Streamdown>;

export const Response = memo(
  ({ className, ...props }: ResponseProps) => (
    <Streamdown
      className={cn(
        "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_code]:whitespace-pre-wrap [&_code]:break-words [&_pre]:max-w-full [&_pre]:overflow-x-auto",
        className,
      )}
      plugins={plugins}
      {...props}
    />
  ),
  (prevProps, nextProps) => prevProps.children === nextProps.children,
);

Response.displayName = "Response";
