# Frontend Patterns

TypeScript/Next.js conventions for the samplespace frontend.

## Imports

- Use the `@/` path alias for all imports outside the current directory: `import { Button } from "@/components/ui/button"` not `from "./ui/button"`
- Relative imports (`./`, `../`) are only acceptable for sibling files in the same directory (e.g., `./sample-card` from within `elements/`)

## UI Components

- Use shadcn/ui components from `components/ui/` instead of raw HTML elements. Specifically:
  - `<Button>` over `<button>` — provides consistent focus rings, disabled states, and cursor styles
  - `<Input>` over `<input>` — except for hidden file inputs (`type="file" className="hidden"`) which are fine as raw elements
  - `<Textarea>` over `<textarea>`
  - `<Skeleton>` over `<div className="animate-pulse ...">` for loading placeholders
  - `<Tooltip>` over `title=` attributes on interactive elements (buttons, icon buttons). Native `title=` is acceptable on text elements for truncation hints.
  - `<Separator>` for standalone visual dividers. Border classes (`border-b`, `border-t`) are fine when the border is part of a container's layout (e.g., section headers with padding).
  - `<AlertDialog>` for confirmation dialogs, `<DropdownMenu>` for context menus, `<Collapsible>` for expandable sections, `<Sheet>` for slide-out panels
- When a shadcn component's default variant matches your needs, don't rewrite the styles — just use the variant. Override with `className` only for styles the variant doesn't cover.
- Do not manually edit files in `components/ui/` unless adding a new shadcn component or customizing an existing variant.

## Code Style

- Use kebab-case for filenames (e.g., `sample-browser.tsx`, `audio-block.tsx`)
- Colocate types with their component unless shared across multiple files
- Prefer `useCallback` and `memo` for expensive renders and callbacks passed as props
- Use `cn()` from `@/lib/utils` for conditional class merging
