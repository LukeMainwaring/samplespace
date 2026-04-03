import type { SampleType } from "@/api/generated/types.gen";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002";

/** Maximum upload file size in megabytes. Must match backend UPLOAD_MAX_SIZE_MB. */
export const MAX_UPLOAD_SIZE_MB = 50;

/** All sample types, derived from backend SampleType enum via generated client. */
export const SAMPLE_TYPES = [
  "bass",
  "clap",
  "cymbal",
  "drum",
  "fx",
  "guitar",
  "hihat",
  "horn",
  "keys",
  "kick",
  "pad",
  "percussion",
  "snare",
  "strings",
  "synth",
  "vocal",
] as const satisfies readonly SampleType[];
