import { client } from "./generated/client.gen";

client.setConfig({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8002",
  withCredentials: true,
});
