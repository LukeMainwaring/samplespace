import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listSamplesQueryKey,
  uploadSampleMutation,
} from "../generated/@tanstack/react-query.gen";

// Ensure client is configured with baseURL
import "../client";

export const useUploadSample = () => {
  const queryClient = useQueryClient();

  return useMutation({
    ...uploadSampleMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listSamplesQueryKey() });
    },
  });
};
