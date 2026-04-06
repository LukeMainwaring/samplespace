import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  deleteSampleMutation,
  listSamplesQueryKey,
  updateSampleMutation,
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

export const useUpdateSample = () => {
  const queryClient = useQueryClient();

  return useMutation({
    ...updateSampleMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listSamplesQueryKey() });
    },
  });
};

export const useDeleteSample = () => {
  const queryClient = useQueryClient();

  return useMutation({
    ...deleteSampleMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listSamplesQueryKey() });
    },
  });
};
