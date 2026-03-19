"""torchaudio Dataset for loading audio samples as mel spectrograms."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Audio preprocessing constants
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
DURATION_SEC = 2.0  # Pad/trim all samples to this length
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)

# Sample type labels (alphabetical for deterministic ordering)
SAMPLE_TYPES = [
    "bass",
    "clap",
    "fx",
    "hihat",
    "keys",
    "kick",
    "lead",
    "pad",
    "percussion",
    "snare",
    "vocal",
]
NUM_CLASSES = len(SAMPLE_TYPES)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(SAMPLE_TYPES)}


def _load_and_preprocess(file_path: str) -> torch.Tensor:
    """Load audio file and convert to fixed-length mel spectrogram.

    Returns a tensor of shape (1, N_MELS, time_frames).
    """
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    # data shape: (num_frames, num_channels) -> convert to (num_channels, num_frames)
    waveform = torch.from_numpy(data.T)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Pad or trim to fixed length
    if waveform.shape[1] < TARGET_LENGTH:
        padding = TARGET_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :TARGET_LENGTH]

    # Compute mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_spec = mel_transform(waveform)

    # Convert to log scale (dB)
    amplitude_to_db = T.AmplitudeToDB()
    mel_spec_db: torch.Tensor = amplitude_to_db(mel_spec)

    return mel_spec_db


class SampleDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset that loads audio files as mel spectrograms with sample type labels.

    Expects audio files organized in subdirectories by type:
        data/samples/kick/file1.wav
        data/samples/snare/file2.wav
        ...
    """

    def __init__(
        self,
        samples_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for sample_type, idx in LABEL_TO_IDX.items():
            type_dir = self.samples_dir / sample_type
            if not type_dir.exists():
                continue
            for audio_file in sorted(type_dir.glob("*")):
                if audio_file.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".aiff"}:
                    self.samples.append((audio_file, idx))

        logger.info(f"Loaded {len(self.samples)} samples from {self.samples_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        mel_spec = _load_and_preprocess(str(file_path))

        if self.augment:
            mel_spec = self._apply_augmentation(mel_spec)

        return mel_spec, label

    def _apply_augmentation(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to mel spectrogram."""
        # Time masking (mask a random time segment)
        if torch.rand(1).item() > 0.5:
            time_mask = T.TimeMasking(time_mask_param=20)
            mel_spec = time_mask(mel_spec)

        # Frequency masking (mask random frequency bands)
        if torch.rand(1).item() > 0.5:
            freq_mask = T.FrequencyMasking(freq_mask_param=15)
            mel_spec = freq_mask(mel_spec)

        # Random gain adjustment
        if torch.rand(1).item() > 0.5:
            gain_db = (torch.rand(1).item() - 0.5) * 10  # -5 to +5 dB
            mel_spec = mel_spec + gain_db

        return mel_spec
