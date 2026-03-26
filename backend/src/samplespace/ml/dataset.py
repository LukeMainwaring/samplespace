"""torchaudio Dataset for loading audio samples as mel spectrograms."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F
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

# Shared transforms — created once at module load, reused by dataset and inference
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
amplitude_to_db = T.AmplitudeToDB()


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

    # Compute mel spectrogram using shared transforms
    mel_spec = mel_transform(waveform)
    mel_spec_db: torch.Tensor = amplitude_to_db(mel_spec)

    return mel_spec_db


def _load_waveform(file_path: str) -> torch.Tensor:
    """Load audio file as a mono waveform at SAMPLE_RATE.

    Returns a tensor of shape (1, num_frames) — NOT fixed-length.
    Callers handle padding/trimming after any waveform-level augmentations.
    """
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform


def _pad_or_trim(waveform: torch.Tensor, target_length: int = TARGET_LENGTH) -> torch.Tensor:
    """Pad (zero) or trim a waveform to exactly target_length samples."""
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]
    return waveform


class SampleDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset that loads audio files as mel spectrograms with sample type labels.

    Expects audio files organized in subdirectories by type:
        data/samples/kick/file1.wav
        data/samples/snare/file2.wav
        ...

    Augmentation pipeline (training only):
        1. Waveform-level: random pitch shift (±2 semitones), random time stretch
           (0.9-1.1x), random crop (for samples longer than DURATION_SEC)
        2. Spectrogram-level: time masking, frequency masking, random gain (±5 dB)
    """

    def __init__(
        self,
        samples_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []
        self._cache: dict[int, torch.Tensor] = {}

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

        if self.augment:
            # Waveform-level augmentations require loading raw audio
            waveform = _load_waveform(str(file_path))
            waveform = self._apply_waveform_augmentation(waveform)
            waveform = _pad_or_trim(waveform)

            mel_spec = mel_transform(waveform)
            mel_spec = amplitude_to_db(mel_spec)
            mel_spec = self._apply_spectrogram_augmentation(mel_spec)
        else:
            # Cache base spectrograms for non-augmented access (validation/inference)
            if idx not in self._cache:
                self._cache[idx] = _load_and_preprocess(str(file_path))
            mel_spec = self._cache[idx]

        return mel_spec, label

    def _apply_waveform_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random waveform-level augmentations before mel conversion."""
        # Random pitch shift (±2 semitones)
        if torch.rand(1).item() > 0.5:
            semitones = (torch.rand(1).item() - 0.5) * 4  # -2 to +2
            waveform = F.pitch_shift(waveform, SAMPLE_RATE, n_steps=semitones)

        # Random time stretch (0.9x to 1.1x)
        if torch.rand(1).item() > 0.5:
            rate = 0.9 + torch.rand(1).item() * 0.2  # 0.9 to 1.1
            waveform = F.speed(waveform, SAMPLE_RATE, factor=rate)[0]

        # Random crop (for samples longer than target, pick random start)
        if waveform.shape[1] > TARGET_LENGTH:
            max_start = waveform.shape[1] - TARGET_LENGTH
            start = int(torch.randint(0, max_start, (1,)).item())
            waveform = waveform[:, start : start + TARGET_LENGTH]

        return waveform

    def _apply_spectrogram_augmentation(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply random spectrogram-level augmentations (SpecAugment-style)."""
        # Time masking
        if torch.rand(1).item() > 0.5:
            time_mask = T.TimeMasking(time_mask_param=20)
            mel_spec = time_mask(mel_spec)

        # Frequency masking
        if torch.rand(1).item() > 0.5:
            freq_mask = T.FrequencyMasking(freq_mask_param=15)
            mel_spec = freq_mask(mel_spec)

        # Random gain adjustment
        if torch.rand(1).item() > 0.5:
            gain_db = (torch.rand(1).item() - 0.5) * 10  # -5 to +5 dB
            mel_spec = mel_spec + gain_db

        return mel_spec
