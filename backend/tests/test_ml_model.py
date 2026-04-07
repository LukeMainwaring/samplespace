"""Tests for the dual-head CNN architecture (SampleCNN, SEBlock, ConvBlock)."""

import torch

from samplespace.ml.model import CNN_EMBEDDING_DIM, ConvBlock, SampleCNN, SEBlock


class TestSEBlock:
    def test_preserves_shape(self) -> None:
        block = SEBlock(channels=64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_different_channel_counts(self) -> None:
        for channels in [32, 128, 256]:
            block = SEBlock(channels=channels)
            x = torch.randn(1, channels, 8, 8)
            assert block(x).shape == x.shape

    def test_reduction_floor(self) -> None:
        """Small channel counts should not reduce below 4."""
        block = SEBlock(channels=16, reduction=16)
        # mid = max(16 // 16, 4) = 4
        x = torch.randn(1, 16, 4, 4)
        assert block(x).shape == x.shape


class TestConvBlock:
    def test_halves_spatial_dims(self) -> None:
        block = ConvBlock(in_channels=1, out_channels=64)
        x = torch.randn(2, 1, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_skip_connection_same_channels(self) -> None:
        block = ConvBlock(in_channels=64, out_channels=64)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 8, 8)


class TestSampleCNN:
    def test_output_shapes(self) -> None:
        model = SampleCNN(num_classes=16)
        x = torch.randn(2, 1, 128, 128)
        logits, emb = model(x)
        assert logits.shape == (2, 16)
        assert emb.shape == (2, CNN_EMBEDDING_DIM)

    def test_embedding_l2_normalized(self) -> None:
        model = SampleCNN(num_classes=16)
        model.eval()
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            _, emb = model(x)
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_deterministic_in_eval(self) -> None:
        model = SampleCNN(num_classes=16)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            logits1, emb1 = model(x)
            logits2, emb2 = model(x)
        assert torch.equal(logits1, logits2)
        assert torch.equal(emb1, emb2)

    def test_single_sample_batch(self) -> None:
        """Single-sample inference works in eval mode (BatchNorm1d requires batch>1 in train)."""
        model = SampleCNN(num_classes=16)
        model.eval()
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            logits, emb = model(x)
        assert logits.shape == (1, 16)
        assert emb.shape == (1, CNN_EMBEDDING_DIM)
