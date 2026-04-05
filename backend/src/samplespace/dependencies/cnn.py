from fastapi import Request

from samplespace.ml.model import SampleCNN


def get_cnn_model(request: Request) -> SampleCNN | None:
    return getattr(request.app.state, "cnn_model", None)
