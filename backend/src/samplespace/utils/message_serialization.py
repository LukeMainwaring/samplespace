from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ToolReturnPart
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessage
from pydantic_ai.ui.vercel_ai.response_types import DataChunk


def prepare_messages_for_storage(
    messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    return result


def deserialize_messages(message_data: list[dict[str, Any]]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(message_data)


def _restore_metadata_chunks(messages: list[ModelMessage]) -> None:
    """Reconstruct DataChunk objects from plain dicts after deserialization.

    ToolReturnPart.metadata is typed as Any, so DataChunk instances become
    plain dicts through the JSON serialization round-trip. iter_metadata_chunks()
    relies on isinstance() checks, so we need to restore the original types.
    """
    for msg in messages:
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart) or part.metadata is None:
                continue
            if isinstance(part.metadata, dict):
                if isinstance(part.metadata.get("type"), str) and part.metadata["type"].startswith("data-"):
                    part.metadata = DataChunk(**part.metadata)
            elif isinstance(part.metadata, list):
                part.metadata = [
                    DataChunk(**item)
                    if isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"].startswith("data-")
                    else item
                    for item in part.metadata
                ]


def dump_messages_for_frontend(message_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_messages = deserialize_messages(message_data)
    _restore_metadata_chunks(model_messages)
    ui_messages = VercelAIAdapter.dump_messages(model_messages)
    return [msg.model_dump(mode="json", by_alias=True) for msg in ui_messages]


def extract_latest_user_text(messages: list[UIMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    return part.text
    return ""
