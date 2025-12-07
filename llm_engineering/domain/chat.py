from datetime import datetime, timezone
import uuid
from pydantic import UUID4, BaseModel, Field
from typing import List, Optional

from llm_engineering.domain.orm.nosql import NoSQLBaseDocument
from llm_engineering.domain.types import DataCategory, Role


class Message(BaseModel):
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_content(cls, content: str):
        return cls(role=Role.USER, content=content)


class Conversation(NoSQLBaseDocument):
    user_id: str | None = None
    messages: list[Message] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "chat_sessions"

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_context(self, last_n: int = 5) -> list[Message]:
        return self.messages[-last_n:] if len(self.messages) > last_n else self.messages

    def to_langchain_format(self, last_n: int = 5) -> list[dict]:
        context_messages = self.get_context(last_n=last_n)
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in context_messages
        ]

