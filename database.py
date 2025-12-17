import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )
    messages = Column(Text, nullable=False, default="[]")


class ChatDatabase:
    def __init__(self, db_path: str = "/data/chats.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"[ChatDB] Database initialized: {db_path}")

    def create_chat(self, user_id: str, title: str = "Новый чат") -> int:
        session = self.Session()
        try:
            chat = Chat(user_id=user_id, title=title, messages="[]")
            session.add(chat)
            session.commit()
            chat_id = chat.id
            logger.info(f"[ChatDB] Created chat {chat_id} for user {user_id}")
            return chat_id
        finally:
            session.close()

    def get_user_chats(self, user_id: str) -> List[dict]:
        session = self.Session()
        try:
            chats = (
                session.query(Chat)
                .filter(Chat.user_id == user_id)
                .order_by(Chat.updated_at.desc())
                .all()
            )
            result = [
                {
                    "id": chat.id,
                    "title": chat.title,
                    "created_at": chat.created_at.isoformat(),
                    "updated_at": chat.updated_at.isoformat(),
                    "message_count": len(json.loads(chat.messages)),
                }
                for chat in chats
            ]
            logger.info(f"[ChatDB] Retrieved {len(result)} chats for user {user_id}")
            return result
        finally:
            session.close()

    def get_chat(self, chat_id: int, user_id: str) -> Optional[dict]:
        session = self.Session()
        try:
            chat = (
                session.query(Chat)
                .filter(Chat.id == chat_id, Chat.user_id == user_id)
                .first()
            )
            if chat:
                return {
                    "id": chat.id,
                    "title": chat.title,
                    "created_at": chat.created_at.isoformat(),
                    "updated_at": chat.updated_at.isoformat(),
                    "messages": json.loads(chat.messages),
                }
            return None
        finally:
            session.close()

    def get_chat_messages(self, chat_id: int, user_id: str) -> List[dict]:
        session = self.Session()
        try:
            chat = (
                session.query(Chat)
                .filter(Chat.id == chat_id, Chat.user_id == user_id)
                .first()
            )
            if chat:
                return json.loads(chat.messages)
            return []
        finally:
            session.close()

    def add_message(self, chat_id: int, user_id: str, role: str, content: str):
        session = self.Session()
        try:
            chat = (
                session.query(Chat)
                .filter(Chat.id == chat_id, Chat.user_id == user_id)
                .first()
            )
            if chat:
                messages = json.loads(chat.messages)
                messages.append(
                    {
                        "role": role,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                chat.messages = json.dumps(messages, ensure_ascii=False)
                chat.updated_at = datetime.now(timezone.utc)
                session.commit()
                logger.info(f"[ChatDB] Added {role} message to chat {chat_id}")
        finally:
            session.close()

    def update_chat_title(self, chat_id: int, user_id: str, title: str):
        session = self.Session()
        try:
            chat = (
                session.query(Chat)
                .filter(Chat.id == chat_id, Chat.user_id == user_id)
                .first()
            )
            if chat:
                chat.title = title
                chat.updated_at = datetime.now(timezone.utc)
                session.commit()
                logger.info(f"[ChatDB] Updated title for chat {chat_id}")
        finally:
            session.close()

    def delete_chat(self, chat_id: int, user_id: str):
        session = self.Session()
        try:
            chat = (
                session.query(Chat)
                .filter(Chat.id == chat_id, Chat.user_id == user_id)
                .first()
            )
            if chat:
                session.delete(chat)
                session.commit()
                logger.info(f"[ChatDB] Deleted chat {chat_id}")
        finally:
            session.close()
