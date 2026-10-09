from typing import Dict, List
import json

class InMemorySessionStore:
    def __init__(self):
        # In production, swap this with PostgreSQL or Redis
        self.sessions: Dict[str, List[dict]] = {}

    def get_history(self, session_id: str) -> List[dict]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI Agent for Finance & HR operations. "
                        "Always validate parameters before invoking tools."
                    )
                }
            ]
        return self.sessions[session_id]

    def add_message(self, session_id: str, message: dict):
        self.sessions[session_id].append(message)

session_db = InMemorySessionStore()