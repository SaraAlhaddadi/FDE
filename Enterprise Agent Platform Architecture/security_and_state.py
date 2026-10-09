from typing import Dict, List

USER_ROLES_DB = {
    "user_admin": {"name": "سارة الحدادي", "role": "admin", "permissions": ["payroll_view", "payroll_update", "policy_search"]},
    "user_staff": {"name": "أحمد علي", "role": "staff", "permissions": ["policy_search"]}
}

class SessionStateStore:
    def __init__(self):
        self.sessions: Dict[str, List[dict]] = {}

    def get_history(self, session_id: str) -> List[dict]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI Agent for Finance and HR operations. "
                        "Always maintain exact facts and enforce role authorization."
                    )
                }
            ]
        return self.sessions[session_id]

    def add_message(self, session_id: str, message: dict):
        if session_id in self.sessions:
            self.sessions[session_id].append(message)

session_store = SessionStateStore()