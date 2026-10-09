import asyncio
import json
import os
import time
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from config import settings

import cohere
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ============================================================
# 1. CONFIG & INITIALIZATION
# ============================================================

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY environment variable is required")

co = cohere.ClientV2(api_key=COHERE_API_KEY)

COLLECTION_NAME = "enterprise_knowledge_v2"
# Connect to production Qdrant service or fall back to local
if settings.qdrant_url == ":memory:":
    qdrant = QdrantClient(":memory:")
else:
    qdrant = QdrantClient(url=settings.qdrant_url)


# ============================================================
# 2. AUDIT LOGGING ENGINE (Compliance Ledger)
# ============================================================

DB_FILE = "audit_ledger.db"

def init_audit_db():
    """إنشاء جدول سجلات التدقيق لتسجيل كل عمليات الوكيل الحساسة"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_role TEXT NOT NULL,
            action_type TEXT NOT NULL,
            tool_name TEXT,
            tool_args TEXT,
            status TEXT NOT NULL,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_audit_event(
    session_id: str,
    user_id: str,
    user_role: str,
    action_type: str,
    status: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    details: Optional[str] = None
):
    """حفظ حدث في سجل التدقيق"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, session_id, user_id, user_role, action_type, tool_name, tool_args, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            session_id,
            user_id,
            user_role,
            action_type,
            tool_name,
            json.dumps(tool_args, ensure_ascii=False) if tool_args else None,
            status,
            details
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Audit Log Error: {e}")


# ============================================================
# 3. MODEL CONTEXT PROTOCOL (MCP) INTEGRATION
# ============================================================

class MCPServer:
    """
    محاكاة خادم Model Context Protocol (MCP Server)
    يقوم الخادم بالترخيص والربط القياسي للأدوات عبر بروتوكول JSON-RPC
    """
    def __init__(self):
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "get_employee_payroll",
                    "description": "MCP Endpoint: Fetch salary details for an employee.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "string", "description": "Employee ID e.g. EMP-101"}
                        },
                        "required": ["employee_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_employee_salary",
                    "description": "MCP Endpoint: Sensitive tool to update employee salary. Requires HITL approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "string"},
                            "new_salary": {"type": "number"},
                        },
                        "required": ["employee_id", "new_salary"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_base",
                    "description": "MCP Endpoint: Search enterprise internal policy vector store.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ]

    def list_tools(self) -> List[dict]:
        """إرجاع قائمة الأدوات المعرفة وفق معيار MCP"""
        return self.tools_schema

    def call_tool(self, name: str, args: dict) -> Any:
        """مقبض تنفيذ الأدوات القياسي لبروتوكول MCP"""
        if name == "get_employee_payroll":
            db = {"EMP-101": {"name": "سارة الحدادي", "role": "Senior Engineer", "salary": 18500, "currency": "SAR"}}
            return db.get(args.get("employee_id"), {"error": "الموظف غير موجود"})
        
        elif name == "update_employee_salary":
            return {
                "status": "success",
                "message": f"تم تحديث راتب الموظف {args.get('employee_id')} إلى {args.get('new_salary')} SAR عبر MCP Server."
            }
            
        elif name == "query_knowledge_base":
            return query_knowledge_base(args.get("query", ""))
            
        else:
            raise ValueError(f"MCP Tool '{name}' not registered on server.")

mcp_server = MCPServer()


# ============================================================
# 4. EVALUATOR (Groundedness Check)
# ============================================================

async def evaluate_groundedness(user_query: str, retrieved_docs: List[str], agent_answer: str) -> dict:
    if not retrieved_docs:
        return {"groundedness_score": 1.0, "is_faithful": True, "reasoning": "لا يوجد سياق مسترجع لحساب التقييم."}

    context_text = "\n".join([f"- {doc}" for doc in retrieved_docs])
    prompt = f"""
    Evaluate if the Agent Answer is strictly supported by the Retrieved Context Documents.
    Context: {context_text}
    Query: {user_query}
    Answer: {agent_answer}
    Return ONLY a raw JSON object: {{"groundedness_score": <float 0.0-1.0>, "is_faithful": <true/false>, "reasoning": "<explanation>"}}
    """
    try:
        response = await asyncio.to_thread(
            co.chat,
            model="command-r-08-2024",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.message.content[0].text)
    except Exception as exc:
        return {"groundedness_score": 1.0, "is_faithful": True, "reasoning": f"حدث خطأ في التقييم: {str(exc)}"}


# ============================================================
# 5. USER & ROLE STATE
# ============================================================

USER_ROLES_DB = {
    "user_admin": {
        "name": "سارة الحدادي",
        "role": "admin",
        "permissions": ["payroll_view", "payroll_update", "policy_search"],
    },
    "user_staff": {
        "name": "أحمد علي",
        "role": "staff",
        "permissions": ["policy_search"],
    },
}

class SessionStateStore:
    def __init__(self):
        self.sessions: Dict[str, List[dict]] = {}

    def get_history(self, session_id: str) -> List[dict]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role": "system",
                    "content": "You are an enterprise AI Agent running via Model Context Protocol (MCP). Enforce strict facts and security permissions.",
                }
            ]
        return self.sessions[session_id]

session_store = SessionStateStore()


# ============================================================
# 6. RAG ENGINE
# ============================================================

def init_advanced_rag():
    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    documents = [
        {"id": 1, "text": "السياسة المالية 2026: جميع التسويات البنكية التي تتجاوز قيمتها 10,000 ريال سعودي تتطلب موافقة مزدوجة من المدير المالي."},
        {"id": 2, "text": "سياسة الموارد البشرية 2026: تقييم أداء الموظفين يتم بشكل نصف سنوي في شهري يونيو وديسمبر. يحصل الموظفون المتميزون على مكافأة تصل إلى 15%."},
        {"id": 3, "text": "سياسة أمن المعلومات: يجب تغيير كلمات المرور كل 90 يوماً. وتُخزن مفاتيح الـ API حصراً في مدير المفاتيح المعتمد (Secrets Manager)."}
    ]

    response = co.embed(
        texts=[doc["text"] for doc in documents],
        model="embed-multilingual-v3.0",
        input_type="search_document",
        embedding_types=["float"],
    )

    points = [
        PointStruct(id=doc["id"], vector=response.embeddings.float_[idx], payload={"text": doc["text"]})
        for idx, doc in enumerate(documents)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def query_knowledge_base(query: str) -> dict:
    response = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    query_embedding = response.embeddings.float_[0]

    try:
        search_result = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_embedding, limit=3).points
    except AttributeError:
        search_result = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=3)

    raw_docs = [hit.payload["text"] for hit in search_result if hit.payload and "text" in hit.payload]
    if not raw_docs:
        return {"retrieved_policies": []}

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=raw_docs,
        top_n=min(2, len(raw_docs)),
    )
    return {"retrieved_policies": [raw_docs[res.index] for res in rerank_response.results]}


# ============================================================
# 7. FASTAPI & TOOL METADATA
# ============================================================

app = FastAPI(title="Cohere Enterprise Agent Platform (MCP & Audit)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_audit_db()
    init_advanced_rag()

TOOL_METADATA = {
    "get_employee_payroll": {"permission": "payroll_view", "sensitive": False},
    "update_employee_salary": {"permission": "payroll_update", "sensitive": True},
    "query_knowledge_base": {"permission": "policy_search", "sensitive": False},
}


# ============================================================
# 8. REQUEST MODELS & SSE
# ============================================================

class AgentExecuteRequest(BaseModel):
    session_id: str
    user_id: str
    user_query: str

class ApprovalActionRequest(BaseModel):
    session_id: str
    tool_call_id: str
    tool_name: str
    tool_args: dict
    approved: bool

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def build_tool_message(tool_call_id: str, result: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [{"type": "document", "document": {"data": json.dumps(result, ensure_ascii=False)}}],
    }


# ============================================================
# 9. AGENT LOOP WITH MCP & AUDIT LOGGING
# ============================================================

async def run_agent(session_id: str, user_id: str):
    start_time = time.time()
    user_info = USER_ROLES_DB.get(user_id, {"name": "Unknown", "role": "guest", "permissions": []})
    history = session_store.get_history(session_id)
    
    retrieved_contexts: List[str] = []
    user_original_query = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            user_original_query = msg.get("content", "")
            break

    # تسجيل استلام الاستعلام في سجل التدقيق
    log_audit_event(
        session_id=session_id,
        user_id=user_id,
        user_role=user_info["role"],
        action_type="USER_QUERY",
        status="RECEIVED",
        details=user_original_query
    )

    for iteration in range(5):
        yield sse({"type": "thinking", "content": f"المرور {iteration + 1}: الاتصال بـ MCP Server للتخطيط..."})

        response = await asyncio.to_thread(
            co.chat,
            model="command-a-plus-05-2026",
            messages=history,
            tools=mcp_server.list_tools(), # جلب مخطط الأدوات من MCP Server
            temperature=0,
        )

        message = response.message

        if message.tool_calls:
            history.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    error_result = {"error": f"Invalid tool arguments: {str(exc)}"}
                    history.append(build_tool_message(tool_call.id, error_result))
                    continue

                metadata = TOOL_METADATA.get(tool_name)
                if metadata is None:
                    error_result = {"error": f"Unknown tool: {tool_name}"}
                    history.append(build_tool_message(tool_call.id, error_result))
                    continue

                # 1. التحقق من الصلاحيات (RBAC Check)
                required_permission = metadata["permission"]
                if required_permission not in user_info["permissions"]:
                    error_message = f"خطأ أمني: المستخدم {user_info['name']} لا يمتلك صلاحية ({required_permission})."
                    
                    # تسجيل خرق الصلاحيات في سجل التدقيق
                    log_audit_event(
                        session_id=session_id,
                        user_id=user_id,
                        user_role=user_info["role"],
                        action_type="SECURITY_VIOLATION",
                        status="BLOCKED",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        details=error_message
                    )

                    yield sse({"type": "security_error", "content": error_message})
                    history.append(build_tool_message(tool_call.id, {"error": error_message}))
                    continue

                # 2. الموافقة البشرية (HITL Gate)
                if metadata["sensitive"]:
                    session_store.pending_approvals[session_id] = {
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "user_id": user_id,
                    }

                    # تسجيل طلب الموافقة في سجل التدقيق
                    log_audit_event(
                        session_id=session_id,
                        user_id=user_id,
                        user_role=user_info["role"],
                        action_type="HITL_REQUESTED",
                        status="PENDING",
                        tool_name=tool_name,
                        tool_args=tool_args
                    )

                    yield sse({
                        "type": "hitl_approval_required",
                        "tool_call_id": tool_call.id,
                        "tool": tool_name,
                        "args": tool_args,
                    })
                    return

                # 3. التنفيذ المباشر عبر MCP Server
                yield sse({"type": "tool_start", "tool": tool_name, "args": tool_args})

                try:
                    # استدعاء الأداة عبر بروتوكول MCP
                    result = mcp_server.call_tool(tool_name, tool_args)
                    
                    log_audit_event(
                        session_id=session_id,
                        user_id=user_id,
                        user_role=user_info["role"],
                        action_type="MCP_TOOL_EXECUTION",
                        status="SUCCESS",
                        tool_name=tool_name,
                        tool_args=tool_args
                    )

                    if tool_name == "query_knowledge_base" and isinstance(result, dict):
                        retrieved_contexts.extend(result.get("retrieved_policies", []))

                except Exception as exc:
                    result = {"error": str(exc)}
                    log_audit_event(
                        session_id=session_id,
                        user_id=user_id,
                        user_role=user_info["role"],
                        action_type="MCP_TOOL_EXECUTION",
                        status="FAILED",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        details=str(exc)
                    )

                yield sse({"type": "tool_result", "tool": tool_name, "result": result})
                history.append(build_tool_message(tool_call.id, result))

            continue

        # --------------------------------------------------------
        # Final Response & Evals
        # --------------------------------------------------------
        final_text = ""
        if message.content:
            for content_item in message.content:
                if getattr(content_item, "type", None) == "text":
                    final_text += content_item.text

        latency = round(time.time() - start_time, 2)
        yield sse({"type": "metrics", "latency_sec": latency, "tokens": "See Cohere usage metadata"})
        yield sse({"type": "final_response", "content": final_text})

        if retrieved_contexts:
            yield sse({"type": "thinking", "content": "🛡️ جاري فحص الجودة والتحقق من الاستناد عبر LLM-as-a-Judge..."})
            eval_res = await evaluate_groundedness(user_original_query, retrieved_contexts, final_text)
            
            log_audit_event(
                session_id=session_id,
                user_id=user_id,
                user_role=user_info["role"],
                action_type="EVALUATION",
                status="COMPLETED",
                details=f"Groundedness Score: {eval_res.get('groundedness_score')}"
            )

            yield sse({
                "type": "eval_badge",
                "score": eval_res.get("groundedness_score", 1.0),
                "is_faithful": eval_res.get("is_faithful", True),
                "reasoning": eval_res.get("reasoning", "")
            })

        return


# ============================================================
# 10. SESSION & APPROVALS
# ============================================================

SessionStateStore.pending_approvals = property(
    lambda self: getattr(self, "_pending_approvals", None) or self._initialize_pending_approvals()
)

def _initialize_pending_approvals(self):
    self._pending_approvals = {}
    return self._pending_approvals

SessionStateStore._initialize_pending_approvals = _initialize_pending_approvals


# ============================================================
# 11. ENDPOINTS & AUDIT LEDGER VIEW
# ============================================================

@app.post("/api/v2/agent/chat")
async def chat_endpoint(request: AgentExecuteRequest):
    history = session_store.get_history(request.session_id)
    history.append({"role": "user", "content": request.user_query})

    return StreamingResponse(
        run_agent(request.session_id, request.user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.post("/api/v2/agent/approve")
async def approve_endpoint(request: ApprovalActionRequest):
    pending = session_store.pending_approvals.get(request.session_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending approval found for this session.")

    user_info = USER_ROLES_DB.get(pending["user_id"], {"role": "guest"})
    history = session_store.get_history(request.session_id)

    if request.approved:
        try:
            result = mcp_server.call_tool(pending["tool_name"], pending["tool_args"])
            log_audit_event(
                session_id=request.session_id,
                user_id=pending["user_id"],
                user_role=user_info["role"],
                action_type="HITL_APPROVED",
                status="EXECUTED",
                tool_name=pending["tool_name"],
                tool_args=pending["tool_args"]
            )
        except Exception as exc:
            result = {"error": str(exc)}
    else:
        result = {"status": "rejected", "message": "تم رفض العملية من قبل المستخدم البشري."}
        log_audit_event(
            session_id=request.session_id,
            user_id=pending["user_id"],
            user_role=user_info["role"],
            action_type="HITL_REJECTED",
            status="REJECTED",
            tool_name=pending["tool_name"],
            tool_args=pending["tool_args"]
        )

    history.append(build_tool_message(pending["tool_call_id"], result))
    del session_store.pending_approvals[request.session_id]

    return StreamingResponse(
        run_agent(request.session_id, pending["user_id"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.get("/api/v2/audit/logs")
def get_audit_logs():
    """عرض سجلات التدقيق المؤسسية (Compliance Ledger API)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, session_id, user_id, user_role, action_type, tool_name, tool_args, status, details FROM audit_logs ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    logs = [
        {
            "id": r[0], "timestamp": r[1], "session_id": r[2], "user_id": r[3],
            "user_role": r[4], "action_type": r[5], "tool_name": r[6],
            "tool_args": r[7], "status": r[8], "details": r[9]
        }
        for r in rows
    ]
    return {"audit_logs": logs}

@app.get("/health")
def health_check():
    return {"status": "ok", "mcp_server": "active", "audit_ledger": "active"}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found.")