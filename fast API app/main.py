import json
import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import cohere

app = FastAPI(title="Enterprise Agentic Service")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Tools & Schemas definition
def get_employee_payroll(employee_id: str) -> dict:
    database = {
        "EMP-101": {"name": "Sara Alhaddadi", "role": "Senior Engineer", "salary": 18500, "currency": "SAR"},
        "EMP-102": {"name": "Ahmed Ali", "role": "Backend Developer", "salary": 12000, "currency": "SAR"}
    }
    return database.get(employee_id, {"error": "Employee not found"})

TOOL_REGISTRY = {"get_employee_payroll": get_employee_payroll}

tools_schema = [{
    "type": "function",
    "function": {
        "name": "get_employee_payroll",
        "description": "Fetches payroll details for a given employee ID.",
        "parameters": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"]
        }
    }
}]

@app.get("/api/chat/stream")
async def stream_agent_execution(request: Request, query: str):
    """Streams thought, tool execution, and response steps to the frontend."""
    async def event_generator():
        messages = [
            {"role": "system", "content": "You are an enterprise AI assistant."},
            {"role": "user", "content": query}
        ]
        
        yield {"event": "status", "data": json.dumps({"message": "Agent initialized..."})}
        
        for iteration in range(5):
            if await request.is_disconnected():
                break

            # Non-blocking async API call execution
            response = await asyncio.to_thread(
                co.chat,
                model="command-r-plus-08-2024",
                messages=messages,
                tools=tools_schema
            )
            
            message_output = response.message
            messages.append(message_output)
            tool_calls = getattr(message_output, "tool_calls", None)

            if tool_calls:
                for call in tool_calls:
                    func_name = call.function.name
                    func_args = json.loads(call.function.arguments)
                    
                    # Notify UI that a tool is executing
                    yield {
                        "event": "tool_call",
                        "data": json.dumps({"tool": func_name, "arguments": func_args})
                    }
                    
                    # Execute backend tool
                    tool_result = TOOL_REGISTRY[func_name](**func_args)
                    
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({"tool": func_name, "result": tool_result})
                    }

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result)
                    })
            else:
                final_text = message_output.content[0].text
                yield {
                    "event": "final_answer",
                    "data": json.dumps({"response": final_text})
                }
                break

    return EventSourceResponse(event_generator())