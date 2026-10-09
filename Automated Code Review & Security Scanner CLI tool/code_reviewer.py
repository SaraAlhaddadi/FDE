import os
import sys
import cohere

# Initialize Cohere Client (reads COHERE_API_KEY from environment)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

MODEL_NAME = "north-mini-code-1-0"

SYSTEM_PROMPT = """ You are an expert code reviewer and security auditor powered by North Mini Code.
Your job is to analyze code files provided by the user and produce a structured, actionable review:
1. **Summary**: Brief overview of what the code does.
2. **Security Vulnerabilities**: Identify any SQL injection, hardcoded secrets, unhandled errors, or unsafe calls.
3. **Optimizations & Best Practices**: Suggest improvements for performance, readability, and modern language features.
4. **Refactored Snippet**: Provide clean, corrected code blocks where applicable.
"""


def review_file(file_path: str):
    """Reads a target file and streams an automated code review using North Mini Code."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        code_content = f.read()

    print(f"\n🔍 Analyzing `{file_path}` with model `{MODEL_NAME}`...\n" + "-" * 50)

    # Call Cohere Chat V2 API with streaming response
    response = co.chat_stream(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Please review this code:\n\n```{file_path}\n{code_content}\n```",
            },
        ],
        temperature=0.2,  # Low temperature for precise code analysis
    )

    # Stream the output directly to stdout
    for event in response:
        if event.type == "content-delta":
            print(event.delta.message.content.text, end="", flush=True)

    print("\n" + "-" * 50 + "\n✅ Review complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code_reviewer.py <path_to_code_file>")
        sys.exit(1)

    target_file = sys.argv[1]
    review_file(target_file)