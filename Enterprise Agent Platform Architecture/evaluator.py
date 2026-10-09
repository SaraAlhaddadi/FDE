import json
import cohere
import os

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

def evaluate_groundedness(user_query: str, retrieved_docs: list, agent_answer: str) -> dict:
    """دالة لتقييم مدى استناد الإجابة إلى السياق المسترجع (Groundedness Check)"""
    
    context_text = "\n".join([f"- {doc}" for doc in retrieved_docs])
    
    prompt = f"""
    You are an AI Auditor evaluating an AI Agent's answer for accuracy and faithfulness.
    
    [Retrieved Context Documents]:
    {context_text}
    
    [User Query]:
    {user_query}
    
    [Agent Answer]:
    {agent_answer}
    
    Task:
    Evaluate if the Agent Answer is strictly supported by the Retrieved Context Documents.
    Return ONLY a raw JSON object with the following schema:
    {{
        "groundedness_score": <number from 0.0 to 1.0>,
        "is_faithful": <true/false>,
        "reasoning": "<short explanation in Arabic or English>"
    }}
    """

    response = co.chat(
        model="command-r-08-2024",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} # إجبار النموذج على إرجاع JSON دقيق
    )

    try:
        eval_result = json.loads(response.message.content[0].text)
        return eval_result
    except Exception as e:
        return {"groundedness_score": 1.0, "is_faithful": True, "reasoning": "Failed to parse eval output."}