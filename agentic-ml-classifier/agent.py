# agent.py
# The "LLM manager": a Groq-hosted LLM reads the user's request, decides which ML tool
# to call, extracts the inputs, ACTUALLY runs the tool, then explains the result in plain English.
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from ml_tools import ALL_TOOLS

load_dotenv()

# Groq shut down llama-3.3-70b-versatile (the original model) on 16 Aug 2026; gpt-oss-120b is its recommended replacement.
# If Groq retires this one too, set GROQ_MODEL in your .env to a current model from console.groq.com/docs/models
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are an AI assistant that routes classification requests to specialised machine learning models.
You have three tools:
1. classify_stock_direction: predicts whether Apple (AAPL) stock closes higher or lower tomorrow.
2. classify_ecommerce_return: predicts whether an online purchase will be returned.
3. classify_credit_card_fraud: detects whether a credit card transaction is fraudulent.

Rules:
- Pick the right tool based on the user's request and extract the input values from their message.
- NEVER invent or guess a prediction yourself. Every prediction must come from a tool result.
- If required values are missing, ask the user for them instead of making them up.
- After the tool returns, explain the prediction in one or two clear sentences, and repeat any caution the tool gives.
- If the request is not about one of these three tasks, explain what you can help with."""

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
_llm_with_tools = None


def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        _llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


def run_agent(user_input: str, max_steps: int = 4) -> dict:
    """Reason -> act (call a tool) -> observe the result -> answer. Returns the answer and every tool call."""
    if not os.getenv("GROQ_API_KEY"):
        return {"output": "GROQ_API_KEY is missing. Add it to the .env file (see README).", "intermediate_steps": []}

    llm = _get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
    steps = []

    for _ in range(max_steps):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:  # no tool needed -> this is the final answer
            return {"output": ai_msg.content, "intermediate_steps": steps}

        for call in ai_msg.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            try:
                result = tool.invoke(call["args"]) if tool else f"Error: unknown tool '{call['name']}'"
            except Exception as e:  # e.g. wrong input format: send the error back so the LLM can fix it
                result = f"Error: {e}"
            steps.append({"tool": call["name"], "input": call["args"], "output": str(result)})
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"output": "Sorry, I couldn't complete this request. Please check the input values and try again.",
            "intermediate_steps": steps}


class AgentExecutor:
    """Small wrapper so app.py can call executor.invoke({"input": ...})."""

    def invoke(self, inputs: dict) -> dict:
        return run_agent(inputs.get("input", ""))


executor = AgentExecutor()

if __name__ == "__main__":
    print(run_agent("Is this AAPL day bullish for tomorrow? open 170.5, high 172.1, low 169.8, close 171.9, "
                    "volume 25000000, change 1.4, changePercent 0.82, vwap 171.2, changeOverTime 1.49")["output"])
