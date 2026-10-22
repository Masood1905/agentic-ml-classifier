# app.py
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from agent import MODEL_NAME, executor  # noqa: E402

EXAMPLES = {
    "Credit card fraud (a real fraud case)": (
        "Check this credit card transaction for fraud: 57007.0, -1.2712, 2.4627, -2.8514, 2.3245, -1.3722, "
        "-0.9482, -3.0652, 1.1669, -2.2688, -4.8811, 2.2551, -4.6864, 0.6524, -6.1743, 0.5944, -4.8497, "
        "-6.5365, -3.1191, 1.7155, 0.5605, 0.6529, 0.0819, -0.2213, -0.5236, 0.2242, 0.7563, 0.6328, 0.2502, 0.01"
    ),
    "Credit card fraud (a real legit case)": (
        "Is this transaction fraudulent? 160760.0, -0.6745, 1.4081, -1.1106, -1.3284, 1.389, -1.3084, 1.8859, "
        "-0.6142, 0.3117, 0.6508, -0.8578, -0.23, -0.1998, 0.2664, -0.0465, -0.7414, -0.6056, -0.3926, -0.1626, "
        "0.3943, 0.0801, 0.81, -0.2243, 0.7079, -0.1358, 0.0451, 0.5338, 0.2913, 23.0"
    ),
    "AAPL next-day direction": (
        "Apple closed today at 172.03 after opening at 171.19. High 173.32, low 170.99, volume 26,114,362, "
        "change +1.10 (0.644%), VWAP 172.2129, change over time 1.4938. Will it go up or down tomorrow?"
    ),
    "E-commerce return": (
        "A 34-year-old female Premium member visited from the Android App for 2,800 seconds over TCP, "
        "transferring 1,200 bytes. Her language is English and she paid by Credit Card. Will she return the purchase?"
    ),
}

st.set_page_config(page_title="Agentic ML Classifier", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Config")
    st.success(f"LLM: Groq · {MODEL_NAME}")
    st.markdown("**ML models (tools):**")
    st.markdown("- Credit card fraud · SVM\n- AAPL next-day direction · Random Forest\n- E-commerce returns · Logistic Regression")
    show_trace = st.toggle("Show agent reasoning trace", value=True)
    st.markdown("**Try an example:**")
    clicked = None
    for label, text in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            clicked = text

st.title("Agentic Multi-Model Classification System")
st.caption("LLM manager · 3 ML workers · tool-calling agent · LangChain + Groq")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Describe what you want to classify...") or clicked
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            result = executor.invoke({"input": prompt})
        answer = result["output"]

        if show_trace and result.get("intermediate_steps"):
            with st.expander("Agent reasoning trace", expanded=False):
                for i, step in enumerate(result["intermediate_steps"], 1):
                    st.markdown(f"**Step {i}: called `{step['tool']}`**")
                    st.json(step["input"])
                    st.code(step["output"], language="text")

        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
