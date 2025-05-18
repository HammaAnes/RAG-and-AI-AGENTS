import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict


# -----------------------------
# Setup
# -----------------------------

# Use different model names based on framework
llm_langchain = OllamaLLM(model="llama3")             # For LangChain and LangGraph
llm_crewai = OllamaLLM(model="ollama/llama3")        # For CrewAI (uses LiteLLM under the hood)

# Tasks
rephrase_task = "Rephrase this sentence in a more formal way: 'Hey, just letting you know I won’t make it to the meeting later — something came up.'"
summary_task = """
Summarize this paragraph in 2–3 bullet points:
Artificial intelligence is transforming industries by automating tasks, improving decision-making, and creating new business opportunities. From healthcare to finance, AI technologies are helping companies analyze data more efficiently and deliver better services. However, ethical concerns around bias and job displacement continue to be major challenges.
"""
grammar_task = "Correct the grammar in this sentence: 'She don’t like going to the park because it make her tired.'"

tasks = {
    "Rephrasing": rephrase_task,
    "Summarization": summary_task,
    "Grammar Correction": grammar_task
}

# -----------------------------
# Framework Implementations
# -----------------------------


# --- 1. LangChain ---
def run_langchain(task):
    prompt = PromptTemplate.from_template("{input}")
    chain = prompt | llm_langchain
    return chain.invoke({"input": task})


# --- 2. LangGraph ---
def run_langgraph(task):
    class GraphState(TypedDict):
        input: str
        output: str

    def process_input(state: GraphState):
        response = llm_langchain.invoke(state["input"])
        return {"output": response}

    workflow = StateGraph(GraphState)
    workflow.add_node("process", process_input)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    app = workflow.compile()
    result = app.invoke({"input": task})
    return result["output"]


# --- 3. CrewAI ---
def run_crewai(task):
    agent = Agent(
        role="Professional Assistant",
        goal="Help rephrase, summarize, and correct grammar in text.",
        backstory="You're an expert at writing and editing text professionally.",
        llm=llm_crewai,
        verbose=False
    )

    task_obj = Task(
        description=task,
        expected_output="The requested output",
        agent=agent  # ✅ This is required to avoid "missing agent" error
    )

    crew = Crew(
        agents=[agent],
        tasks=[task_obj],
        verbose=False
    )

    result = crew.kickoff()
    return result


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="🧠 Agent Comparison App", layout="wide")

st.title("🧠 Compare LangChain, LangGraph, and CrewAI")
st.subheader("Using Local LLM (Llama3 via Ollama)")

selected_task = st.selectbox("Select a task:", list(tasks.keys()))
task_prompt = st.text_area("Task Prompt:", value=tasks[selected_task], height=200)

if st.button("Run Comparison"):
    with st.spinner("Running LangChain..."):
        try:
            lc_result = run_langchain(task_prompt)
        except Exception as e:
            lc_result = f"LangChain Error: {e}"

    with st.spinner("Running LangGraph..."):
        try:
            lg_result = run_langgraph(task_prompt)
        except Exception as e:
            lg_result = f"LangGraph Error: {e}"

    with st.spinner("Running CrewAI..."):
        try:
            crew_result = run_crewai(task_prompt)
        except Exception as e:
            crew_result = f"CrewAI Error: {e}"

    # Show Results
    st.markdown("## ✅ Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🦜 LangChain")
        st.info(lc_result)

    with col2:
        st.markdown("### 📊 LangGraph")
        st.info(lg_result)

    with col3:
        st.markdown("### 👥 CrewAI")
        st.info(crew_result)