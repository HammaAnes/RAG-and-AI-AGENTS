import streamlit as st
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import numpy as np

# === Load Vector DB Once ===
@st.cache_resource
def load_vdb():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("new_faiss_book_db", embeddings, allow_dangerous_deserialization=True)


# === Unified LLM Call Function (Fixed) ===
def call_llm(llm, prompt):
    try:
        response = llm(prompt)
        return response.strip() if isinstance(response, str) else "No valid response"
    except Exception as e:
        return f"Error: {str(e)}"


# === Load LLM Once ===
@st.cache_resource
def load_llm(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.5,
        top_p=0.95,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=pipe)


# === Prompt Templates ===
MATH_PROMPT = """
You are a math expert. Solve the following problem step-by-step.

Context: {context}

Question: {query}

Instructions:
1. Analyze the question carefully.
2. Show your work clearly.
3. Provide the final answer boxed.

Solution:
"""

AGENTIC_PLAN_PROMPT = """Plan how to answer the following question:
{query}

Steps:"""

# === Additional Prompt Templates ===


BRANCH_PROMPT = """
Given the question below, generate multiple related sub-questions or reformulations:
Question: {query}

Reformulated questions:
"""

def basic_rag(query, retriever, llm):
    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content[:300] for doc in context_docs])
    prompt = MATH_PROMPT.format(context=context, query=query)
    response = call_llm(llm, prompt)
    return response, context_docs


def self_rag(query, retriever, llm):
    decision_prompt = f"""
    Given the question below, decide whether to retrieve information or answer directly.
    Use [[Retrieve]] if needed, otherwise respond directly.

    Question: {query}
    """
    decision = call_llm(llm, decision_prompt)
    if "[[Retrieve]]" in decision:
        response, docs = basic_rag(query, retriever, llm)
    else:
        response = call_llm(llm, query)
        docs = []
    return response, docs


def adaptive_rag(query, retriever, llm):
    k = 3 if len(query.split()) < 10 else 5
    retriever.search_kwargs["k"] = k
    return basic_rag(query, retriever, llm)


def corrective_rag(query, retriever, llm):
    initial_answer, docs = basic_rag(query, retriever, llm)

    judge_prompt = f"""
    Is the following answer correct? If not, improve it.

    Question: {query}
    Answer: {initial_answer}

    Feedback and improved answer:
    """
    feedback = call_llm(llm, judge_prompt)

    refine_prompt = f"""
    You were asked to solve this math problem:

    {query}

    Your first attempt was:

    {initial_answer}

    Feedback said:

    {feedback}

    Now, rewrite the solution incorporating that feedback.
    """
    refined_answer = call_llm(llm, refine_prompt)

    return refined_answer.strip(), docs


def agentic_rag(query, retriever, llm):
    plan_prompt = AGENTIC_PLAN_PROMPT.format(query=query)
    plan = call_llm(llm, plan_prompt)

    if "retrieve" in plan.lower() or "search" in plan_prompt.lower():
        return basic_rag(query, retriever, llm)
    else:
        response = call_llm(llm, query)
        return response, []


def branched_rag(query, retriever, llm):
    branch_text = call_llm(llm, BRANCH_PROMPT.format(query=query))
    branches = [b.strip() for b in branch_text.split("\n") if b.strip()]
    all_docs = []
    context_parts = []

    for branch in branches:
        if branch.strip():
            docs = retriever.get_relevant_documents(branch.strip())
            all_docs.extend(docs)
            context_parts.extend([doc.page_content[:300] for doc in docs])

    context = "\n".join(set(context_parts))  # deduplicate
    prompt = MATH_PROMPT.format(context=context, query=query)
    response = call_llm(llm, prompt)
    return response, all_docs


def rag_with_memory(query, retriever, llm, memory=None):
    if memory is None:
        memory = []

    full_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in memory[-3:]])  # Use last 3 interactions
    extended_query = f"Based on previous questions:\n{full_context}\n\nNow answer this question:\n{query}"

    docs = retriever.get_relevant_documents(extended_query)
    context = "\n".join([doc.page_content[:300] for doc in docs])
    prompt = MATH_PROMPT.format(context=context, query=extended_query)
    response = call_llm(llm, prompt)

    memory.append((query, response))  # Update memory
    return response, docs, memory

# === Streamlit UI ===
st.set_page_config(page_title="RAG Comparison Tool", layout="wide")
st.title("🧮 RAG Methods Comparative Analysis")
#
# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose Model", ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    compare_all = st.checkbox("Compare All RAG Types", value=True)

MATH_PROMPTS = [
    "Explain the Pythagorean Theorem.",
    "Solve for x: 3x + 4 = 19",
    "What is the area of a circle with radius 7?",
    "What is the difference between mean, median, and mode?",
    "What is 6 × 7?",
    "A triangle has sides of length 5 cm, 12 cm, and 13 cm. Is it a right triangle? Explain."
]

selected_prompt = st.selectbox("Select a Math Prompt", MATH_PROMPTS)

if st.button("🚀 Run Comparison"):
    with st.spinner("Running all RAG types..."):

        db = load_vdb()
        retriever = db.as_retriever()
        llm = load_llm(model_choice)

        # Initialize memory for stateful RAG
        memory = []

        rag_methods = {
            "Branched RAG": lambda q, r, l: branched_rag(q, r, l),
            "RAG with Memory": lambda q, r, l: rag_with_memory(q, r, l, memory)[0:2],  # Ignore updated memory
            "Basic RAG": basic_rag,
            "Adaptive RAG": adaptive_rag,
            "Self-RAG": self_rag,
            "Corrective RAG": corrective_rag,
            "Agentic RAG": agentic_rag
        }

        results = []

        for name, method in rag_methods.items():
            start_time = time.time()
            try:
                response, docs = method(selected_prompt, retriever, llm)
                retrieval_used = "Yes" if docs else "No"
                context = "\n\n".join([doc.page_content[:200] for doc in docs]) if docs else "N/A"
                elapsed = round(time.time() - start_time, 2)
            except Exception as e:
                response = str(e)
                retrieval_used = "Error"
                context = ""
                elapsed = "Error"

            results.append({
                "Method": name,
                "Used Retrieval": retrieval_used,
                "Retrieved Context": context,
                "Response": response.strip(),
                "Time (s)": elapsed
            })

        df = pd.DataFrame(results)
        df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce")
        df["Time (s)"] = df["Time (s)"].fillna(9999)  # Replace invalid times

    st.markdown("### 📊 Results Table")
    st.dataframe(
    df[["Method", "Used Retrieval", "Time (s)"]]
    .style.highlight_min(subset=["Time (s)"], color='lightgreen', axis=0)
    )

    st.markdown("### 📄 Full Responses")
    for res in results:
        with st.expander(f"{res['Method']} ({res['Used Retrieval']}, {res['Time (s)']}s)"):
            st.markdown("#### Retrieved Context:")
            st.text(res["Retrieved Context"])
            st.markdown("#### Final Response:")
            st.write(res["Response"])