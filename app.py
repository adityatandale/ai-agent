import json
import os
import tempfile

import streamlit as st

from agent import Agent, ChatError, Database, DemoLLM, LLMClient, LLMConfig, build_default_tools

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")

EXAMPLES = ["What is 17 * 23 + 5?", "What time is it in Tokyo?", "What's the weather in Pune?",
            "Which product sold the most units?", "Add a task: buy milk", "List my tasks", "Delete task 1",
            "Remember that I like tea"]


def get_config():
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    return LLMConfig.from_env(secrets)


@st.cache_resource
def real_llm(api_key, base_url, model):
    return LLMClient(LLMConfig(api_key=api_key, base_url=base_url, model=model))


cfg = get_config()
ss = st.session_state

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Agent settings")
    options = (["LLM agent (real model)"] if cfg else []) + ["Demo mode (rule-based, no LLM)"]
    mode = st.radio("Brain", options, help="Demo mode needs no API key but only understands a few request types.")
    max_steps = st.slider("Max steps per request", 1, 10, 6, help="Safety limit on tool-use loops.")
    history_turns = st.slider("Short-term memory (turns)", 1, 12, 6)
    st.caption("Destructive actions (delete) always ask for your approval.")

llm_key = (mode, cfg.model if cfg else None)
if ss.get("llm_key") != llm_key or "agent" not in ss:           # (re)build the agent when the brain changes
    ss["db"] = Database(os.path.join(tempfile.mkdtemp(), "agent.db"))
    llm = real_llm(cfg.api_key, cfg.base_url, cfg.model) if mode.startswith("LLM") else DemoLLM()
    ss["agent"] = Agent(llm, build_default_tools(ss["db"]), ss["db"])
    ss["llm_key"], ss["chat"], ss["pending"] = llm_key, [], None
agent: Agent = ss["agent"]
db: Database = ss["db"]
agent.max_steps, agent.history_turns = max_steps, history_turns

with st.sidebar:
    st.divider()
    st.subheader("🧠 State")
    st.markdown("**Tasks**")
    tasks = db.list_tasks()
    st.write("\n".join(f"- #{t['id']} {t['title']}" for t in tasks) if tasks else "_none_")
    st.markdown("**Long-term memory (facts)**")
    facts = db.facts()
    st.write("\n".join(f"- {f}" for f in facts) if facts else "_none_")
    st.caption(f"Model: `{agent.llm.model}` · messages in context: {len(agent.messages)}")
    if st.button("🗑️ Reset conversation", use_container_width=True):
        agent.reset()
        ss["chat"], ss["pending"] = [], None
        st.rerun()


# ---------- helpers ----------
def render_events(events):
    if not events:
        return
    with st.expander(f"🔧 Agent trace: {len(events)} tool call(s)"):
        for i, e in enumerate(events, 1):
            badge = "✅" if e.ok else "⚠️"
            appr = {True: " · approved", False: " · rejected", None: ""}[e.approved]
            st.markdown(f"**{i}. {badge} `{e.name}`**{appr} · {e.seconds * 1000:.0f} ms")
            st.code(json.dumps(e.arguments), language="json")
            st.text(e.result)


def handle(result):
    if result.status == "needs_confirmation":
        ss["pending"] = result
    else:
        ss["pending"] = None
        ss["chat"].append({"role": "assistant", "content": result.answer, "events": result.events,
                           "flag": result.status})


# ---------- main ----------
st.title("🤖 AI Agent")
st.caption("Understand task → Decide action → Use tool → Process result → Respond")
if mode.startswith("Demo"):
    st.info("Demo mode: a rule-based planner (not an AI model) drives the same tools, safety checks and UI. "
            "Add `LLM_API_KEY` to unlock a real LLM agent. See the README.")
    st.write("Try:", "  ·  ".join(f"`{x}`" for x in EXAMPLES))

for m in ss["chat"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("flag") == "max_steps":
            st.warning("Stopped by the step limit.")
        render_events(m.get("events"))

pending = ss.get("pending")
if pending:
    p = pending.pending
    with st.chat_message("assistant"):
        render_events(pending.events)
        st.warning(f"**Approval needed:** the agent wants to run `{p.call.name}` with `{json.dumps(p.arguments)}`.")
        c1, c2 = st.columns(2)
        decision = None
        if c1.button("✅ Approve", use_container_width=True):
            decision = True
        if c2.button("❌ Reject", use_container_width=True):
            decision = False
        if decision is not None:
            try:
                with st.spinner("Continuing..."):
                    handle(agent.resume(decision))
            except ChatError as e:
                ss["pending"] = None
                ss["chat"].append({"role": "assistant", "content": f"⚠️ {e.user_message}"})
            st.rerun()

text = st.chat_input("Ask the agent to do something...", disabled=bool(pending))
if text:
    ss["chat"].append({"role": "user", "content": text})
    try:
        with st.spinner("Working..."):
            handle(agent.run(text))
    except ChatError as e:
        ss["chat"].append({"role": "assistant", "content": f"⚠️ {e.user_message}"})
    st.rerun()
