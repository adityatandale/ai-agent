# 🤖 AI Agent: Tools, Memory & Safe Automation

A from-scratch AI agent that **plans, calls tools, reads the results, and answers**, with short- and long-term memory, human approval for risky actions, and a trace UI that shows every step.

```
User request
   │
   ▼
1. Understand task ──► 2. Decide action (LLM picks a tool + arguments)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          3. Use tool (validated,    needs approval? ──► pause ──► Approve / Reject
             sandboxed, timed)
                 │
                 ▼
          4. Process result (fed back to the LLM) ──► repeat, up to max_steps
                 │
                 ▼
          5. Respond (final answer + visible tool trace)
```

**Live demo:** _optional: paste your Streamlit URL here_

## Chatbot vs agent
| | Chatbot | This agent |
|---|---|---|
| Output | Text from the model's memory | Actions plus text grounded in tool results |
| Steps | One reply per message | A loop: decide, act, observe, repeat |
| Data | Only what it was trained on | Live APIs and a database |
| State | Chat history | Chat history + long-term facts + pending approvals |
| Risk | Wrong text | Wrong *actions*, so guardrails matter |

## Tools
| Tool | What it does | Safety |
|---|---|---|
| `calculator` | Exact arithmetic | Parsed with `ast`, **no `eval`**; exponent and length caps |
| `get_current_time` | Time in any IANA timezone | Input validated |
| `get_weather` | Live weather (Open-Meteo, no key) | Fixed HTTPS hosts, timeout, errors become readable text |
| `query_database` | SQL over `customers`, `products`, `orders` | **Read-only** (see below) |
| `add_task` / `list_tasks` | To-do list in SQLite | Length-checked |
| `delete_task` | Remove a task | **Requires human approval** |
| `remember_fact` | Long-term memory | Length-checked |

## Memory and state
- **Short-term:** the conversation messages, trimmed to the last N *turns*. Trimming only cuts at user messages so a tool call is never separated from its result (APIs reject that).
- **Long-term:** facts saved with `remember_fact` live in SQLite and are injected into the system prompt on every call, so they survive "Reset conversation".
- **Control state:** a queue of pending tool calls lets the loop pause for approval and resume later.
- **Failure safety:** if the LLM call fails, the half-finished turn is rolled back so history stays valid.

## Safety design
1. **Allowlist:** the model can only call registered tools; unknown names return an error.
2. **Schema validation:** missing, extra or wrongly typed arguments are rejected before any code runs.
3. **Read-only SQL, four layers deep:** single `SELECT` only → separate read-only connection → SQLite *authorizer* allows only `SELECT` on three tables → row and time limits. `DROP`, `UPDATE`, `ATTACH`, `PRAGMA`, reading `tasks`/`facts`, and recursion bombs are all blocked by tests.
4. **Human in the loop:** destructive tools pause until you click Approve; a new message counts as rejection.
5. **Loop control:** `max_steps` cap; identical repeated calls are refused.
6. **Prompt-injection hygiene:** tool output is labelled untrusted data in the system prompt. This reduces risk but is not a guarantee; the approval step and read-only access are the real backstops.
7. **Contained failures:** a crashing tool, bad JSON arguments or a timeout becomes a message the model can react to, never a crash.

## Run locally
```bash
git clone https://github.com/adityatandale/ai-agent.git
cd ai-agent
pip install -r requirements.txt
streamlit run app.py
```
- **No key needed:** pick **Demo mode** (a rule-based planner, *not* an AI model) to try the tools, approvals and trace UI.
- **Real LLM agent:** copy `.env.example` to `.env` and set `LLM_API_KEY` (free key at console.groq.com). Any OpenAI-compatible provider with tool calling works via `LLM_BASE_URL` / `LLM_MODEL`.
- Terminal version: `python -m agent.cli`

## Tests and evaluation
```bash
python -m pytest -q                  # 55 offline tests: tools, SQL guard, loop, approvals, memory
python tests/ui_smoke.py             # drives the real UI: calculate, add, reject, approve
python -m evaluation.eval_agent      # task / tool-use / safety scenarios -> evaluation/report.md
```
The evaluation covers tool selection, multi-step reasoning, memory, and safety (blocked destructive SQL, approval before delete, injection text inside data). Run it with your LLM key for meaningful results; in demo mode the LLM-only cases are skipped.

## Frameworks: why a custom loop?
LangGraph, CrewAI, the OpenAI Agents SDK and others add graphs, multi-agent coordination, tracing and persistence. This project implements the core loop by hand (about 150 lines) so every step is visible and testable. Pick a framework when you need branching workflows, multiple cooperating agents, or managed persistence.

## Project layout
| Path | Purpose |
|---|---|
| `agent/core.py` | The agent loop, memory, approval pause/resume, loop guards |
| `agent/tools.py` | Tool definitions, schema validation, safe calculator, registry |
| `agent/db.py` | SQLite: sales data (read-only), tasks, facts, guarded SQL |
| `agent/llm.py` | OpenAI-compatible client with tool calling, retries, friendly errors |
| `agent/demo.py` | Rule-based demo planner (no key) |
| `app.py` | Streamlit UI: chat, trace viewer, approvals, state sidebar |
| `evaluation/eval_agent.py` | Scenario evaluation and report |

## Limitations
- Agent quality depends on the model: smaller models pick wrong tools or malformed arguments more often (the client retries malformed tool calls; the evaluation shows how often it still fails).
- The demo planner only recognises a few phrasings.
- Weather needs internet; there is no caching.
- One conversation per browser session; the database is temporary and lost on restart.
- No streaming responses, and tool calls run sequentially.
- Ideas: persistent storage, streaming, per-tool permissions per user, an audit log, more tools (email, calendar) behind approval.
