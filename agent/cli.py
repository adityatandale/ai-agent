"""Terminal agent:  python -m agent.cli     (uses LLM_API_KEY if set, otherwise demo mode)"""
import os
import tempfile

from .core import Agent
from .db import Database
from .demo import DemoLLM
from .llm import ChatError, LLMClient, LLMConfig
from .tools import build_default_tools


def main():
    db = Database(os.path.join(tempfile.mkdtemp(), "agent.db"))
    cfg = LLMConfig.from_env()
    llm = LLMClient(cfg) if cfg else DemoLLM()
    agent = Agent(llm, build_default_tools(db), db)
    print(f"Agent ready ({llm.model}). Type 'quit' to exit.")
    while (text := input("\nYou: ").strip()) and text.lower() != "quit":
        try:
            res = agent.run(text)
            while res.status == "needs_confirmation":
                p = res.pending
                ok = input(f"  Approve {p.call.name}({p.arguments})? [y/N] ").strip().lower() == "y"
                res = agent.resume(ok)
        except ChatError as e:
            print("Error:", e.user_message)
            continue
        for ev in res.events:
            print(f"  tool> {ev.name}({ev.arguments}) -> {ev.result[:100]}")
        print("Agent:", res.answer)


if __name__ == "__main__":
    main()
