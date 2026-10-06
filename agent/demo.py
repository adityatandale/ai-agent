"""DemoLLM: a tiny rule-based planner with the same interface as the real LLM client.

It is NOT an AI model. It pattern-matches a few request types and emits tool calls, so the agent
loop, tools, confirmation flow and UI can be tried without an API key. Open-ended requests
need a real LLM.
"""
import re
from itertools import count

from .llm import LLMReply, ToolCall

TZ = {"tokyo": "Asia/Tokyo", "london": "Europe/London", "new york": "America/New_York",
      "india": "Asia/Kolkata", "pune": "Asia/Kolkata", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
      "paris": "Europe/Paris", "sydney": "Australia/Sydney", "utc": "UTC"}

TOP_PRODUCTS_SQL = ("SELECT p.name, SUM(o.quantity) AS units FROM orders o JOIN products p ON p.id = o.product_id "
                    "GROUP BY p.name ORDER BY units DESC LIMIT 3")

HELP = ("Demo mode (rule-based, no LLM). Try: 'What is 17 * 23 + 5?', 'Weather in Pune', "
        "'What time is it in Tokyo?', 'Add a task: buy milk', 'List my tasks', 'Delete task 1', "
        "'Remember that I like tea', or 'Which product sold the most?'. Add an API key for open-ended requests.")


class DemoLLM:
    model = "demo-rule-based-planner"

    def __init__(self):
        self._ids = count(1)

    def _call(self, name, **args):
        import json
        return LLMReply(None, [ToolCall(f"demo-{next(self._ids)}", name, json.dumps(args))])

    def chat(self, messages, tools=None, **_):
        last = messages[-1]
        if last["role"] == "tool":                       # Process result -> Respond
            return LLMReply(f"Here is what I found:\n\n{last['content']}")
        text = last["content"].strip()
        low = text.lower()

        m = re.search(r"(?:delete|remove)\s+task\s*#?(\d+)", low)
        if m:
            return self._call("delete_task", task_id=int(m.group(1)))
        m = re.search(r"(?:add (?:a )?task|remind me to)\s*:?\s*(.+)", text, re.I)
        if m:
            return self._call("add_task", title=m.group(1).strip().rstrip("."))
        if re.search(r"\b(list|show|what are)\b.*\btasks?\b|\bmy tasks\b|\bto-?do\b", low):
            return self._call("list_tasks")
        m = re.search(r"remember (?:that )?(.+)", text, re.I)
        if m:
            return self._call("remember_fact", fact=m.group(1).strip().rstrip("."))
        m = re.search(r"weather (?:in|for|at)\s+([A-Za-z .'-]+)", text, re.I)
        if m:
            return self._call("get_weather", city=m.group(1).strip(" ?.!"))
        if re.search(r"\btime\b", low):
            tz = next((v for k, v in TZ.items() if k in low), "UTC")
            return self._call("get_current_time", timezone_name=tz)
        if re.search(r"sold the most|best.?sell|top product|most popular|total sales|revenue", low):
            return self._call("query_database", sql=TOP_PRODUCTS_SQL)
        m = re.search(r"(\d[\d\s+\-*/().%^]*\d|\d)", text)
        if m and re.search(r"[+\-*/^%]", m.group(1)):
            return self._call("calculator", expression=m.group(1).replace("^", "**").strip())
        return LLMReply(HELP)
