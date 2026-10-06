"""Tools the agent can call, plus the registry that validates and executes them safely."""
import ast
import json
import math
import operator
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .db import Database, DBError

MAX_RESULT_CHARS = 2000


class ToolError(Exception):
    """An expected failure with a message the model can read and act on."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                      # JSON Schema (type: object)
    func: Callable
    requires_confirmation: bool = False   # destructive tools pause for human approval


class ToolRegistry:
    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self):
        return list(self._tools)

    def schemas(self) -> list:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}} for t in self._tools.values()]

    @staticmethod
    def _validate(tool: Tool, args) -> dict:
        if not isinstance(args, dict):
            raise ToolError("Arguments must be a JSON object.")
        props = tool.parameters.get("properties", {})
        unknown = set(args) - set(props)
        if unknown:
            raise ToolError(f"Unknown argument(s): {sorted(unknown)}. Allowed: {sorted(props)}.")
        missing = [r for r in tool.parameters.get("required", []) if r not in args]
        if missing:
            raise ToolError(f"Missing required argument(s): {missing}.")
        clean = {}
        for key, val in args.items():
            typ = props[key].get("type")
            if typ == "integer":
                if isinstance(val, str) and val.strip().lstrip("-").isdigit():
                    val = int(val)
                if isinstance(val, bool) or not isinstance(val, int):
                    raise ToolError(f"'{key}' must be an integer.")
            elif typ == "string" and not isinstance(val, str):
                raise ToolError(f"'{key}' must be a string.")
            clean[key] = val
        return clean

    def execute(self, name: str, args) -> tuple:
        """Return (ok, text). Never raises: errors become text the model can read."""
        tool = self.get(name)
        if tool is None:
            return False, f"Error: unknown tool '{name}'. Available tools: {', '.join(self.names())}."
        try:
            out = str(tool.func(**self._validate(tool, args)))
            ok = True
        except (ToolError, DBError) as e:
            out, ok = f"Error: {e}", False
        except Exception as e:  # noqa: BLE001 - a buggy tool must not crash the agent
            out, ok = f"Error: the tool failed unexpectedly ({type(e).__name__}).", False
        if len(out) > MAX_RESULT_CHARS:
            out = out[:MAX_RESULT_CHARS] + " ...[truncated]"
        return ok, out


# ---------------- calculator (no eval!) ----------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round}


def safe_calc(expression: str):
    """Evaluate arithmetic by walking the syntax tree. Only numbers, + - * / // % ** and a few functions."""
    if len(expression) > 200:
        raise ToolError("Expression too long (max 200 characters).")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ToolError("Exponent too large.")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
                and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError("Unsupported expression. Use numbers, + - * / // % ** and sqrt/abs/round.")

    try:
        result = ev(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        raise ToolError("Division by zero.")
    except (SyntaxError, ValueError, OverflowError, TypeError):
        raise ToolError("Could not evaluate that expression.")
    return int(result) if isinstance(result, float) and result.is_integer() and abs(result) < 1e15 else result


# ---------------- weather (Open-Meteo, no API key) ----------------
WEATHER_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
                 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
                 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
                 81: "rain showers", 82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail",
                 99: "thunderstorm with hail"}


def http_get_json(url: str, params: dict, timeout: float = 8.0) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "ai-agent-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # only called with fixed https hosts
        return json.loads(resp.read().decode("utf-8"))


def build_default_tools(db: Database, http_get: Callable = http_get_json,
                        now_fn: Callable = lambda: datetime.now(timezone.utc)) -> ToolRegistry:
    def calculator(expression: str) -> str:
        return f"{expression} = {safe_calc(expression)}"

    def get_current_time(timezone_name: str = "UTC") -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            now = now_fn().astimezone(ZoneInfo(timezone_name))
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            raise ToolError(f"Unknown timezone '{timezone_name}'. Use an IANA name like 'Asia/Kolkata'.")
        return now.strftime(f"%A, %d %B %Y, %H:%M ({timezone_name})")

    def get_weather(city: str) -> str:
        city = city.strip()
        if not city or len(city) > 80:
            raise ToolError("Provide a city name.")
        try:
            geo = http_get("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1})
            if not geo.get("results"):
                raise ToolError(f"Could not find a place called '{city}'.")
            p = geo["results"][0]
            data = http_get("https://api.open-meteo.com/v1/forecast", {
                "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto",
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"})
        except ToolError:
            raise
        except Exception:  # noqa: BLE001 - network problems become a readable error
            raise ToolError("The weather service could not be reached. Try again later.")
        cur, units = data["current"], data.get("current_units", {})
        desc = WEATHER_CODES.get(cur.get("weather_code"), "unknown conditions")
        return (f"{p['name']}, {p.get('country', '')}: {cur['temperature_2m']}{units.get('temperature_2m', '°C')}, "
                f"{desc}, humidity {cur['relative_humidity_2m']}%, wind {cur['wind_speed_10m']} "
                f"{units.get('wind_speed_10m', 'km/h')}.")

    def query_database(sql: str) -> str:
        cols, rows, more = db.read_query(sql)
        if not rows:
            return "No rows returned."
        lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows]
        return "\n".join(lines) + ("\n(more rows not shown)" if more else "")

    def add_task(title: str) -> str:
        title = title.strip()
        if not title or len(title) > 200:
            raise ToolError("Task title must be 1-200 characters.")
        return f"Added task #{db.add_task(title)}: {title}"

    def list_tasks() -> str:
        tasks = db.list_tasks()
        return "\n".join(f"#{t['id']}: {t['title']}" for t in tasks) if tasks else "No tasks yet."

    def delete_task(task_id: int) -> str:
        if not db.delete_task(task_id):
            raise ToolError(f"No task with id {task_id}.")
        return f"Deleted task #{task_id}."

    def remember_fact(fact: str) -> str:
        fact = fact.strip()
        if not fact or len(fact) > 300:
            raise ToolError("Fact must be 1-300 characters.")
        db.add_fact(fact)
        return f"Remembered: {fact}"

    S = lambda desc: {"type": "string", "description": desc}  # noqa: E731
    obj = lambda props, req: {"type": "object", "properties": props, "required": req}  # noqa: E731
    return ToolRegistry([
        Tool("calculator", "Evaluate an arithmetic expression exactly. Use for any math.",
             obj({"expression": S("e.g. '17 * 23 + 5' or 'sqrt(144)'")}, ["expression"]), calculator),
        Tool("get_current_time", "Get the current date and time in an IANA timezone.",
             obj({"timezone_name": S("IANA timezone, e.g. 'Asia/Kolkata'. Default UTC.")}, []), get_current_time),
        Tool("get_weather", "Get the current weather for a city.",
             obj({"city": S("City name, e.g. 'Pune'")}, ["city"]), get_weather),
        Tool("query_database",
             "Run ONE read-only SQL SELECT on the sales database. Tables: "
             "customers(id,name,city); products(id,name,category,price); "
             "orders(id,customer_id,product_id,quantity,order_date YYYY-MM-DD). Max 50 rows.",
             obj({"sql": S("A single SELECT statement")}, ["sql"]), query_database),
        Tool("add_task", "Add an item to the user's to-do list.",
             obj({"title": S("Short task description")}, ["title"]), add_task),
        Tool("list_tasks", "List all tasks on the user's to-do list with their ids.",
             obj({}, []), list_tasks),
        Tool("delete_task", "Delete a task by id. Requires user confirmation.",
             obj({"task_id": {"type": "integer", "description": "Id from list_tasks"}}, ["task_id"]),
             delete_task, requires_confirmation=True),
        Tool("remember_fact", "Save a fact about the user for future conversations (long-term memory).",
             obj({"fact": S("A short fact, e.g. 'User prefers metric units'")}, ["fact"]), remember_fact),
    ])
