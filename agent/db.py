"""SQLite database used by the agent's tools.

* sales data (customers, products, orders): READ-ONLY for the agent, via a locked-down connection
* tasks: the agent can add / list / delete
* facts: long-term memory (things the user told the agent to remember)
"""
import sqlite3
import time
from pathlib import Path

READABLE_TABLES = {"customers", "products", "orders"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE IF NOT EXISTS products  (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE IF NOT EXISTS orders    (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                                      quantity INTEGER, order_date TEXT);
CREATE TABLE IF NOT EXISTS tasks     (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
                                      created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS facts     (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT NOT NULL,
                                      created TEXT DEFAULT CURRENT_TIMESTAMP);
"""

CUSTOMERS = [(1, "Asha Rao", "Pune"), (2, "Ben Carter", "Austin"), (3, "Chen Wei", "Singapore"),
             (4, "Dana Ortiz", "Madrid"), (5, "Elif Demir", "Istanbul"), (6, "Farid Khan", "Mumbai")]
PRODUCTS = [(1, "Trail Backpack", "bags", 89.0), (2, "Summit Jacket", "apparel", 149.0),
            (3, "Trailhead Boots", "footwear", 129.0), (4, "Camp Stove", "gear", 59.0),
            (5, "Wool Socks", "apparel", 14.0), (6, "Headlamp", "gear", 34.0)]
ORDERS = [(1, 1, 1, 2, "2026-01-12"), (2, 2, 3, 1, "2026-01-15"), (3, 3, 2, 1, "2026-01-20"),
          (4, 4, 5, 6, "2026-02-02"), (5, 5, 4, 2, "2026-02-09"), (6, 6, 1, 1, "2026-02-14"),
          (7, 1, 6, 3, "2026-02-21"), (8, 2, 5, 4, "2026-03-03"), (9, 3, 1, 3, "2026-03-11"),
          (10, 4, 3, 2, "2026-03-18"), (11, 5, 2, 1, "2026-03-25"), (12, 6, 5, 8, "2026-04-02"),
          (13, 1, 4, 1, "2026-04-09"), (14, 2, 1, 2, "2026-04-16"), (15, 3, 6, 2, "2026-04-23"),
          (16, 4, 2, 2, "2026-05-04"), (17, 5, 5, 5, "2026-05-12"), (18, 6, 3, 1, "2026-05-19")]


class DBError(Exception):
    """Expected, user-presentable database errors."""


class Database:
    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        if self.conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            self.conn.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
            self.conn.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
            self.conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", ORDERS)
            self.conn.commit()

    # ---------- tasks ----------
    def add_task(self, title: str) -> int:
        cur = self.conn.execute("INSERT INTO tasks(title) VALUES (?)", (title,))
        self.conn.commit()
        return cur.lastrowid

    def list_tasks(self) -> list:
        return [dict(r) for r in self.conn.execute("SELECT id, title FROM tasks ORDER BY id")]

    def delete_task(self, task_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---------- long-term memory ----------
    def add_fact(self, fact: str):
        self.conn.execute("INSERT INTO facts(fact) VALUES (?)", (fact,))
        self.conn.commit()

    def facts(self, limit: int = 20) -> list:
        rows = self.conn.execute("SELECT fact FROM facts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r["fact"] for r in reversed(rows)]

    # ---------- guarded read-only SQL ----------
    def read_query(self, sql: str, limit: int = 50, timeout_s: float = 2.0):
        """Run one SELECT against the sales tables. Defence in depth:
        1) text checks, 2) a separate read-only connection, 3) a SQLite authorizer that only
        allows SELECT on whitelisted tables, 4) a row limit and a time limit."""
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise DBError("Empty query.")
        if ";" in sql:
            raise DBError("Only a single statement is allowed.")
        if not sql.lower().startswith(("select", "with")):
            raise DBError("Only SELECT queries are allowed.")

        ro = sqlite3.connect(Path(self.path).as_uri() + "?mode=ro", uri=True)

        def authorizer(action, arg1, arg2, dbname, source):
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 in READABLE_TABLES:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        ro.set_authorizer(authorizer)
        deadline = time.monotonic() + timeout_s
        ro.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        try:
            cur = ro.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(limit + 1)
        except sqlite3.Error as e:
            msg = str(e)
            if "not authorized" in msg or "prohibited" in msg:
                raise DBError(f"Not allowed. You may only read tables: {', '.join(sorted(READABLE_TABLES))}.")
            if "interrupted" in msg:
                raise DBError("Query took too long and was stopped.")
            raise DBError(f"SQL error: {msg}")
        finally:
            ro.close()
        return cols, rows[:limit], len(rows) > limit
