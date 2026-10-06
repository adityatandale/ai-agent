"""UI smoke test (no network, no key):  python tests/ui_smoke.py"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.pop("LLM_API_KEY", None)

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert not at.exception, at.exception
assert any("Demo mode" in i.value for i in at.info)

def chat(text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception

chat("What is 17 * 23 + 5?")
assert "396" in at.chat_message[-1].markdown[0].value, [m.markdown[0].value for m in at.chat_message]

chat("Add a task: buy milk")
assert "milk" in " ".join(m.value for m in at.sidebar.markdown), "sidebar should list the new task"

chat("Delete task 1")
assert at.chat_input[0].disabled, "chat input must be disabled while approval is pending"
assert any("Approval needed" in w.value for w in at.warning)
[b for b in at.button if "Reject" in b.label][0].click().run()
assert not at.exception
assert "milk" in " ".join(m.value for m in at.sidebar.markdown), "rejected delete must keep the task"

chat("Delete task 1")
[b for b in at.button if "Approve" in b.label][0].click().run()
assert not at.exception
assert "_none_" in " ".join(m.value for m in at.sidebar.markdown), "approved delete must remove the task"
print("UI OK")
