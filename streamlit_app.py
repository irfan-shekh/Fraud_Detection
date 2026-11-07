"""
Streamlit Community Cloud Root Entry Point.
Directs execution to dashboard/app.py while ensuring proper module resolution.
"""
import os
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

target_app = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
with open(target_app, "r", encoding="utf-8") as f:
    code = compile(f.read(), target_app, "exec")
    exec(code, globals())

