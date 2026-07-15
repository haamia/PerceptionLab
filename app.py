"""
PerceptionLab

Application Entry Point
"""

from ui.dashboard import build_dashboard


app = build_dashboard()


if __name__ == "__main__":
    app.launch()