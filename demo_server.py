"""Local offline entry point, preserving the upstream training UI."""
import os
import argparse
import re
from pathlib import Path

os.environ["ALPHAMASTER_DEMO"] = "1"
os.environ.setdefault("ALPHAMASTER_CPU_THREADS", "2")
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from web.app import app

# An offline demo has no reason to start the live-market polling thread.
app.router.on_startup[:] = [
    f for f in app.router.on_startup if f.__name__ != "_startup_realtime"
]


@app.middleware("http")
async def offline_demo(request, call_next):
    if request.method != "GET" and request.url.path.startswith(
        ("/api/realtime", "/api/ai/")
    ):
        return JSONResponse({"detail": "此体验入口仅提供本地训练与回测。"}, status_code=403)
    if request.url.path == "/":
        html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
        html = re.sub(r'<link[^>]+href="https://fonts\.[^>]+>', '', html)
        html = html.replace("https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js", "/static/chart.umd.min.js")
        html = html.replace("</head>", '<style>#aiPanelSection,[data-page="realtime"]{display:none!important}</style></head>')
        banner = '<div style="padding:12px;background:#193a31;color:white;text-align:center">A股离线体验 · 只做多 · 80步短训练　<a href="/demo/" style="color:#a6f3c8">查看留出检验与经典策略对比 →</a></div>'
        html = html.replace("<body>", "<body>" + banner, 1)
        return HTMLResponse(html)
    return await call_next(request)


(ROOT / "demo/report").mkdir(parents=True, exist_ok=True)
app.mount("/demo", StaticFiles(directory=ROOT / "demo/report", html=True), name="demo-report")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn
    print(f"AlphaMaster 训练页：http://127.0.0.1:{args.port}/", flush=True)
    print(f"离线对比报告：http://127.0.0.1:{args.port}/demo/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
