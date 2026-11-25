"""Local FastAPI endpoints. Run one worker; this is not a multi-user service."""
import json
import threading
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .service import PaperService, ROOT
from .training import load_checkpoint, generate


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retriever: str = "bm25"
    generator: str = "extractive"
    k: int = Field(default=3, ge=1, le=10)
    min_coverage: float = Field(default=0.23, ge=0, le=1)


class LabRequest(BaseModel):
    text: str = Field(default="Research questions", max_length=2000)
    max_new_tokens: int = Field(default=48, ge=0, le=128)


def create_app(workspace=None):
    app = FastAPI(title="PaperLab", version="1.0.0", description="Local research evidence and language-model lab",
                  docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    service = PaperService(workspace)
    app.state.service = service
    lab_lock = threading.Lock()
    checkpoint = ROOT / "runs" / "mini_gpt.npz"
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def local_writes(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if request.headers.get("x-paperlab") != "1" or (origin and urlparse(origin).netloc != request.headers.get("host")):
                return JSONResponse({"detail": "Use the local dashboard or send X-PaperLab: 1."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; object-src 'none'"
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def model_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    @app.get("/api/status")
    def status():
        return service.status()

    @app.post("/api/ask")
    def ask(body: AskRequest):
        if not body.question.strip():
            raise ValueError("Enter a question.")
        return service.ask(body.question, mode=body.retriever, generator=body.generator,
                           k=body.k, min_coverage=body.min_coverage)

    @app.post("/api/upload")
    async def upload(request: Request):
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(413, "PDFs must be 20 MiB or smaller.")
        try:
            # PDF parsing runs outside the event loop.
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(service.upload, request.headers.get("x-filename", ""), bytes(data))
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/papers/{name}")
    def paper(name: str):
        if name not in {p.name for p in service.folder.glob("*.pdf")}:
            raise HTTPException(404, "Paper not found.")
        return FileResponse(service.folder / name, media_type="application/pdf")

    @app.get("/api/results")
    def results():
        data = {"scope": "Included teaching collection, not uploaded papers"}
        for name in ("training", "benchmark"):
            path = ROOT / "runs" / f"{name}.json"
            data[name] = json.loads(path.read_text()) if path.exists() else None
        return data

    @app.post("/api/lab")
    def lab(body: LabRequest):
        if not checkpoint.exists():
            raise HTTPException(409, "Train first: python -m paperlab train")
        with lab_lock:
            model, tok, report = load_checkpoint(checkpoint)
            ids = tok.encode(body.text)
            return {"ids": ids, "round_trip": tok.decode(ids),
                    "utf8_bytes": len(body.text.encode()), "bpe_tokens": len(ids),
                    "vocabulary": tok.vocab_size, "parameters": model.parameter_count,
                    "sample": generate(model, tok, body.text, max_new_tokens=body.max_new_tokens),
                    "warning": "Educational next-token model; its sample is not a factual answer."}
    return app


app = create_app()

