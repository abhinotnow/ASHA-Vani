"""ASHA-Vani package entrypoint (`uv run asha-vani`)."""


def main() -> None:
    import uvicorn

    uvicorn.run("asha_vani.server:app", host="127.0.0.1", port=8000, reload=True)
