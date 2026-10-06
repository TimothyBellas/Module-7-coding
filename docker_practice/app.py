"""A small FastAPI application for practicing Docker commands."""

from fastapi import FastAPI

app = FastAPI(title="My First Docker API")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello from Docker!"}
