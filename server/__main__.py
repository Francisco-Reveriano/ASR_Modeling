"""Run the local app with python -m server after building the React frontend."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("server.app:create_app", factory=True, host="127.0.0.1", port=8000,
                workers=1, ws_max_size=1024 * 1024, ws_max_queue=16,
                timeout_graceful_shutdown=5)
