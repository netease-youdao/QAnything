# Debugging QAnything with VS Code

This guide explains how to attach a debugger to the QAnything Python services using
Visual Studio Code. It is meant for developers who run QAnything from source and want
to step through the RAG chain (query rewrite → embedding → retrieval → rerank → LLM)
instead of relying only on the log files described in the [README](../README.md#debug).

> All service names, paths and ports below come from `scripts/entrypoint.sh` and
> `qanything_kernel/configs/model_config.py` in this repository. If you changed those,
> adjust the values here accordingly.

## 1. Understand the process layout

`scripts/entrypoint.sh` starts six independent Python processes with `nohup`:

| # | Script (relative to repo root) | Log file |
|---|---|---|
| 1 | `qanything_kernel/dependent_server/rerank_server/rerank_server.py` | `logs/debug_logs/rerank_server.log` |
| 2 | `qanything_kernel/dependent_server/embedding_server/embedding_server.py` | `logs/debug_logs/embedding_server.log` |
| 3 | `qanything_kernel/dependent_server/pdf_parser_server/pdf_parser_server.py` | `logs/debug_logs/pdf_parser_server.log` |
| 4 | `qanything_kernel/dependent_server/ocr_server/ocr_server.py` | `logs/debug_logs/ocr_server.log` |
| 5 | `qanything_kernel/dependent_server/insert_files_serve/insert_files_server.py --port 8110 --workers 1` | `logs/debug_logs/insert_files_server.log` |
| 6 | `qanything_kernel/qanything_server/sanic_api.py --host $USER_IP --port 8777 --workers 1` | `logs/debug_logs/main_server.log` |

The last one is the HTTP gateway; the RAG chain itself lives in
`qanything_kernel/core/local_doc_qa.py` and `qanything_kernel/core/retriever/`.
Because each service is a separate process, you must attach to the specific process
whose code you want to step through.

## 2. Install the Python debugger extension

Install the official **Python Debugger** extension (`ms-python.debugpy`) in VS Code.
It provides both "launch" and "attach" configurations and works with Python 3.8+.
No changes to the repository are required for attach mode.

## 3. Add an attach configuration

Create `.vscode/launch.json` at the repository root (the `.vscode/` directory is
normally git-ignored, so this stays local to your machine):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach: main API (sanic_api)",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}", "remoteRoot": "/workspace/QAnything" }
      ],
      "justMyCode": false
    },
    {
      "name": "Attach: rerank server",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5679 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}", "remoteRoot": "/workspace/QAnything" }
      ],
      "justMyCode": false
    }
  ]
}
```

`pathMappings` matters when the code runs inside the Docker container exposed by
`docker-compose-linux.yaml`, where the checkout is mounted at `/workspace/QAnything`.
If you run directly on the host, set `remoteRoot` to the absolute path of your checkout.
Set `justMyCode` to `false` so you can step into library code such as `sanic` and
`langchain`, which is often where a retrieval bug surfaces.

## 4. Start a service under the debugger

The debugpy attach mode needs the target process to be started with a debug socket.
For a local (non-container) run, start the gateway directly with debugpy instead of
letting `entrypoint.sh` launch it:

```bash
# From the repository root, with the virtualenv / conda env activated
python -m debugpy --listen 5678 --wait-for-client \
  qanything_kernel/qanything_server/sanic_api.py --host 127.0.0.1 --port 8777 --workers 1
```

Then in VS Code select **Attach: main API (sanic_api)** and press F5. The process
stays paused until the debugger attaches because of `--wait-for-client`.

For the other services, use a different port from the one chosen above
(for example `5679` for rerank) and start them the same way:

```bash
python -m debugpy --listen 5679 --wait-for-client \
  qanything_kernel/dependent_server/rerank_server/rerank_server.py
```

> If you prefer to debug the exact containerised setup, you can install `debugpy`
> into the image and change the corresponding `nohup python3 -u ...` line in
> `scripts/entrypoint.sh` to `nohup python3 -m debugpy --listen 0.0.0.0:5678 ...`.
> Remember that the container must publish the debug port for VS Code to reach it.

## 5. Useful breakpoints in the RAG chain

Once attached, these are the high-signal places to break (paths are relative to the
repository root):

| Goal | File | What to inspect |
|---|---|---|
| Request entry | `qanything_server/sanic_api.py` | route registration and request payload |
| Multi-turn rewrite | `core/chains/condense_q_chain.py` | the rewritten, standalone question |
| Embedding call | `connector/embedding/embedding_for_online_client.py` | query vectors sent to port 9001 |
| Retrieval | `core/retriever/parent_retriever.py` | child hits and the parent documents fetched back |
| Rerank | `connector/rerank/rerank_for_online_client.py` | scores returned from port 8001 |
| Prompt assembly | `core/local_doc_qa.py` | `generate_prompt` and token-budget trimming |
| LLM streaming | `connector/llm/llm_for_openai_api.py` | streamed chunks and the final `[DONE]` marker |

All paths above are under `qanything_kernel/`; prepend it when setting a file breakpoint.

## 6. Debugging in the browser too

The HTTP gateway returns a `text/event-stream` response. To see what the backend sends,
open the front end at `http://<host>:8777/qanything/` and use the browser's Developer
Tools (F12) → **Network** → select the `local_doc_chat` request → **Response** to watch
the streamed chunks. This complements the Python breakpoints: breakpoints show how the
answer is built, the Network tab shows what the client actually received.

## 7. Troubleshooting

- **Debugger never attaches**: confirm the port is not already in use and that the
  process was started with `python -m debugpy --listen ...` (plain `python` cannot
  be attached to after the fact).
- **Breakpoints show as "unverified"**: fix `pathMappings` — a mismatch between the
  local checkout and `remoteRoot` is the most common cause.
- **Services fail to start**: the five dependent services load models via symlinks
  created by `scripts/entrypoint.sh`; run it once (or create those symlinks) before
  starting a service under the debugger.
