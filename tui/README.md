# CoreLine OpenTUI Frontend

This folder contains an OpenTUI frontend for CoreLine while keeping Python as backend.

## Requirements

- Bun (OpenTUI runtime)
- Python environment for backend dependencies

## Run

From project root:

1. Install frontend deps:
   - `cd tui`
   - `bun install`
2. Start frontend:
   - `cd tui && bun start`

## Build (optional standalone binary)

From `tui/`:

- `bun run build`

This will produce `tui/coreline-tui`. You can run it from the repo root like:

- `./tui/coreline-tui`

The frontend starts `coreline_backend_bridge.py`, which talks to your existing CoreLine logic via JSON lines over stdin/stdout.

## Notes

- Existing backend commands and flags are preserved (`*won`, `*runon`, `*voiceon`, etc.).
- `[RUN: ...]` confirmations are handled through the input line (enter `y` or `n` when prompted).
