"""Command-line entry point: ``python -m vectordb`` or the ``vectordb`` script."""

from __future__ import annotations

import argparse
import logging

from vectordb import __version__
from vectordb.api import create_app
from vectordb.api.app import build_services
from vectordb.config import Settings


def main() -> None:
    defaults = Settings()
    parser = argparse.ArgumentParser(prog="vectordb", description="Run the VectorDB server.")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--debug", action="store_true", default=defaults.debug)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = Settings()
    services = build_services(settings)
    app = create_app(services)

    ollama_up = services.ollama.is_available()
    print(f"=== VectorDB Engine v{__version__} ===")
    print(f"http://localhost:{args.port}")
    print(
        f"{len(services.vector_db)} demo vectors | {settings.demo_dims} dims | "
        "HNSW + KD-Tree + BruteForce"
    )
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {settings.embed_model}  gen model: {settings.gen_model}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
