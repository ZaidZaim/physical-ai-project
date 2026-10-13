"""puppy-mind CLI.  `puppy-mind --help`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .brain import Brain
from .config import get_settings


def _p(x) -> None:
    print(json.dumps(x, indent=2, default=str, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="puppy-mind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create constraints/indexes in Aura and verify connectivity")
    o = sub.add_parser("observe", help="describe one image file and write it to memory")
    o.add_argument("image")
    o.add_argument("--location")
    w = sub.add_parser("watch", help="continuously observe CAMERA_SOURCE")
    w.add_argument("--source")
    w.add_argument("--interval", type=float)
    a = sub.add_parser("ask", help="ask a question (omit for interactive chat)")
    a.add_argument("question", nargs="?")
    a.add_argument("--evidence", action="store_true")
    sd = sub.add_parser("seed", help="replay a scripted scenario (no camera/OpenAI needed)")
    sd.add_argument("scenario", nargs="?", default="eval/scenarios/booth.json")
    sub.add_parser("stats")
    sub.add_parser("reset").add_argument("--yes", action="store_true")
    e = sub.add_parser("eval", help="run the question set against current memory")
    e.add_argument("questions", nargs="?", default="eval/questions.json")
    sv = sub.add_parser("serve", help="web UI + robot ingest API")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("puppy_mind.server:app", host=args.host, port=args.port)
        return 0

    s = get_settings()
    need_ai = args.cmd in {"observe", "watch", "ask", "eval"}
    if need_ai and not s.openai_api_key:
        print("OPENAI_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2
    brain = Brain(s, need_openai=need_ai)
    try:
        if args.cmd == "init":
            brain.mem.init_schema()
            _p({"connected": True, "stats": brain.mem.stats(), "nams": brain.nams.enabled})
        elif args.cmd == "stats":
            _p(brain.mem.stats())
        elif args.cmd == "observe":
            _p(brain.observe_frame(Path(args.image).read_bytes(), source=args.image,
                                   location=args.location, skip_if_similar=False))
        elif args.cmd == "watch":
            from .sources import open_source

            src = open_source(args.source or s.camera_source)
            interval = args.interval or s.watch_interval_s
            print(f"watching {src.name} every {interval}s - Ctrl+C to stop")
            brain.watch(src, interval, lambda r: print(json.dumps(r, default=str)[:400]))
            brain._watch.join()  # type: ignore[union-attr]
        elif args.cmd == "seed":
            from .scenario import replay

            brain.mem.init_schema()
            for r in replay(brain, args.scenario):
                print(r["ts"][:19], "new:", [x.split(" (")[0] for x in r["new_objects"]],
                      "missed:", [m["label"] for m in r["missed"]])
        elif args.cmd == "ask":
            if args.question:
                r = brain.ask(args.question)
                print(r["answer"])
                if args.evidence:
                    _p(r["evidence"])
            else:
                print("Ask about your environment (empty line to quit)")
                while (q := input("you> ").strip()):
                    print("puppy>", brain.ask(q)["answer"])
        elif args.cmd == "reset":
            if not args.yes:
                print("refusing: this deletes ALL robot memory. Re-run with --yes")
                return 1
            brain.mem.reset()
            print("memory cleared")
        elif args.cmd == "eval":
            from .evaluate import run

            return run(brain, args.questions)
    except KeyboardInterrupt:
        pass
    finally:
        brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
