#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue un banco di regressione di domande contro l'API FlyTax."
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/",
        help="Endpoint chat API (default: http://127.0.0.1:8000/)",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tests/fixtures/forfettario_regression_cases.json"),
        help="File JSON con i casi di regressione.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("api_regression_report.json"),
        help="File JSON di report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = []
    passed = 0

    for idx, case in enumerate(cases, start=1):
        payload = {
            "content": case["question"],
            "regime_id": "forfettario",
        }
        req = urllib.request.Request(
            args.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))

        message = body.get("message", "")
        lowered = message.lower()
        missing = [
            term for term in case["expected_all"] if term.lower() not in lowered
        ]
        ok = not missing
        if ok:
            passed += 1

        report.append(
            {
                "index": idx,
                "question": case["question"],
                "expected_all": case["expected_all"],
                "ok": ok,
                "missing": missing,
                "message": message,
                "sources": body.get("sources", []),
            }
        )
        print(f"[{idx}/{len(cases)}] {'OK' if ok else 'FAIL'}")

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPassati: {passed}/{len(cases)}")
    print(f"Report: {args.output}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
