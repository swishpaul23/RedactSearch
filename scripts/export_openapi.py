"""Write the API's OpenAPI schema to web/openapi.json, the source of the web app's TypeScript types.

python scripts/export_openapi.py && (cd web && npm run gen:api)

tests/test_api.py fails if web/openapi.json no longer matches the code, so the frontend
types cannot silently drift from the backend.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("web/openapi.json")


def schema() -> dict:
    from callsearch.api.main import create_app
    from callsearch.api.services import Services

    return create_app(Services(), web_dist=Path("/nonexistent")).openapi()


def render() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
