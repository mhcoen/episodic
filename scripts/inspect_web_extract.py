#!/usr/bin/env python3
"""Standalone web extraction diagnostics for a single URL.

Usage:
  .venv/bin/python scripts/inspect_web_extract.py "https://example.com"
"""

from __future__ import annotations

import argparse
import traceback


def _preview(text: str, n: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect web extraction stages for one URL")
    parser.add_argument("url", help="URL to inspect")
    parser.add_argument("--timeout", type=int, default=12, help="HTTP timeout (seconds)")
    args = parser.parse_args()

    url = args.url

    print(f"URL: {url}")

    # Imports kept local so script is self-contained for debugging environments.
    import requests
    from bs4 import BeautifulSoup

    from episodic.web_extract import (
        WebContentExtractor,
        fetch_page_content_sync,
        _sanitize_soup,
        _normalize_text,
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=args.timeout)
        print(f"HTTP status: {resp.status_code}")
        print(f"HTML bytes: {len(resp.text)}")
    except Exception as e:
        print(f"HTTP fetch error: {type(e).__name__}: {e}")
        return 1

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        raw_text = soup.get_text(separator=" ", strip=True)
        print(f"Raw soup text chars: {len(raw_text)}")
        print(f"Raw preview: {_preview(raw_text)}")
    except Exception as e:
        print(f"Parse error: {type(e).__name__}: {e}")
        print(traceback.format_exc(limit=3).strip())
        return 1

    try:
        for script in soup(["script", "style"]):
            script.decompose()
        no_script_text = soup.get_text(separator=" ", strip=True)
        print(f"After script/style removal chars: {len(no_script_text)}")
        print(f"After script/style preview: {_preview(no_script_text)}")
    except Exception as e:
        print(f"Script/style removal error: {type(e).__name__}: {e}")

    try:
        sanitized_soup = _sanitize_soup(soup, url)
        sanitized_text = sanitized_soup.get_text(separator=" ", strip=True)
        print(f"After _sanitize_soup chars: {len(sanitized_text)}")
        print(f"After _sanitize_soup preview: {_preview(sanitized_text)}")
    except Exception as e:
        print(f"_sanitize_soup error: {type(e).__name__}: {e}")
        print(traceback.format_exc(limit=4).strip())
        sanitized_soup = None

    try:
        extractor = WebContentExtractor()
        source_soup = sanitized_soup if sanitized_soup is not None else soup
        extracted = extractor._extract_main_content(source_soup, url)
        print(f"_extract_main_content chars: {len(extracted or '')}")
        print(f"_extract_main_content preview: {_preview(extracted or '')}")
    except Exception as e:
        print(f"_extract_main_content error: {type(e).__name__}: {e}")
        print(traceback.format_exc(limit=4).strip())

    try:
        normalized = _normalize_text(extracted or "")
        print(f"After _normalize_text chars: {len(normalized)}")
        print(f"After _normalize_text preview: {_preview(normalized)}")
    except Exception as e:
        print(f"_normalize_text error: {type(e).__name__}: {e}")

    # Optional: third-party extractor comparison
    try:
        import trafilatura  # type: ignore

        try:
            trafi_text = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
                output_format="txt",
            )
            trafi_text = trafi_text or ""
            print(f"trafilatura chars: {len(trafi_text)}")
            print(f"trafilatura preview: {_preview(trafi_text)}")
        except Exception as e:
            print(f"trafilatura extract error: {type(e).__name__}: {e}")
            print(traceback.format_exc(limit=4).strip())
    except ImportError:
        print("trafilatura: not installed (pip install trafilatura)")

    try:
        final = fetch_page_content_sync(url)
        print(f"fetch_page_content_sync chars: {len(final or '')}")
        print(f"fetch_page_content_sync preview: {_preview(final or '')}")
    except Exception as e:
        print(f"fetch_page_content_sync error: {type(e).__name__}: {e}")
        print(traceback.format_exc(limit=4).strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
