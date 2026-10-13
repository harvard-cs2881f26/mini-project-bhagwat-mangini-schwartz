#!/usr/bin/env python
"""Check that the paper hasn't changed since configs/paper/claims.yaml was read. Free: one arXiv API call.

    python scripts/check_paper_version.py   # exit 0: same version; 1: newer version; 2: network error

Asks the arXiv API for the latest version of `paper.arxiv` and compares it with `paper.version`.
Needs the network, so it is not part of CI (CI stays offline). The network and XML parts use only
the standard library; claims.yaml is read with the project's own loader (asimov.claims).
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.claims import load_claims  # noqa: E402

ARXIV_API = "http://export.arxiv.org/api/query?id_list={arxiv}"
ATOM = "{http://www.w3.org/2005/Atom}"
TIMEOUT_SECONDS = 20
VERSION = re.compile(r"v(\d+)$")
PAPER_READER = ".claude/agents/paper-reader.md"

EXIT_SAME, EXIT_NEWER, EXIT_NETWORK = 0, 1, 2


def latest_version(atom_xml: str) -> str:
    """The version ("v2") at the end of the first entry's <id>, e.g. .../abs/2509.21651v2."""
    try:
        feed = ET.fromstring(atom_xml)
    except ET.ParseError as e:
        raise ValueError(f"arXiv reply is not valid XML: {e}") from e
    entry_id = feed.findtext(f"{ATOM}entry/{ATOM}id")
    if not entry_id:
        raise ValueError("arXiv reply has no <entry><id>; is the paper id right?")
    match = VERSION.search(entry_id.strip())
    if not match:
        raise ValueError(f"arXiv entry id {entry_id!r} does not end in a version like v2")
    return match.group(0)


def version_number(version: str) -> int:
    match = VERSION.fullmatch(version)
    if not match:
        raise ValueError(f"not an arXiv version: {version!r} (expected e.g. 'v2')")
    return int(match.group(1))


def is_newer(latest: str, read: str) -> bool:
    """True if `latest` is a later arXiv version than `read` (numeric, so v10 > v9)."""
    return version_number(latest) > version_number(read)


def fetch_feed(arxiv: str) -> str:
    with urllib.request.urlopen(ARXIV_API.format(arxiv=arxiv), timeout=TIMEOUT_SECONDS) as reply:
        return reply.read().decode("utf-8")


def main() -> int:
    paper = load_claims().paper
    try:
        feed = fetch_feed(paper.arxiv)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"could not reach the arXiv API for {paper.arxiv} ({e}); check the network and retry", file=sys.stderr)
        return EXIT_NETWORK
    latest = latest_version(feed)
    if is_newer(latest, paper.version):
        print(
            f"WARNING: arXiv {paper.arxiv} is now {latest}, but claims.yaml was read from {paper.version}. "
            f"Re-run the paper-reader subagent ({PAPER_READER}) and review the differences."
        )
        return EXIT_NEWER
    print(f"paper is still {paper.version} (claims.yaml up to date)")
    return EXIT_SAME


if __name__ == "__main__":
    sys.exit(main())
