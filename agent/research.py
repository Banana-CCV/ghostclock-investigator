"""Context research for the investigator: a sourced web answer via the Linkup search API.

Used by the `research_context` tool when LINKUP_API_KEY is set (in .env). The agent uses it for
questions that change how a finding should be read: does this recorder model lose its clock
after a power cut, was there a local outage that evening, did the clocks change that weekend.
NOT yet exercised against the live API in this repo (no key was available when it was written).
"""
from __future__ import annotations

import os

import requests

URL = "https://api.linkup.so/v1/search"


def linkup_search(query: str, depth: str = "standard") -> dict:
    key = os.environ.get("LINKUP_API_KEY")
    if not key:
        return {"status": "unavailable", "note": "LINKUP_API_KEY is not set"}
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"},
                          json={"q": query, "depth": depth, "outputType": "sourcedAnswer"}, timeout=45)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # network / auth / quota: the agent carries on without context
        return {"status": "error", "note": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"status": "ok", "answer": j.get("answer", ""),
            "sources": [{"name": s.get("name"), "url": s.get("url"), "snippet": (s.get("snippet") or "")[:300]}
                        for s in (j.get("sources") or [])[:6]]}
