from __future__ import annotations

import re
from pathlib import Path

LOCAL_LINK = re.compile(r"(?<!!)\[[^\]]+\]\((?!https?://|mailto:|#)([^)#]+)(?:#[^)]+)?\)")


def test_local_markdown_links_resolve() -> None:
    repository_root = Path(__file__).parents[1]
    broken_links = []

    for markdown_path in repository_root.rglob("*.md"):
        if ".venv" in markdown_path.parts:
            continue
        content = markdown_path.read_text(encoding="utf-8")
        for match in LOCAL_LINK.finditer(content):
            target = (markdown_path.parent / match.group(1)).resolve()
            if not target.exists():
                broken_links.append(f"{markdown_path.relative_to(repository_root)} -> {match.group(1)}")

    assert not broken_links, "仓库内 Markdown 链接失效:\n" + "\n".join(broken_links)
