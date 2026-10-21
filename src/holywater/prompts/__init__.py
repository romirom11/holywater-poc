"""Prompt templates as versioned files.

Format: a `---` frontmatter block with `version:`, then `## system` and `## user` sections.
Placeholders use `$name` (`string.Template`) so JSON braces in examples stay literal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Template

from holywater.observability.errors import ConfigError

_PROMPTS_DIR = Path(__file__).parent
_SECTION_RE = re.compile(r"^## (\w+)\s*$", flags=re.M)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    sections: dict[str, str]

    @property
    def version_tag(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, section: str, **values: object) -> str:
        try:
            template = self.sections[section]
        except KeyError as exc:
            msg = f"prompt {self.name!r} has no section {section!r}"
            raise ConfigError(msg) from exc
        try:
            return Template(template).substitute(values).strip()
        except KeyError as exc:
            msg = f"prompt {self.name!r}/{section} is missing placeholder value {exc}"
            raise ConfigError(msg) from exc


@cache
def load_prompt(name: str) -> PromptTemplate:
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        msg = f"prompt file not found: {path}"
        raise ConfigError(msg, context={"path": str(path)})
    text = path.read_text(encoding="utf-8")
    version, body = _split_frontmatter(text, path)
    parts = _SECTION_RE.split(body)
    # parts = [preamble, name1, body1, name2, body2, ...]
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not sections:
        msg = f"prompt {path} has no '## section' headers"
        raise ConfigError(msg)
    return PromptTemplate(name=name, version=version, sections=sections)


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    if not text.startswith("---"):
        msg = f"prompt {path} must start with a '---' frontmatter block"
        raise ConfigError(msg)
    _, frontmatter, body = text.split("---", 2)
    match = re.search(r"^version:\s*(\S+)", frontmatter, flags=re.M)
    if not match:
        msg = f"prompt {path} frontmatter has no 'version:'"
        raise ConfigError(msg)
    return match.group(1), body
