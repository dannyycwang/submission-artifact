"""Content-preserving attribute serialization and canonical controls."""

from __future__ import annotations
import hashlib, random, re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _random_order(values: list[str], product_id: str, seed: int) -> list[str]:
    result = list(values)
    digest = hashlib.sha256(f"{seed}:{product_id}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def _blocks(
    record: dict, attributes: list[str] | None = None, description: str | None = None
) -> list[tuple[str, str]]:
    attrs = record["attributes"] if attributes is None else attributes
    desc = record["description"] if description is None else description
    values = {
        "title": record["title"],
        "class": record.get("class", ""),
        "category": record.get("category", ""),
        "description": desc,
        "attributes": record["attribute_separator"].join(attrs),
    }
    return [
        (key, values[key]) for key in record["section_order"] if values.get(key, "")
    ]


def _plain(
    record: dict,
    attributes: list[str] | None = None,
    description: str | None = None,
    reverse_sections: bool = False,
) -> str:
    blocks = _blocks(record, attributes, description)
    if reverse_sections:
        blocks.reverse()
    return "\n".join(value for _, value in blocks)


SCHEDULES = ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]
RULES = ["lexical_ascending", "lexical_descending", "field_priority_type"]


def ordered_attributes(record, schedule):
    attributes = list(record["attributes"])
    if schedule == "C0":
        return attributes
    if schedule == "C1":
        return attributes[::-1]
    if schedule.startswith("C2s"):
        return _random_order(
            attributes, str(record["product_id"]), 20260910 + int(schedule[3:])
        )
    if schedule.startswith("P"):
        return _random_order(attributes, str(record["product_id"]), int(schedule[1:]))
    if schedule.startswith("E"):
        import itertools

        return list(sorted(set(itertools.permutations(attributes)))[int(schedule[1:])])
    if schedule == "field_priority_type":

        def key(value):
            field = (
                value.split(":", 1)[0].lower() if ":" in value else value[:40].lower()
            )
            return (
                0 if any(t in field for t in ["producttype", "type", "style"]) else 1,
                value.casefold(),
                value,
            )

        return sorted(attributes, key=key)
    if schedule in RULES:
        return sorted(
            attributes,
            key=lambda x: (x.casefold(), x),
            reverse=schedule == "lexical_descending",
        )
    if schedule.startswith("view"):
        base = sorted(attributes, key=lambda x: (x.casefold(), x))
        return _random_order(
            base, str(record["product_id"]), 20260970 + int(schedule[4:])
        )
    raise ValueError(f"Unknown ordering: {schedule}")


def serialize(record, schedule="C0"):
    return _plain(record, ordered_attributes(record, schedule))


def audit(record, condition, text, original):
    return {
        "pass": Counter(text) == Counter(original)
        and Counter(tokens(text)) == Counter(tokens(original))
    }
