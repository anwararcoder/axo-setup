#!/usr/bin/env python3
"""Read-only inventory of fetched challenge folders (the layout fetch_challenges.py writes). Standard library only.

    python3 scan_challenges.py [ROOT] [--also DIR ...] [--md OUT.md] [--json OUT.json] [--now 2026-10-12T10:00]

ROOT (default ./challenges) holds one folder per challenge with description.md, baseline.md,
public-leaderboard.md, kaggle-notebook.md and the archived channel post <date>_<time>_msg<id>.md.

For every folder it reports, without downloading anything and without ever printing a dataset link:
  card        domain, level, compute, tags and baseline (baseline.md)
  status      the newest status block (public-leaderboard.md): state, whether the account submitted, solvers
              above the baseline, top score, submissions, credits, the closing countdown turned into an
              estimated UTC time (the file's modification time plus the countdown), whether the account is
              out of the continuing round, the board's 3rd and 10th scores and any marked row ("<- ...")
  rules       metric direction and sentence, runtime-limit sentences, the sentences about pretrained models
              with a HEURISTIC policy class (A allowed, B from scratch, C unclear or mixed signals) that must be
              confirmed by reading the quotes, the prohibitive bullet points, and the row counts the text names
  work        local work in the folder (notes, rules, solutions, submission folders, data) and in a folder
              of the same name under each --also directory (another workspace on this machine)
  selectable  state OPEN, the account not shut out of the continuing round, the estimated close not passed

Every status value is as old as its file: the report prints each status file's age. Refresh the files with the
workspace's fetch_challenges.py (it uses the user's Telegram login: run it only with the user's approval).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

FETCHED = re.compile(r"^(description|baseline|kaggle-notebook|public-leaderboard)\.md$|_msg\d+(_.*)?\.md$")
POST = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_msg\d+\.md$")
WORK_FILES = ("NOTES.md", "WORKLOG.md", "LOG.md", "RULES.md", "RESEARCH.md", "STATE.md", "experiments.jsonl")
WORK_DIRS = ("data", "dl", "out", "runs", "experiments", "submit")
POLICY_B = (r"\bno pre-?trained\b", r"pre-?trained[^.\n]{0,80}\b(not allowed|prohibited|forbidden|banned|not permitted)\b",
            r"\b(not allowed|prohibited|forbidden|banned|do not use)\b[^.\n]{0,80}pre-?trained", r"\bfrom[- ]scratch\b",
            r"\brandom(ly)? initiali[sz]", r"\bno pretrained weights\b")
POLICY_A = (r"pre-?trained[^.\n]{0,80}?(?<!not )\b(allowed|permitted|may be used|can be used)\b",
            r"(?<!not )\b(allowed|permitted|you may|may fine-?tune)\b[^.\n]{0,120}pre-?trained",
            r"\bfine-?tune (a|an|the) [^.\n]{0,50}pre-?trained")
POLICY_TOPIC = re.compile(r"pre-?trained|checkpoint|from[- ]scratch|random(ly)? initiali|fine-?tun|transfer learning|"
                          r"embeddings? (trained|from)|tokeni[sz]ers? trained|external (learned )?(weights|models?)", re.I)
COUNT = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{4,})\b((?:\s+[\w-]+){0,3}?\s+(?:rows|examples|bundles|items|images|lines|pairs|"
                   r"records|samples|stations|compounds|documents|utterances|trials|tiles|banks|episodes|plays))\b", re.I)


def read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def mtime(path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_card(text):
    card, key = {}, None
    for line in text.splitlines():
        match = re.match(r"^(Domain|Level|Compute|Tags|Baseline)\s+(.*)$", line)
        if match:
            key = match.group(1).lower()
            card[key] = match.group(2).strip()
        elif key and line.startswith((" ", "\t")) and line.strip():
            card[key] = (card[key] + " " + line.strip()).strip()
    card["tags"] = re.sub(r",\s*$", "", re.sub(r",\s*", ", ", card.get("tags", ""))).strip(", ")
    return card


def parse_status(text, written):
    grab = lambda pattern: (re.search(pattern, text) or [None, None])[1]
    status = {
        "state": (grab(r"Status:\s*([^\n]+)") or "?").strip(),
        "submitted": grab(r"Submitted:\s*(Yes|No)"),
        "above_ai": grab(r"(\d+/\d+) solvers beat AI"),
        "baseline": number(grab(r"AI Baseline:\s*([-\d.]+)")),
        "top": number(grab(r"Top Score[^:\n]*:\s*([-\d.]+)")),
        "submissions": grab(r"Submissions:\s*(\d+)"),
        "credits": grab(r"Submission Credits:\s*(\d+/\d+)"),
        "out_of_round": bool(re.search(r"was not among the top 10|no longer takes submissions", text)),
        "written_utc": written.isoformat(timespec="minutes") if written else None,
    }
    countdown = re.search(r"Closing in:\s*(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m)?", text)
    if countdown and any(countdown.groups()) and written:
        d, h, m = (int(x or 0) for x in countdown.groups())
        status["closes_utc_est"] = (written + timedelta(days=d, hours=h, minutes=m)).isoformat(timespec="minutes")
    rows = [(int(r), number(s), b) for r, s, b in re.findall(r"^\s*(\d+)\s+([-\d.]+)\s+(OK|--)\b", text, re.M)]
    scores = {rank: score for rank, score, _ in rows}
    status["third"], status["tenth"] = scores.get(3), scores.get(10)
    marked = re.search(r"^\s*(\d+)\s+([-\d.]+)\s+(?:OK|--)[^\n]*<-\s*([^\n]+)$", text, re.M)
    if marked:
        status["marked_row"] = {"rank": int(marked.group(1)), "score": number(marked.group(2)),
                                "who": marked.group(3).strip()}
    return status


def units(text):
    """The description as sentences: hard-wrapped lines joined, each bullet kept apart, code blocks and headings dropped."""
    text = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    blocks, current = [], []
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        bullet = re.match(r"^([*\-•]|\d+\.)\s+", stripped)
        if not stripped or stripped.startswith("#") or bullet:
            if current:
                blocks.append(" ".join(current))
            current = [re.sub(r"^([*\-•]|\d+\.)\s+", "", stripped)] if bullet else []
        else:
            current.append(stripped)
    found = []
    for block in blocks:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z*\"'(`])", block.replace("**", "")):
            sentence = sentence.strip()
            if len(sentence) > 3 and sentence not in found:
                found.append(sentence)
    return found


def matching(sentences, pattern, limit=None):
    hits = [s for s in sentences if re.search(pattern, s, re.I)]
    return hits[:limit] if limit else hits


def parse_description(text, card):
    sentences = units(text)
    direction = "higher is better" if re.search(r"higher is better", text, re.I) else (
        "lower is better" if re.search(r"lower is better", text, re.I) else "not stated")
    heading = re.search(r"^\W*(?:\d+\.\s*)?(?:Evaluation|Evaluation metric|Metric)\b[^\n]*\n", text, re.M | re.I)
    metric = next((s for s in units(text[heading.end():]) if len(s) > 15), None) if heading else None
    runtime = matching(sentences, r"\b\d+(\.\d+)?[\s-]*(minutes?|hours?)\b")
    runtime.sort(key=lambda s: not re.search(r"\b(finish|within|under|limit|session|must|budget|end to end)\b", s, re.I))
    runtime = runtime[:4]
    policy = matching(sentences, POLICY_TOPIC.pattern)
    signals = "\n".join(policy) + "\n" + card.get("domain", "") + " " + card.get("tags", "")
    b = any(re.search(p, signals, re.I) for p in POLICY_B) or card.get("domain", "").lower() == "from scratch"
    a = any(re.search(p, signals, re.I) for p in POLICY_A) or card.get("domain", "").lower() == "fine-tuning"
    hint = "C (mixed signals: read the quotes)" if a and b else "A (pretrained allowed)" if a else \
        "B (from scratch)" if b else "C (not stated)"
    # The most decisive sentences first: those that name pretrained models or scratch training.
    policy.sort(key=lambda s: not re.search(r"pre-?trained|scratch", s, re.I))
    prohibitions = matching(sentences, r"^(do not|don't|no |not allowed|never|must not|avoid)|\b(not allowed|prohibited|"
                                       r"forbidden|must not|do not use|is not permitted|are not permitted)\b", 14)
    counts = []
    for match in COUNT.finditer(" ".join(sentences)):
        snippet = (match.group(1) + match.group(2)).strip()
        if snippet not in counts:
            counts.append(snippet)
    banned = matching(sentences, r"\b(do not|don't|no|not allowed|never)\b[^.]*\b(web search|searching|internet search)\b")
    return {"direction": direction, "metric_line": metric[:240] if metric else None, "runtime": runtime,
            "policy_hint": hint, "policy_quotes": [s[:300] for s in policy[:6]], "prohibitions": prohibitions,
            "counts": counts[:6], "web_search_banned": bool(banned)}


def work_in(folder):
    if not folder.is_dir():
        return None
    files = sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    own = [p for p in files if not FETCHED.search(p.name)]
    found = sorted({p.name for p in folder.iterdir() if p.name in WORK_FILES})
    found += sorted(p.name + "/" for p in folder.iterdir() if p.is_dir() and
                    (p.name in WORK_DIRS or p.name.startswith("submission")))
    found += sorted(p.name for p in folder.iterdir() if p.is_file() and re.match(r"solution.*\.py$", p.name))
    best = None
    for name in ("LOG.md", "NOTES.md", "WORKLOG.md", "STATE.md"):
        hit = re.search(r"^\s*\**BEST:?\**:?\s*(.+)$", read(folder / name), re.M)
        if hit:
            best = f"{name}: {hit.group(1).strip()[:140]}"
            break
    newest = max((mtime(p) for p in own), default=None)
    return {"items": found, "best": best, "own_files": len(own),
            "last_change_utc": newest.isoformat(timespec="minutes") if newest else None}


def scan(root, also, now):
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p / "description.md").is_file()):
        post_files = sorted(p for p in folder.iterdir() if POST.match(p.name))
        post = read(post_files[-1]) if post_files else ""
        card = parse_card(read(folder / "baseline.md"))
        board_path = folder / "public-leaderboard.md"
        status = parse_status(read(board_path), mtime(board_path)) if board_path.is_file() else {"state": "no status file"}
        title = next((l.strip() for l in post.splitlines() if l.strip() and not l.startswith("<!--") and "Status:" not in l),
                     None) or next((l.lstrip("# ").strip() for l in read(folder / "description.md").splitlines() if l.strip()),
                                   folder.name)
        row = {"slug": folder.name, "title": title, "card": card,
               "post_state": (re.search(r"Status:\s*([^\n]+)", post) or [None, None])[1],
               "post_utc": (re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2} UTC", post) or [None, None])[1],
               "status": status, "rules": parse_description(read(folder / "description.md"), card),
               "work": work_in(folder), "elsewhere": {}}
        for other in also:
            found = work_in(other / folder.name)
            if found and found["own_files"]:
                row["elsewhere"][str(other)] = found
        closes = status.get("closes_utc_est")
        reasons = []
        if not status.get("state", "").upper().startswith("OPEN"):
            reasons.append(f"state {status.get('state')}")
        if status.get("out_of_round"):
            reasons.append("account not in the continuing round")
        if closes and datetime.fromisoformat(closes) <= now:
            reasons.append("estimated close has passed")
        row["selectable"] = not reasons
        row["not_selectable_because"] = reasons
        if closes:
            row["hours_to_close_est"] = round((datetime.fromisoformat(closes) - now).total_seconds() / 3600, 1)
        if status.get("written_utc"):
            row["status_age_hours"] = round((now - datetime.fromisoformat(status["written_utc"])).total_seconds() / 3600, 1)
        rows.append(row)
    return rows


def fmt(x):
    return "-" if x is None else (f"{x:g}" if isinstance(x, float) else str(x))


def markdown(rows, root, now):
    out = [f"# Challenge inventory ({now:%Y-%m-%d %H:%M} UTC)", "",
           f"Root `{root}`: {len(rows)} challenge folders. Status values are as old as their file (column 'age h'). "
           "Policy is a heuristic hint from quoted sentences: confirm it by reading them.", "",
           "| # | Challenge | State (age h) | Domain / compute | Base | Top | 10th | Above AI | Subs | Credits | "
           "Closes UTC (est.) | Marked row | Policy hint | Work here / elsewhere |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        s, c = r["status"], r["card"]
        marked = s.get("marked_row")
        marked_text = "#%d %s (%s)" % (marked["rank"], fmt(marked["score"]), marked["who"]) if marked else "-"
        here = r["work"]["items"] if r["work"] else []
        elsewhere = "; ".join(f"{Path(k).parent.name}: {', '.join(v['items']) or str(v['own_files']) + ' files'}"
                              for k, v in r["elsewhere"].items())
        out.append(f"| {i} | {r['title'][:60]} | {s.get('state', '?')} ({fmt(r.get('status_age_hours'))}) | "
                   f"{c.get('domain', '?')} / {c.get('compute', '?')} | {c.get('baseline', '-')} | {fmt(s.get('top'))} | "
                   f"{fmt(s.get('tenth'))} | {fmt(s.get('above_ai'))} | {fmt(s.get('submissions'))} | {fmt(s.get('credits'))} | "
                   f"{(s.get('closes_utc_est') or '-').replace('T', ' ')[:16]} | {marked_text} | "
                   f"{r['rules']['policy_hint']} | {', '.join(here) or '-'}{' / ' + elsewhere if elsewhere else ''} |")
    selectable = [r for r in rows if r["selectable"]]
    out += ["", "## Selectable now", ""]
    out += [f"- **{r['title']}** (`{r['slug']}`): submitted {r['status'].get('submitted')}, "
            f"{fmt(r['status'].get('above_ai'))} above the baseline, credits {fmt(r['status'].get('credits'))}, "
            f"closes {('in about ' + fmt(r.get('hours_to_close_est')) + ' h') if r.get('hours_to_close_est') is not None else 'at an unknown time'}"
            for r in selectable] or ["- none"]
    out += ["", "## Not selectable", ""]
    out += [f"- {r['title']}: {'; '.join(r['not_selectable_because'])}" for r in rows if not r["selectable"]] or ["- none"]
    out += ["", "## Rules per challenge (quoted)", ""]
    for r in rows:
        rules = r["rules"]
        out += [f"### {r['title']}", "",
                f"- Metric: {rules['direction']}; {rules['metric_line'] or 'no evaluation line found'}",
                f"- Sizes named in the text: {', '.join(rules['counts']) or 'none found'}",
                f"- Runtime sentences: " + (" | ".join(rules["runtime"]) or "none found (platform default: about 1 hour)"),
                f"- Policy hint {rules['policy_hint']}; quotes: " + (" | ".join(rules["policy_quotes"]) or "none"),
                f"- Web search banned by a strict reading: {'YES' if rules['web_search_banned'] else 'no'}",
                f"- Prohibitions ({len(rules['prohibitions'])}): " + (" | ".join(p[:140] for p in rules["prohibitions"][:8]) or "none found"),
                ""]
        if r["work"] and r["work"]["best"]:
            out.insert(len(out) - 1, f"- Local best: {r['work']['best']}")
        for k, v in r["elsewhere"].items():
            out.insert(len(out) - 1, f"- Work elsewhere ({k}): {', '.join(v['items'])}; last change {v['last_change_utc']}; "
                                     f"{v['best'] or 'no BEST line'}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="challenges")
    ap.add_argument("--also", action="append", default=[], help="another challenges/ folder to check for work")
    ap.add_argument("--md", help="write the markdown report here")
    ap.add_argument("--json", help="write the full inventory as JSON here")
    ap.add_argument("--now", help="reference time in UTC, ISO format (default: now)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a folder: {root}")
    now = datetime.fromisoformat(a.now).replace(tzinfo=timezone.utc) if a.now else datetime.now(timezone.utc)
    rows = scan(root, [Path(p).resolve() for p in a.also], now)
    report = markdown(rows, root, now)
    print(report)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(report + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
