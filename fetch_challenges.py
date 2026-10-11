"""Fetch the latest Axo challenge posts from the "AXO Challanges" Telegram channel into challenges/.

Logs in with your personal Telegram account through Telethon (MTProto user login, not the Bot API).
The first run asks for your phone number and the login code Telegram sends you (and your 2FA password
if you have one), then keeps the login in fetch_challenges.session so later runs need no code.

Each challenge is one channel post: its text holds the status, title, baseline card and dataset commands,
the attached Markdown file is the full description, and the live leaderboard is the bot's comment under it.
(The bot's private chat only has submission/payout messages, never challenge content.)

Usage:
    python fetch_challenges.py        # latest 4 challenge posts
    python fetch_challenges.py 10     # latest 10 challenge posts

For each challenge, challenges/<title-slug>/ gets:
    <date>_<time>_msg<id>.md (+ attachment)    the post as first downloaded (UTC); skipped if already saved
    description.md, baseline.md, kaggle-notebook.md, public-leaderboard.md
                                               refreshed from the current post and comment when they change
kaggle-notebook.md keeps everything from "# Code 3" on, so your solution cell survives the hourly link refresh.
"""
import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "challenges"
SESSION = ROOT / "fetch_challenges"  # Telethon adds the .session suffix

CHALLENGE_FILES = ["description.md", "baseline.md", "kaggle-notebook.md", "public-leaderboard.md"]
STATUS_COMMENT = re.compile(r"Submission Credits:|solvers beat AI")


def load_config():
    load_dotenv(ROOT / ".env")
    missing = [key for key in ("API_ID", "API_HASH") if not os.getenv(key)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)} (copy .env.example to .env and fill it in)")
    # The channel has no public @username, so it is found by its title (or numeric id) among your chats.
    return int(os.environ["API_ID"]), os.environ["API_HASH"], os.getenv("CHANNEL", "AXO Challanges")


async def find_channel(client, name):
    async for dialog in client.iter_dialogs():
        if dialog.name == name or str(dialog.id) == name:
            return dialog.entity
    sys.exit(f"No chat named {name!r} in your Telegram account (set CHANNEL in .env)")


def parse_post(text):
    """Split a challenge post into its parts; returns None for posts without a baseline card (not a challenge)."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r"Domain[ \t]+\S", line)), None)
    end = next((i for i, line in enumerate(lines) if re.match(r"Baseline[ \t]+\S", line)), None)
    if start is None or end is None or end < start:
        return None
    title = next((line.strip() for line in lines[:start] if line.strip() and "Status:" not in line), None)
    if title is None:
        return None
    return {
        "title": title,
        "status": next((line.split("Status:", 1)[1].strip() for line in lines if "Status:" in line), "?"),
        "card": "\n".join(lines[start:end + 1]),
        "curl": next((line for line in lines if line.startswith("!curl ")), None),
        "unzip": next((line for line in lines if line.startswith("!unzip ")), None),
    }


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def safe_name(name):
    return re.sub(r"[^\w.-]+", "_", name).strip("._") or "file"


def update(path, content):
    """Write content only when it differs from what is on disk."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return "unchanged"
    path.write_text(content, encoding="utf-8")
    return "written"


def update_notebook(path, curl, unzip):
    # The dataset link changes every hour; refresh Code 1/2 but keep your own cells from "# Code 3" on.
    cells = f"# Code 1\n{curl}\n" + (f"\n# Code 2\n{unzip}\n" if unzip else "")
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    own = re.search(r"^# Code 3", existing, re.M)
    return update(path, cells + ("\n" + existing[own.start():] if own else ""))


async def save_attachment(client, msg, folder, stem):
    """Download the post's photo or document next to its archive file; returns the path or None."""
    # Link previews, polls, etc. are not files (msg.photo would also return a link-preview image).
    if not isinstance(msg.media, (MessageMediaPhoto, MessageMediaDocument)) or msg.file is None:
        return None
    name = safe_name(msg.file.name) if msg.file.name else f"attachment{msg.file.ext or ''}"
    target = folder / f"{stem}_{name}"
    part = target.with_name(target.name + ".part")
    await client.download_media(msg, file=str(part))
    part.replace(target)  # only a complete download gets the final name
    return target


async def latest_status(client, channel, post):
    """Newest leaderboard/status comment the bot posted under the challenge, or None."""
    if not post.replies:  # comments not enabled on this post
        return None
    async for reply in client.iter_messages(channel, reply_to=post.id):
        if STATUS_COMMENT.search(reply.raw_text or ""):
            return reply.raw_text
    return None


async def save_challenge(client, channel, channel_name, msg, post):
    folder = OUT_DIR / slugify(post["title"])
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{msg.date:%Y-%m-%d_%H-%M-%S}_msg{msg.id}"
    archive = folder / f"{stem}.md"
    if archive.exists():
        attachment = next((p for p in folder.glob(f"{stem}_*") if p.suffix != ".part"), None)
        saved = "already downloaded, skipped"
    else:
        # Attachment first, archive last: the archive file marks the post as fully downloaded.
        attachment = await save_attachment(client, msg, folder, stem)
        header = f"<!-- Telegram post {msg.id} in {channel_name}, {msg.date:%Y-%m-%d %H:%M:%S} UTC"
        if attachment is not None:
            header += f", attachment: {attachment.name}"
        archive.write_text(f"{header} -->\n\n{msg.raw_text}\n", encoding="utf-8")
        saved = "saved"

    results = {"baseline.md": update(folder / "baseline.md", post["card"] + "\n")}
    if attachment is not None and attachment.suffix.lower() == ".md":
        results["description.md"] = update(folder / "description.md", attachment.read_text(encoding="utf-8"))
    if post["curl"]:
        results["kaggle-notebook.md"] = update_notebook(folder / "kaggle-notebook.md", post["curl"], post["unzip"])
    board = await latest_status(client, channel, msg)
    if board:
        results["public-leaderboard.md"] = update(folder / "public-leaderboard.md", board.strip() + "\n")

    print(f"\n[{post['status']}] {post['title']}")
    print(f"  {folder.relative_to(ROOT)}/{archive.name}" + (f" + {attachment.name}" if attachment else "") + f": {saved}")
    for name in CHALLENGE_FILES:
        print(f"  {name:<22} {results.get(name, 'MISSING (not found in the post)')}")


async def fetch(limit):
    api_id, api_hash, channel_name = load_config()
    async with TelegramClient(str(SESSION), api_id, api_hash) as client:
        channel = await find_channel(client, channel_name)
        found = 0
        async for msg in client.iter_messages(channel):  # newest first
            post = parse_post(msg.raw_text or "")
            if post is None:
                continue  # announcements and other non-challenge posts
            await save_challenge(client, channel, channel_name, msg, post)
            found += 1
            if found == limit:
                break
        print(f"\n{found} challenge post(s) from {channel_name!r} in {OUT_DIR.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description="Fetch the latest Axo challenge posts from the Telegram channel.")
    parser.add_argument("limit", nargs="?", type=int, default=4,
                        help="number of latest challenge posts to fetch (default: 4)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be at least 1")
    asyncio.run(fetch(args.limit))


if __name__ == "__main__":
    main()
