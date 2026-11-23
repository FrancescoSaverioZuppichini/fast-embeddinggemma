"""Builds aspect-retrieval-5k, the whole dataset and its card, in one go: a text→image, near-duplicate and sketch
retrieval benchmark of 5,040 Open Images photos, 720 on each of the 7 canvases fast-embeddinggemma snaps a picture to,
so every aspect bucket is measured equally, searched among 20,000 more photos (the distractors).

Every stage keeps its result in --work, so a crash or a rerun never downloads a file or pays for a caption twice.

    uv run --group dataset scripts/build_dataset.py --work DIR --until sample      # pick the pictures
    uv run --group dataset scripts/build_dataset.py --work DIR --pilot 5           # caption 5, print the cost estimate
    uv run --group dataset scripts/build_dataset.py --work DIR                     # everything, saved locally
    uv run --group dataset scripts/build_dataset.py --work DIR --push int8.json    # published, eval JSONs as baselines
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from datasets import Image as ImageFeature
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from fast_embeddinggemma.modeling.canvas import CANVASES, pick_canvas
from fast_embeddinggemma.quality import HEADLINE

REPO = "Francesco/aspect-retrieval-5k"
SEED = 0
PER_CANVAS = 720
# [Agent] Spare pictures per canvas, taken in the same random order when one is dropped (bad download, near-copy of
# another, unsafe, no usable caption), so the set stays 720 per canvas without a second sampling.
RESERVE = 180
# [Agent] The fewest spares a canvas may have: tall portraits are rare (899 eligible in all of test and validation).
MIN_RESERVE = 60
THUMB, THUMB_QUALITY = 320, 78  # stash's previews: fit inside 320x320, never enlarged, WebP quality 78
# [Agent] The `image` column: the file as Open Images hosts it (1024 px on its long side for 99%), untouched; a
# bigger one is shrunk to 1024, never cropped.
STORED, STORED_QUALITY = 1024, 95
MIN_LONG = 640
DUPLICATE_BITS = 4  # dHash distance at or under which two pictures count as the same photo
MODEL = "openai/gpt-6-luna"
PROMPT_VERSION = 1
REQUESTS = 48  # OpenRouter calls in flight at once
SPLITS = ("test", "validation")
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
OI = "https://storage.googleapis.com/openimages/2018_04/{split}/{split}-images-with-rotation.csv"
LN = "https://storage.googleapis.com/localized-narratives/annotations/open_images_{split}_localized_narratives.jsonl"
S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
STAGES = ("meta", "narratives", "sizes", "sample", "fetch", "captions", "distractors", "table")

PROMPT = """You write search queries for a photo retrieval benchmark. Look at the photo and answer with JSON only:
{"short": ..., "sentence": ..., "unsafe": ...}
- "short": what a person would type into a photo search box to find exactly this photo: 3 to 8 words, lowercase, no
  punctuation, the main subject plus one or two details that set it apart from similar photos.
- "sentence": one plain sentence of 12 to 30 words describing the main subject, the setting and the details that make
  this photo distinct.
- "unsafe": true if the photo shows nudity, sexual content, graphic violence or gore, else false.
Describe only what is visible. Never start with "a photo of", "an image of" or "a picture of". Never mention the photo's
orientation, shape, size, resolution, quality, or that it is a photo. Do not guess who people are. Write in English."""


def canvas_name(grid):
    return f"{grid[0]}x{grid[1]}"


def rank(image_id):
    # [Agent] The order pictures are considered in: a hash of the seed and the id, so it does not depend on the order
    # of Open Images' CSV and the same seed always picks the same photos.
    return hashlib.sha256(f"{SEED}:{image_id}".encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append_jsonl(path, rows):
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_env(names):
    # [Agent] Keys come from the environment or the repo's gitignored .env, never from the command line.
    env = Path(__file__).resolve().parent.parent / ".env"
    pairs = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line) if env.exists() else {}
    return {name: os.environ.get(name) or pairs.get(name, "").strip().strip('"') for name in names}


def download(url, path):
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url, timeout=None, follow_redirects=True) as response, tmp.open("wb") as f:
        response.raise_for_status()
        for chunk in response.iter_bytes(1 << 20):
            f.write(chunk)
    tmp.rename(path)


def load_meta(work):
    """Open Images rows that may go in: CC BY 2.0 and stored upright (Rotation 0)."""
    rows = {}
    for split in SPLITS:
        path = work / f"{split}-images.csv"
        if not path.exists():
            download(OI.format(split=split), path)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["License"] == CC_BY and row["Rotation"] and float(row["Rotation"]) == 0:
                    rows[row["ImageID"]] = row | {"split": split}
    print(f"meta: {len(rows)} CC BY 2.0 pictures stored upright")
    return rows


LEADS = [
    re.compile(
        r"^(in|on) (this|the) (image|picture|photo|foreground|front|middle|center|centre|background),?\s*", re.I
    ),
    re.compile(r"^(on|at|to) the (left|right|top|bottom)( side)?( of (this|the) (image|picture|photo))?,?\s*", re.I),
    re.compile(r"^this (image|picture|photo) (consists of|contains|shows|is of|is|of)\s+", re.I),
    re.compile(
        r"^((i|we) can (see|find|notice|observe)|there (is|are)|it (seems|looks) like|this is|(i|we) see)( that)?,?\s+",
        re.I,
    ),
]


# [Agent] A sentence about where the photo was taken, not what it shows; the next one is used.
FILLER = re.compile(
    r"\b(might|may|seems to) be (taken|clicked|captured)|\btaken (indoors|outdoors|inside|outside)", re.I
)


def first_sentence(caption):
    # [Agent] Localized Narratives are spoken and transcribed ("In this image we can see a dog. ..."): the first
    # sentence that says what it shows, without its lead-ins, is the human query.
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    text = next((s for s in sentences if not FILLER.search(s)), sentences[0])
    while (stripped := next((lead.sub("", text, count=1) for lead in LEADS if lead.match(text)), text)) != text:
        text = stripped
    return text[:1].upper() + text[1:]


def load_narratives(work, ids):
    """One human caption per picture: the lowest annotator id's, for every picture that has one."""
    path = work / "narratives.jsonl"
    if not path.exists():
        kept = {}
        for split in SPLITS:
            with httpx.stream("GET", LN.format(split=split), timeout=None) as response:
                for line in response.iter_lines():
                    row = json.loads(line)
                    image_id, annotator = row["image_id"], row["annotator_id"]
                    if image_id in ids and (image_id not in kept or annotator < kept[image_id][0]):
                        kept[image_id] = (annotator, row["caption"])
        append_jsonl(path, ({"id": i, "narrative": c} for i, (_, c) in kept.items()))
    narratives = {row["id"]: row["narrative"] for row in read_jsonl(path)}
    print(f"narratives: {len(narratives)} pictures with a human caption")
    return narratives


def fit(size, longest):
    w, h = size
    scale = longest / max(w, h)
    return (w, h) if scale >= 1 else (max(1, round(w * scale)), max(1, round(h * scale)))


def is_eligible(size):
    # [Agent] Big enough to be shrunk for the thumbnail, and on the same canvas as original, stored copy and thumbnail:
    # a picture on a bucket's edge could otherwise be measured on one canvas and labelled with another.
    if not size or size.get("error") or size["format"] != "JPEG" or size["orientation"] != 1:
        return False
    w, h = size["w"], size["h"]
    if max(w, h) < MIN_LONG or size["mode"] not in ("RGB", "L"):
        return False
    return pick_canvas(w, h) == pick_canvas(*fit((w, h), STORED)) == pick_canvas(*fit((w, h), THUMB))


def read_size(client, row):
    url = S3.format(split=row["split"], id=row["ImageID"])
    for span in (32_768, 524_288, None):
        try:
            response = client.get(url, headers={"Range": f"bytes=0-{span - 1}"} if span else {})
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as im:
                orientation = im.getexif().get(274, 1)
                return {
                    "id": row["ImageID"],
                    "w": im.width,
                    "h": im.height,
                    "format": im.format,
                    "mode": im.mode,
                    "orientation": orientation,
                }
        except (OSError, httpx.HTTPError):
            continue
    return {"id": row["ImageID"], "error": "unreadable header"}


def load_sizes(work, meta, narratives):
    """Each candidate's size from the first bytes of its file, in rank order, until every canvas has enough."""
    path = work / "sizes.jsonl"
    sizes = {row["id"]: row for row in read_jsonl(path)}
    order = sorted((i for i in meta if i in narratives), key=rank)
    need = PER_CANVAS + RESERVE

    def counts():
        found = {g: 0 for g in CANVASES}
        for i in order:
            if i in sizes and is_eligible(sizes[i]):
                found[pick_canvas(sizes[i]["w"], sizes[i]["h"])] += 1
        return found

    with httpx.Client(timeout=60, limits=httpx.Limits(max_connections=64)) as client, ThreadPoolExecutor(64) as pool:
        todo = [i for i in order if i not in sizes]
        while todo and min(counts().values()) < need:
            batch, todo = todo[:4000], todo[4000:]
            rows = list(pool.map(lambda i: read_size(client, meta[i]), batch))
            append_jsonl(path, rows)
            sizes |= {row["id"]: row for row in rows}
            print(f"sizes: {len(sizes)} read, per canvas {list(counts().values())}", flush=True)
    return sizes


def pick(order_by_canvas, dropped):
    """The first PER_CANVAS pictures of each canvas that were not dropped, in rank order."""
    return {g: [i for i in ids if i not in dropped][:PER_CANVAS] for g, ids in order_by_canvas.items()}


def sample(work, meta, narratives, sizes):
    by_canvas = {g: [] for g in CANVASES}
    for i in sorted((i for i in meta if i in narratives and i in sizes and is_eligible(sizes[i])), key=rank):
        by_canvas[pick_canvas(sizes[i]["w"], sizes[i]["h"])].append(i)
    short = {canvas_name(g): len(ids) for g, ids in by_canvas.items() if len(ids) < PER_CANVAS + MIN_RESERVE}
    if short:
        raise SystemExit(f"sample: not enough pictures on {short}; read more sizes first")
    chosen = {canvas_name(g): ids[: PER_CANVAS + RESERVE] for g, ids in by_canvas.items()}
    (work / "sample.json").write_text(json.dumps(chosen))
    print(f"sample: {PER_CANVAS} + spares per canvas, {dict((g, len(ids)) for g, ids in chosen.items())}")
    return {g: chosen[canvas_name(g)] for g in CANVASES}


def fetch_one(client, row, size, folder):
    """None when the file is here, decodes whole and has the size its header scan read, else the picture's id. Open
    Images' MD5 is the Flickr original's, and the copies it hosts are re-encoded, so a full decode is the check. A
    network error stops the run, which resumes where it was."""
    path = folder / f"{row['ImageID']}.jpg"
    if path.exists():
        return None
    response = client.get(S3.format(split=row["split"], id=row["ImageID"]))
    if response.status_code in (403, 404):
        return row["ImageID"]
    data = response.raise_for_status().content
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.size != (size["w"], size["h"]):
                return row["ImageID"]
    except OSError:
        return row["ImageID"]
    path.write_bytes(data)
    return None


def fetch(work, meta, sizes, ids):
    """The files as Open Images hosts them; one that is broken or not the size its header said drops the picture."""
    folder = work / "originals"
    folder.mkdir(exist_ok=True)
    bad_path = work / "bad.json"
    bad = set(json.loads(bad_path.read_text())) if bad_path.exists() else set()
    todo = [i for i in ids if i not in bad]
    with httpx.Client(timeout=120, limits=httpx.Limits(max_connections=32)) as client, ThreadPoolExecutor(32) as pool:
        bad |= {i for i in pool.map(lambda i: fetch_one(client, meta[i], sizes[i], folder), todo) if i}
    bad_path.write_text(json.dumps(sorted(bad)))
    print(f"fetch: {len(list(folder.glob('*.jpg')))} files, {len(bad)} broken or not the size their header said")
    return bad


def encode(im, fmt, **options):
    out = io.BytesIO()
    im.save(out, fmt, **options)
    return out.getvalue()


def make_thumb(im):
    return encode(im.resize(fit(im.size, THUMB), Image.Resampling.LANCZOS), "WEBP", quality=THUMB_QUALITY)


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def thumb_of(work, image_id):
    path = work / "thumbs" / f"{image_id}.webp"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(make_thumb(open_rgb(work / "originals" / f"{image_id}.jpg")))
    return path.read_bytes()


HASHES = {}


def dhash(data):
    with Image.open(io.BytesIO(data)) as im:
        pixels = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return int(np.packbits((pixels[:, 1:] > pixels[:, :-1]).ravel()).view(">u8")[0])


def find_copies(work, order):
    """Pictures that repeat an earlier one in the given order (dHash distance ≤ DUPLICATE_BITS)."""
    kept, count, copies = np.empty(len(order), np.uint64), 0, set()
    for image_id in order:
        if image_id not in HASHES:
            HASHES[image_id] = dhash(thumb_of(work, image_id))
        bits = np.uint64(HASHES[image_id])
        if count and np.bitwise_count(kept[:count] ^ bits).min() <= DUPLICATE_BITS:
            copies.add(image_id)
        else:
            kept[count], count = bits, count + 1
    return copies


def parse_caption(text):
    """The caption JSON if it follows the rules, else None (the picture is asked again)."""
    try:
        caption = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    short, sentence = str(caption.get("short", "")).strip(), str(caption.get("sentence", "")).strip()
    short = re.sub(r"^(a |an )?(photo|image|picture) of ", "", short.lower())
    banned = r"\b(photo|image|picture|orientation|vertical|horizontal|aspect ratio)\b"
    if not 2 <= len(short.split()) <= 10 or not 8 <= len(sentence.split()) <= 40 or re.search(banned, short):
        return None
    return {"short": re.sub(r"[^\w\s'-]", "", short), "sentence": sentence, "unsafe": caption.get("unsafe") is True}


def ask(client, key, thumb, prompt, max_tokens=400):
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64," + base64.b64encode(thumb).decode()}},
    ]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "minimal"},
        "usage": {"include": True},
    }
    for attempt in range(6):
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
        )
        if response.status_code in (408, 429, 500, 502, 503, 504):
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"].get("content"), data.get("usage", {}), data.get("model")
    raise RuntimeError(f"OpenRouter kept failing: {response.status_code}")


def caption(work, client, key, image_id, note=None):
    """One caption request, kept with its raw answer and cost; up to 3 tries until it follows the rules."""
    rows = []
    for _ in range(3):
        text, usage, model = ask(client, key, thumb_of(work, image_id), PROMPT + (f"\n\n{note}" if note else ""))
        rows.append(
            {
                "id": image_id,
                "note": note,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "text": text,
                "usage": usage,
                "caption": parse_caption(text),
            }
        )
        if rows[-1]["caption"]:
            break
    return rows


def caption_all(work, items, notes=None):
    path = work / "captions.jsonl"
    done = {(row["id"], row["note"]) for row in read_jsonl(path)}
    todo = [(i, (notes or {}).get(i)) for i in items if (i, (notes or {}).get(i)) not in done]
    key = read_env(["OPENROUTER_API_KEY"])["OPENROUTER_API_KEY"]
    with httpx.Client(timeout=180) as client, ThreadPoolExecutor(REQUESTS) as pool:
        for rows in pool.map(lambda job: caption(work, client, key, *job), todo):
            append_jsonl(path, rows)
    return read_jsonl(path)


def latest_captions(rows):
    """Each picture's caption from its last request that followed the rules."""
    captions = {}
    for row in rows:
        if row["caption"]:
            captions[row["id"]] = row["caption"]
    return captions


def cost(rows):
    return sum(float(row["usage"].get("cost") or 0) for row in rows)


def pilot(work, order_by_canvas, count):
    items = [ids[k] for k in range(count) for ids in order_by_canvas.values() if k < len(ids)][:count]
    rows = [row for row in caption_all(work, items) if row["id"] in items]
    for row in rows:
        usage = row["usage"]
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "caption": row["caption"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cost": usage.get("cost"),
                }
            )
        )
    per = cost(rows) / len(items)
    print(f"pilot: ${per:.6f} a picture, so about ${per * PER_CANVAS * len(CANVASES) * 1.05:.2f} for the set")


SAFETY = (
    'Answer with JSON only: {"unsafe": true} if the photo shows nudity, sexual content, graphic violence or gore, '
    'else {"unsafe": false}.'
)
DISTRACTORS = 20_000


def check_one(work, client, key, image_id):
    """One distractor's safety answer, with every request's cost; one that never parses counts as unsafe."""
    usages = []
    for _ in range(3):
        text, usage, model = ask(client, key, thumb_of(work, image_id), SAFETY, 200)
        usages.append(usage)
        try:
            unsafe = json.loads(text).get("unsafe")
        except (TypeError, AttributeError, json.JSONDecodeError):
            unsafe = None
        if isinstance(unsafe, bool):
            break
    return {"id": image_id, "model": model, "text": text, "usages": usages, "unsafe": unsafe is not False}


def check_safety(work, ids):
    path = work / "safety.jsonl"
    done = {row["id"] for row in read_jsonl(path)}
    key = read_env(["OPENROUTER_API_KEY"])["OPENROUTER_API_KEY"]
    with httpx.Client(timeout=180) as client, ThreadPoolExecutor(REQUESTS) as pool:
        for row in pool.map(lambda i: check_one(work, client, key, i), [i for i in ids if i not in done]):
            append_jsonl(path, [row])
    return {row["id"]: row["unsafe"] for row in read_jsonl(path)}


def settle_distractors(work, meta, sizes, pool, targets):
    """DISTRACTORS more photos for the gallery, in rank order from the eligible ones the sample never touched, refilled
    until none is broken, a near-copy of a target or of an earlier distractor, or unsafe."""
    dropped, reasons = set(), {}
    for _ in range(20):
        chosen = [i for i in pool if i not in dropped][:DISTRACTORS]
        found, reason = fetch(work, meta, sizes, chosen) & set(chosen), "broken file"
        if not found:
            found = find_copies(work, targets + chosen) & set(chosen)
            reason = "look-alike of a test photo or an earlier distractor"
        if not found:
            found, reason = {i for i, flag in check_safety(work, chosen).items() if flag} & set(chosen), "unsafe"
        if not found:
            return chosen, reasons
        dropped |= found
        reasons[reason] = reasons.get(reason, 0) + len(found)
    raise SystemExit("distractors: the set did not stop changing in 20 rounds")


NOTE = 'Another photo already has the query "{short}". Write queries that find this photo and not that one.'


def settle(work, order_by_canvas, bad):
    """Caption the chosen pictures and refill from the spares until the set stops changing: copies of an earlier picture
    go before anything is paid for, then unsafe pictures and ones with no usable caption; a picture whose short query an
    earlier one already has is asked once more with that query named, and dropped if it still repeats it."""
    dropped = set(bad)
    for _ in range(20):
        chosen = pick(order_by_canvas, dropped)
        order = sorted((i for ids in chosen.values() for i in ids), key=rank)
        copies = find_copies(work, order)
        if copies:
            dropped |= copies
            continue
        rows = caption_all(work, order)
        captions = latest_captions(rows)
        failed = {i for i in order if i not in captions or captions[i]["unsafe"]}
        if failed:
            dropped |= failed
            continue
        holder, repeats = {}, {}
        for i in order:
            short = captions[i]["short"]
            if short in holder:
                repeats[i] = short
            else:
                holder[short] = i
        if not repeats:
            return chosen, captions, dropped
        asked = {row["id"] for row in rows if row["note"]}
        again = {i: NOTE.format(short=short) for i, short in repeats.items() if i not in asked}
        if again:
            caption_all(work, list(again), again)
        dropped |= {i for i in repeats if i in asked}
    raise SystemExit("settle: the set did not stop changing in 20 rounds")


def letterbox(im, aspect, fill):
    """The picture centred on a canvas of another shape, as a story or a square post re-shares it: black bars, or the
    picture itself blurred behind it."""
    w, h = im.size
    size = (w, round(w / aspect)) if w / h > aspect else (round(h * aspect), h)
    back = (
        Image.new("RGB", size) if fill == "black" else im.resize(size).filter(ImageFilter.GaussianBlur(max(size) / 30))
    )
    back.paste(im, ((size[0] - w) // 2, (size[1] - h) // 2))
    return back


def make_neardup(path, image_id):
    """A re-shared copy, made the way people re-post photos: a crop of 30–80% of the area with its shape nudged, mirrored
    half the time, colours shifted, sometimes grey, blurred, letterboxed into another shape or shrunk to a small repost,
    and saved as a JPEG of quality 20–80, then thumbnailed like the rest. Seeded by the id; the steps are kept."""
    rng = random.Random(f"{SEED}:{image_id}")
    im = open_rgb(path)
    w, h = im.size
    area, stretch = rng.uniform(0.3, 0.8), rng.uniform(0.67, 1.5)
    cw, ch = min(w, round(w * math.sqrt(area * stretch))), min(h, round(h * math.sqrt(area / stretch)))
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x, y, x + cw, y + ch))
    ops = {"crop": [x, y, cw, ch], "flip": rng.random() < 0.5}
    if ops["flip"]:
        im = ImageOps.mirror(im)
    for name, enhancer in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("saturation", ImageEnhance.Color),
    ):
        ops[name] = rng.uniform(0.6, 1.4)
        im = enhancer(im).enhance(ops[name])
    ops["grayscale"] = rng.random() < 0.15
    if ops["grayscale"]:
        im = ImageOps.grayscale(im).convert("RGB")
    ops["blur"] = rng.uniform(0.5, 2.0) if rng.random() < 0.3 else 0.0
    if ops["blur"]:
        im = im.filter(ImageFilter.GaussianBlur(ops["blur"]))
    boxed = rng.random() < 0.25
    ops["letterbox"] = rng.choice(["9:16", "1:1", "16:9"]) if boxed else ""
    ops["fill"] = rng.choice(["black", "blur"]) if boxed else ""
    if boxed:
        ratio = {"9:16": 9 / 16, "1:1": 1.0, "16:9": 16 / 9}[ops["letterbox"]]
        im = letterbox(im, ratio, ops["fill"])
    ops["shrunk"] = rng.random() < 0.3
    if ops["shrunk"]:
        im = im.resize(fit(im.size, 160), Image.Resampling.LANCZOS)
    ops["jpeg_quality"] = rng.randint(20, 80)
    im = open_rgb(io.BytesIO(encode(im, "JPEG", quality=ops["jpeg_quality"])))
    return make_thumb(im), ops


def make_sketch(thumb):
    """A clean edge map of the thumbnail: black lines on white, as a sketch would be."""
    gray = cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 50, 150)
    return cv2.imencode(".png", 255 - edges)[1].tobytes()


def make_stored(path):
    data = path.read_bytes()
    im = open_rgb(path)
    if max(im.size) <= STORED:
        return data
    return encode(im.resize(fit(im.size, STORED), Image.Resampling.LANCZOS), "JPEG", quality=STORED_QUALITY)


def blob(data):
    return {"bytes": data, "path": None}


def make_row(work, meta, narratives, captions, grid, image_id):
    path = work / "originals" / f"{image_id}.jpg"
    row, thumb = meta[image_id], thumb_of(work, image_id)
    neardup, ops = make_neardup(path, image_id)
    with Image.open(path) as im:
        width, height = im.size
    return {
        "id": image_id,
        "canvas": canvas_name(grid),
        "width": width,
        "height": height,
        "aspect": round(width / height, 4),
        "image": blob(make_stored(path)),
        "thumb": blob(thumb),
        "query_short": captions[image_id]["short"],
        "query_sentence": captions[image_id]["sentence"],
        "query_human": first_sentence(narratives[image_id]),
        "narrative": narratives[image_id],
        "neardup": blob(neardup),
        "neardup_ops": ops,
        "sketch": blob(make_sketch(thumb)),
        "author": row["Author"],
        "author_url": row["AuthorProfileURL"],
        "license": row["License"],
        "source_url": row["OriginalLandingURL"],
        "split": row["split"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_distractor_row(work, meta, sizes, image_id):
    path, row = work / "originals" / f"{image_id}.jpg", meta[image_id]
    width, height = sizes[image_id]["w"], sizes[image_id]["h"]
    return {
        "id": image_id,
        "canvas": canvas_name(pick_canvas(width, height)),
        "width": width,
        "height": height,
        "aspect": round(width / height, 4),
        "thumb": blob(thumb_of(work, image_id)),
        "author": row["Author"],
        "author_url": row["AuthorProfileURL"],
        "license": row["License"],
        "source_url": row["OriginalLandingURL"],
        "split": row["split"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_table(work, meta, narratives, sizes, captions, chosen, distractors):
    """Two splits in one dataset: `test`, the 5,040 photos with their queries, and `distractors`, gallery-only photos."""
    shape = {
        "id": Value("string"),
        "canvas": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
        "aspect": Value("float32"),
    }
    credit = {name: Value("string") for name in ("author", "author_url", "license", "source_url", "split", "sha256")}
    ops = {
        "crop": [Value("int32")],
        "flip": Value("bool"),
        "brightness": Value("float32"),
        "contrast": Value("float32"),
        "saturation": Value("float32"),
        "grayscale": Value("bool"),
        "blur": Value("float32"),
        "letterbox": Value("string"),
        "fill": Value("string"),
        "shrunk": Value("bool"),
        "jpeg_quality": Value("int32"),
    }
    test = Features(
        {
            **shape,
            "image": ImageFeature(),
            "thumb": ImageFeature(),
            "query_short": Value("string"),
            "query_sentence": Value("string"),
            "query_human": Value("string"),
            "narrative": Value("string"),
            "neardup": ImageFeature(),
            "neardup_ops": ops,
            "sketch": ImageFeature(),
            **credit,
        }
    )
    jobs = [(grid, i) for grid, ids in chosen.items() for i in ids]
    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(lambda job: make_row(work, meta, narratives, captions, *job), jobs))
        extra = list(pool.map(lambda i: make_distractor_row(work, meta, sizes, i), distractors))
    table = DatasetDict(
        {
            "test": Dataset.from_list(rows, features=test),
            "distractors": Dataset.from_list(extra, features=Features({**shape, "thumb": ImageFeature(), **credit})),
        }
    )
    table.save_to_disk(work / "table")
    print(f"table: {len(rows)} test rows and {len(extra)} distractors saved to {work / 'table'}")
    return table


CARD = """# Aspect Retrieval 5K

A photo search benchmark where **every aspect ratio counts the same**: 5,040 Open Images photos, 720 in each of seven
shape buckets from tall 9:16 portraits through square to wide 16:9 landscapes, searched among 25,040 photos.

Most photo sets are mostly 4:3 landscapes, so a model, or the way it resizes and cuts a picture into patches, can handle
tall or wide photos badly and the score barely moves. Here each bucket weighs the same and every score is also given per
bucket. The buckets are the seven canvases [fast-embeddinggemma](https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma)
uses, but nothing in the data depends on that package.

```python
from datasets import load_dataset

test = load_dataset("Francesco/aspect-retrieval-5k", "test", revision="v1.0", split="test")
distractors = load_dataset("Francesco/aspect-retrieval-5k", "distractors", revision="v1.0", split="distractors")
```

## What you search, and with what

The gallery is the 320 px thumbnail of all {gallery} photos: the 5,040 `test` photos and {distractors} `distractors`.
Every `test` photo comes with five ways of looking for it, and each has exactly one right answer, the photo it was made
from:

| query | what it is | headline score |
|---|---|---|
| `query_short` | 3 to 8 words, what you would type into a photo search box | R@1 |
| `query_sentence` | one sentence with the details that set the photo apart | R@1 |
| `query_human` | the first sentence a person said when describing the photo | R@10 |
| `neardup` | the photo re-posted: cropped, filtered, letterboxed, recompressed | R@1 |
| `sketch` | a line drawing of the photo | R@10 |

R@k is the share of queries whose photo is among the first k results. The reference scoring is
[`quality.py`](https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma/blob/main/src/fast_embeddinggemma/quality.py)
(`fast-embeddinggemma eval`): R@1, R@5, R@10 and MRR, overall and per bucket, each with a 95% bootstrap interval over
the queries. Any model works the same way: embed the gallery and the queries, rank, count.

## Baselines

{results}

## The buckets

A photo goes to the bucket nearest its width / height, measured in log space.

| bucket | width / height | typical | eligible in Open Images | test | distractors |
|---|---|---|---|---|---|
{buckets}

Tall portraits are rare in Open Images: `15x8` uses 720 of the 899 eligible ones, and the rest were spares, so none are
left for the distractors.

## How it was made, step by step

One script does all of it: [`scripts/build_dataset.py`]({script}). It keeps every download and every model answer on
disk, so it can be rerun and checked.

1. **Start from Open Images.** The test and validation photos (the same from V4 to V7): {total} Flickr photos, all CC
   BY 2.0.
2. **Keep the clean ones.** Only photos stored upright that have a human description (a Localized Narrative):
   {meta} left.
3. **Measure each photo's shape.** We read every photo's size and put it in its bucket. A photo is eligible if it is a
   JPEG, not turned by EXIF, at least {min_long} px on its long side, and still in the same bucket once shrunk to a
   thumbnail, so no photo near a bucket's edge is labelled with one bucket and measured in another.
4. **Pick 720 per bucket at random.** Photos are taken in a fixed random order (seed {seed}), with spares behind them
   in case a photo is dropped later.
5. **Download and check.** Each photo was downloaded from Open Images' own copy and decoded completely, to be sure it is
   whole and the size it claims.
6. **Make the thumbnails.** Each photo shrunk to fit inside {thumb}x{thumb} (Lanczos, WebP quality {quality}). These are
   what gets searched.
7. **Remove look-alikes.** If two photos look the same (their dHash differs in {bits} bits or fewer), the later one goes.
8. **Write the queries with an AI model.** `{model}` looked at each thumbnail and wrote the short query and the
   sentence, and said whether the photo is unsafe (nudity, sexual content, gore), in which case it was dropped. If two
   photos got the same short query, the second was asked again with the clash named. {requests} requests, ${cost_queries};
   every request and answer is in `build/captions.jsonl`.
9. **Take the human query.** The first sentence of the photo's Localized Narrative that says what it shows, without
   the spoken lead-in ("In this image we can see ..."). The whole narrative is in `narrative`.
10. **Make the near-duplicates.** Each photo re-posted the way people do it: a crop of 30-80% of it with its shape
    nudged, mirrored half the time, brightness, contrast and saturation each moved by up to 40%, grey 15% of the time,
    blurred 30% of the time, letterboxed into 9:16, 1:1 or 16:9 a quarter of the time (black bars or a blurred
    background), shrunk to a small 160 px repost 30% of the time, and saved as a JPEG of quality 20-80. The dice are
    seeded by the photo's id and every step is in `neardup_ops`.
11. **Make the sketches.** The thumbnail's edges (OpenCV: Gaussian blur 5x5 with sigma 1.2, Canny 50/150), black on
    white.
12. **Add {distractors} distractors.** More Open Images photos that passed steps 2 to 7, taken from the ones the sample
    never touched, in the same random order. They have no queries; they make the search as crowded as a real photo
    library. Each was checked against the 5,040 and the other distractors for look-alikes, and for safety by the same
    model ({safety_requests} requests, ${cost_safety}). They follow Open Images' natural mix of shapes, so most are 4:3
    landscapes.
13. **Score the baselines** with EmbeddingGemma 2 through fast-embeddinggemma, in INT8 and bf16.

<details>
<summary>The prompt that wrote the queries, word for word</summary>

```text
{prompt}
```
</details>

Photos dropped and replaced along the way:

{reasons}

## Columns

`test`, one row per photo:

| column | what |
|---|---|
| `id` | Open Images image id |
| `canvas` | the bucket, `rows x cols` |
| `width`, `height`, `aspect` | size of the downloaded file |
| `image` | the file as Open Images hosts it, untouched (1024 px on its long side for 99%); a larger one is shrunk to {stored} px, never cropped |
| `thumb` | the {thumb} px thumbnail that is searched |
| `query_short`, `query_sentence` | the AI queries |
| `query_human`, `narrative` | the human query and the whole Localized Narrative |
| `neardup`, `neardup_ops` | the re-posted copy and the exact steps that made it |
| `sketch` | the line drawing |
| `author`, `author_url`, `source_url`, `license` | attribution: every photo is CC BY 2.0 |
| `split`, `sha256` | the Open Images split it came from, and the hash of the downloaded file |

`distractors` has `id`, `canvas`, `width`, `height`, `aspect`, `thumb` and the same attribution columns. Each split is its
own config of the same name, because their columns differ.

## Limitations

- These are Flickr photos, not phone or social-media pictures, and few are 9:16 portraits.
- The distractors follow Open Images' mix of shapes, not the balanced one, because Open Images has too few tall
  portraits to balance them.
- The short and sentence queries come from one AI model and may share a style with captions some models were trained
  on. The human queries are the check on that.
- A query can fit more than one photo; some human descriptions are as general as "Trees.". Only the source photo counts
  as right, which is why the human and sketch scores lead with R@10.
- The thumbnails are made with Pillow and are not byte-identical to any particular app's thumbnails.

## License and attribution

The photos are CC BY 2.0 by their Flickr authors (`author`, `author_url` and `source_url` on every row). `thumb`,
`neardup` and `sketch` are adaptations under the same license. The Localized Narratives (Pont-Tuset et al., 2020) are CC
BY 4.0. The AI queries and everything else here are CC BY 4.0.

## Citation

```bibtex
@misc{{aspect_retrieval_5k,
  title = {{Aspect Retrieval 5K: a photo search benchmark balanced across aspect ratios}},
  author = {{Francesco}},
  year = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/Francesco/aspect-retrieval-5k}}}}
}}
```

Please also cite Open Images (Kuznetsova et al., 2020) and Localized Narratives (Pont-Tuset et al., 2020).
"""


def get_ranges():
    """Each canvas's aspect range (width / height): halfway, in log space, to the next canvas on either side."""
    ratios = sorted((g[1] / g[0], g) for g in CANVASES)
    ranges = {}
    for k, (ratio, grid) in enumerate(ratios):
        low = math.sqrt(ratios[k - 1][0] * ratio) if k else 0.0
        high = math.sqrt(ratio * ratios[k + 1][0]) if k + 1 < len(ratios) else math.inf
        ranges[grid] = (low, high)
    return ranges


def get_reasons(work, dropped, bad, captions):
    rows = read_jsonl(work / "captions.jsonl")
    noted, valid = {r["id"] for r in rows if r["note"]}, {r["id"] for r in rows if r["caption"]}
    reasons = {}
    for i in dropped:
        if i in bad:
            reason = "broken file"
        elif i not in valid:
            reason = "no usable answer in 3 tries"
        elif captions.get(i, {}).get("unsafe"):
            reason = "unsafe"
        elif i in noted:
            reason = "short query still the same as another photo's"
        else:
            reason = "look-alike of another photo"
        reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


def format_results(results):
    """One row per eval run with each query's headline score, then the first run's headline scores per bucket."""
    names = {
        "query_short": "short",
        "query_sentence": "sentence",
        "query_human": "human",
        "neardup": "near-dup",
        "sketch": "sketch",
    }
    head = " | ".join(f"{names[k]} {HEADLINE[k]}" for k in names)
    lines = [f"| model | {head} |", f"|---|{'---|' * len(names)}"]
    for setup, report in results:
        cells = " | ".join(f"{report[k]['all'][HEADLINE[k]][0]:.3f}" for k in names)
        lines.append(f"| EmbeddingGemma 2, {setup['precision']} | {cells} |")
    setup, report = results[0]
    canvases = [c for c in report["query_short"] if c != "all"]
    lines += [
        "",
        f"Per bucket (EmbeddingGemma 2, {setup['precision']}):",
        "",
        f"| query | {' | '.join(canvases)} |",
        f"|---|{'---|' * len(canvases)}",
    ]
    for kind in names:
        cells = " | ".join(format(report[kind][c][HEADLINE[kind]][0], ".3f") for c in canvases)
        lines.append(f"| {names[kind]} {HEADLINE[kind]} | {cells} |")
    return "\n".join(lines)


CARD_YAML = {
    "license": "cc-by-4.0",
    "language": ["en"],
    "pretty_name": "Aspect Retrieval 5K",
    "size_categories": ["10K<n<100K"],
    "task_categories": ["image-feature-extraction", "feature-extraction"],
    "tags": [
        "image-retrieval",
        "text-to-image-retrieval",
        "near-duplicate-detection",
        "sketch-based-image-retrieval",
        "aspect-ratio",
        "benchmark",
        "open-images",
    ],
}


def write_card(work, meta, sizes, chosen, captions, dropped, bad, distractors, results, commit):
    queries, checks = read_jsonl(work / "captions.jsonl"), read_jsonl(work / "safety.jsonl")
    total = sum(1 for split in SPLITS for _ in csv.DictReader((work / f"{split}-images.csv").open(newline="")))
    eligible, extra = {g: 0 for g in CANVASES}, {g: 0 for g in CANVASES}
    for i in meta:
        if i in sizes and is_eligible(sizes[i]):
            eligible[pick_canvas(sizes[i]["w"], sizes[i]["h"])] += 1
    for i in distractors["ids"]:
        extra[pick_canvas(sizes[i]["w"], sizes[i]["h"])] += 1
    names = {
        (11, 11): "square",
        (10, 13): "4:3, 5:4",
        (9, 14): "3:2, 16:10",
        (8, 15): "16:9 and wider",
        (13, 10): "3:4, 4:5",
        (14, 9): "2:3",
        (15, 8): "9:16 and taller",
    }
    buckets = "\n".join(
        f"| `{canvas_name(g)}` | {low:.3f} - {'inf' if high == math.inf else f'{high:.3f}'} | {names[g]} "
        f"| {eligible[g]:,} | {len(chosen[g])} | {extra[g]:,} |"
        for g, (low, high) in get_ranges().items()
    )
    dropped_test = get_reasons(work, dropped, bad, captions)
    reasons = [f"- test, {reason}: {n}" for reason, n in sorted(dropped_test.items())]
    reasons += [f"- distractors, {reason}: {n}" for reason, n in sorted(distractors["reasons"].items())]
    safety_cost = sum(float(u.get("cost") or 0) for row in checks for u in row["usages"])
    return CARD.format(
        results=format_results(results),
        buckets=buckets,
        total=f"{total:,}",
        meta=f"{len(meta):,}",
        seed=SEED,
        min_long=MIN_LONG,
        thumb=THUMB,
        quality=THUMB_QUALITY,
        bits=DUPLICATE_BITS,
        model=MODEL,
        requests=f"{len(queries):,}",
        cost_queries=f"{cost(queries):.2f}",
        distractors=f"{len(distractors['ids']):,}",
        gallery=f"{len(distractors['ids']) + sum(len(ids) for ids in chosen.values()):,}",
        safety_requests=f"{sum(len(row['usages']) for row in checks):,}",
        cost_safety=f"{safety_cost:.2f}",
        prompt=PROMPT,
        reasons="\n".join(reasons),
        stored=STORED,
        script=f"https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma/blob/{commit}/scripts/build_dataset.py",
    )


def get_commit():
    """The commit the card links to; the script must be committed so the link shows the code that built the set."""
    repo = Path(__file__).resolve().parent.parent
    if subprocess.run(["git", "-C", repo, "status", "--porcelain", "scripts"], capture_output=True, text=True).stdout:
        raise SystemExit("push: commit scripts/ first, the card links to the commit")
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def push(work, card_text):
    """Both splits, the card (the Hub's generated YAML kept, ours added), every model answer, then the v1.0 tag."""
    from huggingface_hub import DatasetCard, HfApi

    token = read_env(["HF_TOKEN"])["HF_TOKEN"]
    # [Agent] One config per split: the Hub needs every split of a config to share its columns, and the distractors
    # have no queries.
    for name, split in load_from_disk(work / "table").items():
        split.push_to_hub(
            REPO,
            config_name=name,
            split=name,
            private=False,
            max_shard_size="500MB",
            token=token,
            commit_message=f"aspect-retrieval-5k v1.0: {name}",
        )
    card = DatasetCard.load(REPO, token=token)
    for key, value in CARD_YAML.items():
        setattr(card.data, key, value)
    # [Agent] The viewer and a bare load_dataset open on test, the split with the queries.
    next(c for c in card.data.configs if c["config_name"] == "test")["default"] = True
    card.text = card_text
    card.push_to_hub(REPO, token=token, commit_message="aspect-retrieval-5k v1.0: the card")
    api = HfApi(token=token)
    for name in ("captions.jsonl", "safety.jsonl"):
        api.upload_file(
            path_or_fileobj=work / name,
            path_in_repo=f"build/{name}",
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"aspect-retrieval-5k v1.0: every request and answer ({name})",
        )
    api.create_tag(REPO, tag="v1.0", repo_type="dataset")
    print(f"push: https://huggingface.co/datasets/{REPO} at v1.0")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--until", choices=STAGES, default="table")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--push", nargs="+", metavar="RESULTS", help="publish, with these eval JSONs as baselines")
    args = parser.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    stop = STAGES.index(args.until)
    meta = load_meta(work)
    if stop < 1:
        return
    narratives = load_narratives(work, set(meta))
    if stop < 2:
        return
    sizes = load_sizes(work, meta, narratives)
    if stop < 3:
        return
    order_by_canvas = sample(work, meta, narratives, sizes)
    if stop < 4 and not args.pilot:
        return
    bad = fetch(work, meta, sizes, [i for ids in order_by_canvas.values() for i in ids])
    if args.pilot:
        return pilot(work, {g: [i for i in ids if i not in bad] for g, ids in order_by_canvas.items()}, args.pilot)
    if stop < 5:
        return
    chosen, captions, dropped = settle(work, order_by_canvas, bad)
    rows = read_jsonl(work / "captions.jsonl")
    print(f"captions: {len(rows)} requests, ${cost(rows):.2f}, {len(dropped)} pictures dropped")
    if stop < 6:
        return
    taken = {i for ids in order_by_canvas.values() for i in ids}
    pool = sorted((i for i in meta if i in sizes and is_eligible(sizes[i]) and i not in taken), key=rank)
    targets = sorted((i for ids in chosen.values() for i in ids), key=rank)
    extra, reasons = settle_distractors(work, meta, sizes, pool, targets)
    print(f"distractors: {len(extra)} chosen, dropped {reasons}")
    if stop < 7:
        return
    if not (work / "table").exists():
        build_table(work, meta, narratives, sizes, captions, chosen, extra)
    if args.push:
        results = [(r["setup"], r["report"]) for r in (json.loads(Path(f).read_text()) for f in args.push)]
        distractors = {"ids": extra, "reasons": reasons}
        card = write_card(work, meta, sizes, chosen, captions, dropped, bad, distractors, results, get_commit())
        (work / "README.md").write_text(card)
        push(work, card)


if __name__ == "__main__":
    main()
