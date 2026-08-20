#!/usr/bin/env python3
"""
collect_news.py — Nepali MANUAL corpus collector: news (Onlinekhabar +
Khabarhub).

"news" is a SUBCATEGORY of the "manual" top-level corpus source (the other
top-level source is "indiccorp") — never a separate top-level source. Raw
output lives under nepali/data/raw/manual/news/.

-----------------------------------------------------------------------------
SCOPE (rewritten 2026-08-20)
-----------------------------------------------------------------------------
The Nepali manual corpus is exactly:

    manual/
    └── news/
        ├── onlinekhabar   (nepali/data/raw/manual/news/onlinekhabar_*.jsonl)
        └── khabarhub      (nepali/data/raw/manual/news/khabarhub_*.jsonl)

Wikipedia (collect_wikipedia.py) and Setopati were REMOVED from the active
pipeline on 2026-08-20:
  * Wikipedia's collector script and already-downloaded dump/extracted
    text were deleted outright — a full pipeline run measured Wikipedia
    contributing only 8.76M of 75.27M manual tokens, a small fraction next
    to onlinekhabar/khabarhub's growing volume.
  * Setopati was dropped because its discoverable-URL pool (homepage +
    category-page crawling — it has no article sitemap/API) was nearly
    exhausted: a full-depth run (500 URL cap, 8 pages/category) found
    only ~202 candidate URLs, almost all already collected.

government/textbooks/parliament/other_official were explored in an even
earlier iteration and are also OUT OF SCOPE — their collectors live under
nepali/scripts/manual/_deprecated_out_of_scope/ and are not invoked by
anything in the active pipeline (see nepali_data.yaml's `manual:` block).

-----------------------------------------------------------------------------
COLLECTION MECHANISM
-----------------------------------------------------------------------------
Both sources are collected via the SAME mechanism: each exposes an open,
unauthenticated WordPress REST API (`/wp-json/wp/v2/posts`) that returns
full article HTML directly in the JSON response, so no page-scraping is
needed (or possible to get wrong) for either source.

  * Onlinekhabar (onlinekhabar.com): robots.txt (`User-agent: *`) has NO
    disallow rules at all — re-verified live on 2026-08-18.
  * Khabarhub (khabarhub.com): robots.txt (`User-agent: *`) disallows only
    /wp-admin/ — verified live 2026-08-19. No AI-crawler user-agent block,
    no content-signal restriction. Confirmed live: 260,328 posts, all
    Devanagari-script Nepali.

Do NOT extend this collector to other sites without independently
checking that site's robots.txt/terms first — see nepali/friend's/
collect_news.py for examples of sites that were checked and REJECTED for
an explicit AI-crawler disallow (bbc.com, bhaskar.com, gorkhapatraonline.com)
or an ambiguous/unset content-signal (ekantipur.com).

-----------------------------------------------------------------------------
OUTPUT FORMAT
-----------------------------------------------------------------------------
Sharded JSONL (~max_docs_per_shard documents per shard file), one shard
sequence per source_name — genuinely separate files, not a shared prefix:

    nepali/data/raw/manual/news/onlinekhabar_00000.jsonl, _00001.jsonl, ...
    nepali/data/raw/manual/news/khabarhub_00000.jsonl, _00001.jsonl, ...

Each line is one document record using the SAME common schema fields as
every other manual/collect_*.py collector (manual_common.DocumentMetadata)
PLUS the article text itself and a deterministic doc_id — unlike the
single-document collectors (government/textbooks/parliament), a per-
article MetadataStore JSONL rewrite would not scale to tens/hundreds of
thousands of articles, so the metadata travels WITH the document in the
same shard line instead of a separate sidecar file:

    {
      "doc_id": "...", "text": "...",
      "data_source": "manual", "source_type": "news",
      "source_name": "onlinekhabar" | "khabarhub",
      "title": "...", "source_url": "...", "language": "ne",
      "document_type": "html_article",
      "collection_date": "YYYY-MM-DD",
      "original_filename": "", "local_filename": "",
      "license_or_access_basis": "...",
      "extraction_method": "...", "extraction_status": "success"
    }

original_filename/local_filename are always "" here — no raw HTML/JSON is
kept on disk per article (would not scale); only the extracted text is
kept, same as every field the schema defines still being present and
truthful about what was and wasn't preserved.

license_or_access_basis records ONLY what is actually verified (the
robots.txt check above) — never an invented "open license"/"public
domain" claim. News articles are copyrighted by their publishers; access
here is on the basis of a permissive robots.txt for automated collection,
not a content license grant.

-----------------------------------------------------------------------------
RESUMABILITY
-----------------------------------------------------------------------------
At startup, every existing shard for a source_name is scanned once to
build an in-memory `source_url -> already collected` set (a few hundred
thousand URLs is a few tens of MB — safe on an 8GB machine) so already-
collected articles are skipped without re-fetching them. Each source
persists its own next-page cursor to
nepali/data/raw/manual/news/collection_state.json (onlinekhabar_next_page,
khabarhub_next_page — independent keys) so a restart resumes pagination
near where it left off instead of re-walking already-exhausted pages just
to rediscover already-known URLs.

--overwrite deletes and re-collects only the source_name(s) actually being
run in that invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(__file__))
from manual_common import (  # noqa: E402
    NEWS_SOURCE_NAMES, build_arg_parser, collection_date_now, load_config,
    probe_url, setup_logger,
)

SOURCE_TYPE = "news"

USER_AGENT = (
    "LMA-IDV-NepaliCorpus/1.0 (educational/research corpus collection; "
    "see nepali/scripts/manual/collect_news.py)"
)
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQ = 1.0
MIN_ARTICLE_CHARS = 300
MAX_FETCH_RETRIES = 3

ONLINEKHABAR_API = "https://www.onlinekhabar.com/wp-json/wp/v2/posts"
ONLINEKHABAR_PER_PAGE = 100
ONLINEKHABAR_LICENSE_BASIS = (
    "onlinekhabar.com robots.txt (User-agent: *) has no Disallow rules "
    "(verified live 2026-08-18); its /wp-json/wp/v2/posts REST API is "
    "open and unauthenticated. Articles remain copyrighted by "
    "Onlinekhabar; this records ONLY that automated collection is not "
    "disallowed, not a content license grant."
)

KHABARHUB_API = "https://khabarhub.com/wp-json/wp/v2/posts"
KHABARHUB_PER_PAGE = 100
KHABARHUB_LICENSE_BASIS = (
    "khabarhub.com robots.txt (User-agent: *) disallows only /wp-admin/ "
    "(verified live 2026-08-19); its /wp-json/wp/v2/posts REST API is "
    "open and unauthenticated, no AI-crawler user-agent block, no "
    "content-signal restriction. Articles remain copyrighted by "
    "Khabarhub; this records ONLY that automated collection is not "
    "disallowed, not a content license grant."
)

# One entry per NEWS_SOURCE_NAMES member — the single place that maps a
# source_name to its API endpoint/page-size/license text, so
# run_collect()/run_dry_run()/run_smoke_test() all iterate the same list
# instead of hand-duplicating an onlinekhabar block and a khabarhub block.
SOURCES = {
    "onlinekhabar": {
        "api_url": ONLINEKHABAR_API,
        "per_page": ONLINEKHABAR_PER_PAGE,
        "license_basis": ONLINEKHABAR_LICENSE_BASIS,
    },
    "khabarhub": {
        "api_url": KHABARHUB_API,
        "per_page": KHABARHUB_PER_PAGE,
        "license_basis": KHABARHUB_LICENSE_BASIS,
    },
}
assert tuple(SOURCES.keys()) == NEWS_SOURCE_NAMES, (
    "SOURCES keys must match manual_common.NEWS_SOURCE_NAMES exactly"
)


# ---------------------------------------------------------------------------
# Sharded JSONL writer
# ---------------------------------------------------------------------------

class JsonlShardWriter:
    """Append-only sharded JSONL writer. Resumes an existing shard
    sequence for *prefix* (does not truncate/overwrite it) unless the
    caller has already deleted the prior shards (see --overwrite)."""

    def __init__(self, out_dir: Path, prefix: str, max_docs_per_shard: int = 20_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_docs_per_shard = max_docs_per_shard
        self.total_docs = 0
        self.total_text_chars = 0

        existing = sorted(self.out_dir.glob(f"{prefix}_*.jsonl"))
        self.shard_idx = 0
        self.docs_in_shard = 0
        if existing:
            last = existing[-1]
            try:
                self.shard_idx = int(last.stem.rsplit("_", 1)[-1])
            except ValueError:
                self.shard_idx = len(existing) - 1
            with open(last, "r", encoding="utf-8") as f:
                self.docs_in_shard = sum(1 for line in f if line.strip())
            if self.docs_in_shard >= max_docs_per_shard:
                self.shard_idx += 1
                self.docs_in_shard = 0

        self._f = open(self._shard_path(), "a", encoding="utf-8")

    def _shard_path(self) -> Path:
        return self.out_dir / f"{self.prefix}_{self.shard_idx:05d}.jsonl"

    def write(self, record: Dict[str, Any]) -> None:
        if self.docs_in_shard >= self.max_docs_per_shard:
            self._f.close()
            self.shard_idx += 1
            self.docs_in_shard = 0
            self._f = open(self._shard_path(), "a", encoding="utf-8")
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        self.docs_in_shard += 1
        self.total_docs += 1
        self.total_text_chars += len(record.get("text", ""))

    def close(self) -> None:
        self._f.close()


def load_collected_urls(out_dir: Path, prefix: str) -> Set[str]:
    """Scan every existing shard for *prefix* once and return the set of
    already-collected source_urls, for O(1) skip checks during collection."""
    urls: Set[str] = set()
    for shard in sorted(Path(out_dir).glob(f"{prefix}_*.jsonl")):
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url = rec.get("source_url")
                if url:
                    urls.add(url)
    return urls


def delete_existing_shards(out_dir: Path, prefix: str) -> int:
    n = 0
    for shard in Path(out_dir).glob(f"{prefix}_*.jsonl"):
        shard.unlink()
        n += 1
    return n


def make_doc_id(source_name: str, source_url: str) -> str:
    return hashlib.sha256(f"{source_name}:{source_url}".encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# HTTP helpers (retry + rate limiting)
# ---------------------------------------------------------------------------

def polite_get(url: str, logger, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            resp = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None  # exhausted pagination, not a transient error
            logger.warning(f"GET {url} -> HTTP {resp.status_code} (attempt {attempt}/{MAX_FETCH_RETRIES})")
        except requests.RequestException as exc:
            logger.warning(f"GET {url} failed: {exc} (attempt {attempt}/{MAX_FETCH_RETRIES})")
        if attempt < MAX_FETCH_RETRIES:
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_from_html_fragment(html: str) -> str:
    """For the WordPress REST API, which returns just the article body
    HTML (no surrounding page chrome to strip)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    paras = soup.find_all("p")
    return " ".join(p.get_text(separator=" ") for p in paras) if paras else soup.get_text(separator=" ")


def clean_article_text(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and len(ln) > 3]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# WordPress REST API collection (shared by onlinekhabar + khabarhub)
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def collect_wordpress_source(
    source_name: str,
    api_url: str,
    per_page: int,
    license_basis: str,
    max_posts: int,
    writer: JsonlShardWriter,
    already_collected: Set[str],
    state: Dict[str, Any],
    state_path: Path,
    logger,
) -> Tuple[int, int, int]:
    """Generic WordPress-REST-API collector, shared by onlinekhabar.com and
    khabarhub.com (both open /wp-json/wp/v2/posts endpoints, both
    returning full article HTML in the response). *source_name* is used
    for the state-cursor key, log lines, AND the written record's
    source_name field — each source writes into its own
    {source_name}_*.jsonl shard sequence.

    Returns (fetched, kept, skipped_already_collected). Persists its
    next-page cursor to state_path after every API page so a restart
    resumes near where it left off instead of re-walking exhausted
    pages."""
    fetched = kept = skipped = 0
    state_key = f"{source_name}_next_page"
    page = int(state.get(state_key, 1))

    while fetched < max_posts:
        resp = polite_get(api_url, logger, params={
            "per_page": per_page, "page": page,
        })
        if resp is None:
            logger.info(f"  [{source_name}] page {page} unreachable/exhausted; stopping.")
            break
        try:
            posts = resp.json()
        except ValueError:
            logger.warning(f"  [{source_name}] page {page}: non-JSON response; stopping.")
            break
        if not posts or not isinstance(posts, list):
            logger.info(f"  [{source_name}] page {page}: no more posts; pagination exhausted.")
            break

        for p in posts:
            link = p.get("link")
            if not link or p.get("type") not in (None, "post"):
                continue  # reject obvious non-article API content
            if link in already_collected:
                skipped += 1
                continue
            fetched += 1
            html = (p.get("content") or {}).get("rendered", "")
            cleaned = clean_article_text(extract_from_html_fragment(html))
            if len(cleaned) < MIN_ARTICLE_CHARS:
                continue
            title_html = (p.get("title") or {}).get("rendered", "")
            title = BeautifulSoup(title_html, "html.parser").get_text(separator=" ").strip() or link
            record = {
                "doc_id": make_doc_id(source_name, link),
                "text": cleaned,
                "data_source": "manual",
                "source_type": SOURCE_TYPE,
                "source_name": source_name,
                "title": title,
                "source_url": link,
                "language": "ne",
                "document_type": "html_article",
                "collection_date": collection_date_now(),
                "original_filename": "",
                "local_filename": "",
                "license_or_access_basis": license_basis,
                "extraction_method": "wordpress_rest_api_content_field",
                "extraction_status": "success",
            }
            writer.write(record)
            already_collected.add(link)
            kept += 1
            if fetched >= max_posts:
                break

        page += 1
        state[state_key] = page
        save_state(state_path, state)
        if fetched and fetched % 500 < per_page:
            logger.info(f"  [{source_name}] {fetched} fetched, {kept} kept so far (page {page})")
        time.sleep(SLEEP_BETWEEN_REQ)

    return fetched, kept, skipped


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _news_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("manual", {}).get("news", {})


def _resolve_out_dir(config: Dict[str, Any], repo_root: Path) -> Path:
    news_cfg = _news_cfg(config)
    return repo_root / news_cfg.get("output_dir", "nepali/data/raw/manual/news")


def _max_posts_cli_override(args, source_name: str) -> Optional[int]:
    return {"onlinekhabar": args.max_onlinekhabar, "khabarhub": args.max_khabarhub}[source_name]


# ---------------------------------------------------------------------------
# Dry-run / smoke-test
# ---------------------------------------------------------------------------

def run_dry_run(config: Dict[str, Any], repo_root: Path, args, logger) -> None:
    news_cfg = _news_cfg(config)
    out_dir = _resolve_out_dir(config, repo_root)

    logger.info("=" * 70)
    logger.info("DRY RUN — news collector — no network requests, no files written")
    logger.info("=" * 70)
    logger.info(f"news.enabled:             {news_cfg.get('enabled', True)}")
    logger.info(f"Output dir:               {out_dir}")

    for source_name in NEWS_SOURCE_NAMES:
        cfg = news_cfg.get(source_name, {})
        cli_override = _max_posts_cli_override(args, source_name)
        effective = cli_override if cli_override is not None else cfg.get("max_posts", 0)
        existing = load_collected_urls(out_dir, source_name) if out_dir.exists() else set()
        shards = sorted(out_dir.glob(f"{source_name}_*.jsonl")) if out_dir.exists() else []
        logger.info("-" * 70)
        logger.info(f"[{source_name}] enabled:              {cfg.get('enabled', True)}")
        logger.info(f"[{source_name}] target_tokens (plan):  {cfg.get('target_tokens', 'n/a')}")
        logger.info(f"[{source_name}] max_posts (effective): {effective}")
        logger.info(f"[{source_name}] already collected:     {len(existing):,} URL(s) across existing shards")
        logger.info(f"[{source_name}] existing shard files:  {len(shards)}")

    logger.info("=" * 70)
    logger.info("Dry run complete — no network I/O performed.")


def run_smoke_test(config: Dict[str, Any], logger) -> None:
    logger.info("=" * 70)
    logger.info("SMOKE TEST — news collector — connectivity check only")
    logger.info("=" * 70)

    all_reachable = True
    for source_name, src in SOURCES.items():
        info = probe_url(src["api_url"] + "?per_page=1&page=1", logger,
                          headers={"User-Agent": USER_AGENT})
        logger.info(f"[{source_name}] reachable: {info.get('reachable')} "
                    f"(status={info.get('status_code')})")
        all_reachable = all_reachable and bool(info.get("reachable"))

    if not all_reachable:
        logger.error("SMOKE TEST FAILED — one or more news sources unreachable.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("SMOKE TEST PASSED — all news sources reachable. No articles collected.")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Real collection
# ---------------------------------------------------------------------------

def run_collect(config: Dict[str, Any], repo_root: Path, args, logger) -> None:
    news_cfg = _news_cfg(config)
    if not news_cfg.get("enabled", True):
        logger.info("manual.news.enabled is false — nothing to do.")
        return

    out_dir = _resolve_out_dir(config, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "collection_state.json"
    state = load_state(state_path)

    summary: Dict[str, Any] = {"generated_at": collection_date_now()}

    for source_name in NEWS_SOURCE_NAMES:
        src_cfg = news_cfg.get(source_name, {})
        if not src_cfg.get("enabled", True):
            logger.info(f"manual.news.{source_name}.enabled is false — skipping.")
            continue

        cli_override = _max_posts_cli_override(args, source_name)
        max_posts = cli_override if cli_override is not None else src_cfg.get("max_posts", 500)

        if args.overwrite:
            n = delete_existing_shards(out_dir, source_name)
            logger.info(f"[{source_name}] --overwrite: deleted {n} existing shard(s).")
            state.pop(f"{source_name}_next_page", None)
            save_state(state_path, state)

        writer = JsonlShardWriter(out_dir, prefix=source_name, max_docs_per_shard=news_cfg.get("max_docs_per_shard", 20_000))
        already = load_collected_urls(out_dir, source_name)
        logger.info("=" * 70)
        logger.info(f"{source_name.upper()} — pulling posts via REST API (max_posts={max_posts})")
        logger.info("=" * 70)
        src = SOURCES[source_name]
        fetched, kept, skipped = collect_wordpress_source(
            source_name, src["api_url"], src["per_page"], src["license_basis"],
            max_posts, writer, already, state, state_path, logger,
        )
        writer.close()
        # writer.total_docs only counts documents WRITTEN during this
        # invocation (new_docs_this_run) — it is NOT a running total across
        # prior runs. `already` started as every previously-collected URL
        # and had each newly-kept URL added to it during collection, so
        # len(already) is the genuine all-time total for this source.
        new_docs_this_run = writer.total_docs
        total_docs_all_time = len(already)
        logger.info(
            f"[{source_name}] fetched={fetched} kept={kept} "
            f"skipped_already_collected={skipped} "
            f"new_docs_this_run={new_docs_this_run} "
            f"total_docs_all_time={total_docs_all_time}"
        )
        summary[source_name] = {
            "fetched": fetched, "kept": kept, "skipped_already_collected": skipped,
            "new_docs_this_run": new_docs_this_run,
            "total_docs_all_time": total_docs_all_time,
        }

    out_dir = _resolve_out_dir(config, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "news_collection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("NEWS COLLECTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Summary: {summary}")
    logger.info(f"Summary saved to {summary_path}")


def main() -> None:
    parser = build_arg_parser(
        "Collect Nepali manual-corpus news documents: Onlinekhabar + "
        "Khabarhub (both open WordPress REST API sources, each writing "
        "its own onlinekhabar_*.jsonl / khabarhub_*.jsonl shards). "
        "All large collection limits (--max_onlinekhabar, --max_khabarhub) "
        "are controllable from the command line, without needing to edit "
        "nepali_data.yaml; omitting them falls back to the YAML-configured "
        "defaults."
    )
    parser.add_argument("--max_onlinekhabar", type=int, default=None,
                         help="Maximum Onlinekhabar posts to collect in this run "
                              "(overrides manual.news.onlinekhabar.max_posts). "
                              "Must be a positive integer. Omit to use the YAML default.")
    parser.add_argument("--max_khabarhub", type=int, default=None,
                         help="Maximum Khabarhub posts to collect in this run "
                              "(overrides manual.news.khabarhub.max_posts). "
                              "Must be a positive integer. Omit to use the YAML default.")
    # build_arg_parser() already registers --dry-run (hyphen); --dry_run
    # (underscore) is accepted as an alias for the same dest so both
    # spellings work, matching this script's own --max_onlinekhabar-style
    # underscore convention.
    parser.add_argument("--dry_run", dest="dry_run", action="store_true",
                         help=argparse.SUPPRESS)
    args = parser.parse_args()

    for flag_name, value in (
        ("--max_onlinekhabar", args.max_onlinekhabar),
        ("--max_khabarhub", args.max_khabarhub),
    ):
        if value is not None and value <= 0:
            parser.error(f"{flag_name} must be a positive integer, got {value}")

    config = load_config(args.config)
    logger = setup_logger(f"collect_{SOURCE_TYPE}", config)
    repo_root = Path(args.config).resolve().parent.parent.parent

    if args.dry_run:
        run_dry_run(config, repo_root, args, logger)
        return

    if args.smoke_test:
        run_smoke_test(config, logger)
        return

    run_collect(config, repo_root, args, logger)


if __name__ == "__main__":
    main()
