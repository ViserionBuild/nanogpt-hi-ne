#!/usr/bin/env python3
"""
03_collect_news.py — Hindi MANUAL corpus collector: news / newspapers.

"news" is a SUBCATEGORY of the "manual" top-level corpus source (the other
top-level source is "indiccorp"; the other manual subcategory is
"wikipedia", collected by 01_download_wikipedia.py + 02_extract_wikipedia.py) —
never a separate top-level source. Raw output lives under
hindi/data/raw/manual/news/.

-----------------------------------------------------------------------------
SOURCE VERIFICATION (robots.txt + live sample), done 2026-08-18
-----------------------------------------------------------------------------
The project's candidate list named 10 Hindi news outlets (Dainik Jagran,
Amar Ujala, Dainik Bhaskar, Hindustan, Navbharat Times, Aaj Tak, ABP News,
NDTV Hindi, Jansatta, Prabhat Khabar). Each was checked before being wired
in here — "do not assume all sources will work" — via:
  1. `curl https://<host>/robots.txt` — reject any site whose robots.txt
     disallows AI/LLM crawlers by name (ClaudeBot, Claude-Web, anthropic-ai,
     CCBot, GPTBot, ...), since collecting from it would run against the
     publisher's stated crawling policy even though this project uses
     ordinary HTTP requests, not one of those literal user agents.
  2. A working, reachable article/news XML sitemap.
  3. A live sample article fetched with a plain `requests.get()` (no
     headless browser / JS execution) actually contains extractable
     Devanagari body text via the generic <article>/<p> extractor below.

ENABLED (hindi_data.yaml manual.news.<name>.enabled: true):
    dainik_jagran, hindustan, abp — see hindi_data.yaml for the exact
    robots.txt / sitemap / sample-fetch evidence per source.

DISABLED (kept fully configured, `enabled: false`, with a `disabled_reason`
in hindi_data.yaml): amar_ujala, dainik_bhaskar, jansatta (robots.txt
name-disallows AI crawlers), navbharat_times, prabhat_khabar (JS-rendered
body, nothing recoverable via static HTML), aaj_tak (no working sitemap),
ndtv (edge-blocked). Re-verify and flip `enabled: true` only if a
disabled source's situation changes AND its robots.txt does not
name-disallow AI crawlers.

-----------------------------------------------------------------------------
DISCOVERY ARCHITECTURE — discovery is SEPARATE from article fetching
-----------------------------------------------------------------------------
Earlier versions of this collector conflated "walk the sitemap" with
"fetch the articles it names" into one pass bounded by a single
max_articles number — which meant a source could never accumulate more
than whatever its sitemap happened to expose *right now* (a rolling
window of ~1000-4000 recent URLs). This version splits the two concerns:

  DISCOVERY  (this section) grows a PERSISTENT per-source URL inventory
             (`<output_dir>/_discovery/<source>_urls.jsonl` +
             `..._state.json`) that survives across runs. Each run adds
             to it; nothing is ever re-discovered from scratch.

  FETCH      (see "CONCURRENT ARTICLE FETCHING" below) pulls up to
             `max_articles` URLs that are in the discovery inventory but
             not yet collected (or permanently rejected) and downloads
             their article bodies. Unchanged from before in mechanics
             (thread pool, single writer, retry/backoff).

This is what lets the collector progress BACKWARD through history
instead of repeatedly rediscovering the same recent few thousand URLs:
each run's discovery phase either advances a resumable historical
frontier (dated-pattern / static-index sources) or accumulates whatever
newly-published URLs a rolling sitemap exposes since the last run
(rolling-only sources) — either way, URLs already in the inventory are
never re-discovered or re-fetched.

Three discovery mechanisms are supported, selected per source by
`manual.news.<name>.discovery.type` in hindi_data.yaml (see that file
for exactly which type each enabled source uses and why):

  sitemap_rolling
      The source exposes only a rolling recent-window sitemap (or
      sitemapindex) with no dated/day/month archive anywhere in
      robots.txt or the sitemap tree. Every configured `sitemap_urls`
      entry is walked (bounded by max_sitemap_documents) each run;
      already-known URLs are naturally deduplicated by the persistent
      store, so only genuinely new URLs are added. This is honest about
      its limits: it does NOT invent a historical mechanism the
      publisher doesn't expose (requirement: never invent sitemap URLs).
      dainik_jagran uses this.

  sitemap_index_dated
      The source's OWN sitemap index is itself a per-day dated document
      (e.g. news-DD-MM-YYYY.xml) — the index's <sitemap><loc> children
      already reveal the exact verified filename template. This
      collector (a) walks whatever the index currently lists, then
      (b) CONTINUES that same verified template backward, one day at a
      time, from the oldest date the index lists down to
      `historical_discovery.start_date`, deriving each next URL from a
      confirmed real structure rather than guessing a new one. A run's
      backward walk is bounded by max_sitemap_documents /
      max_discovered_urls and stops early if
      `stop_when_no_older_sitemaps` triggers (N consecutive
      empty/near-empty daily sitemaps — see hindi_data.yaml). The
      frontier (oldest date reached so far) is persisted, so the next
      run resumes the backward walk instead of restarting it. abp uses
      this.

  sitemap_index_static
      The source publishes (and lists in robots.txt) an ALREADY
      POPULATED historical sitemapindex — real dated child sitemaps
      going back years, not merely the current window. This collector
      walks the index's actual <sitemap><loc> entries as-is (never
      deriving/guessing additional ones), filtered to
      [start_date, end_date], and separately walks the source's rolling
      sitemap each run for the very latest articles (which may not yet
      appear in a "closed" historical bucket). Historical buckets, once
      fully captured, are marked CLOSED in persisted state and never
      re-fetched — only buckets newer than `closed_after_days` are
      re-fetched each run, since only those can still be gaining URLs.
      hindustan uses this.

Discovery never uses a headless browser (plain `requests.get()` /
XML parsing only) and never fabricates a sitemap/index URL that wasn't
actually observed being exposed by the publisher (directly, or as a
child of a real sitemapindex, or by extending a filename template the
publisher's own index already demonstrated) — see hindi_data.yaml's
per-source `historical.*` blocks for the exact verification evidence
behind each ENABLED source's discovery.type.

Per-source discovery + fetch statistics (sitemap documents
discovered/fetched, URLs discovered/duplicate/already-collected/newly
available, articles fetched/kept/rejected, fetch failures) are recorded
in hindi/data/raw/manual/news/news_collection_summary.json after every
real run — alongside the raw JSONL shards it summarizes, not in a
separate top-level metadata directory.

-----------------------------------------------------------------------------
CONCURRENT ARTICLE FETCHING (unchanged from prior versions)
-----------------------------------------------------------------------------
Article fetch+extract is the bottleneck (one HTTP request + HTML parse
per article), so it runs on a bounded `concurrent.futures.ThreadPoolExecutor`
(I/O-bound workload -> threads, not processes). Discovery itself (sitemap
document fetching) stays SEQUENTIAL regardless of concurrency — raising
concurrency was explicitly rejected as "the" fix for the rolling-window
problem; the fix is the discovery architecture above, not more threads.

  * Concurrency is configurable via `manual.news.collection.concurrency`
    in hindi_data.yaml (default 8), overridable per source
    (`manual.news.<name>.concurrency`) or per run (`--concurrency`).
  * Each worker thread fetches+extracts+validates exactly ONE article and
    RETURNS the record (or None) — workers never write to the shard file
    or mutate `already_collected`/the discovery store directly. Only the
    MAIN thread, as futures complete (`as_completed`), writes to
    `JsonlShardWriter` and updates shared state — there is only ever one
    writer touching the shard file, so no file lock is needed.
  * Workers share one `requests.Session` per source-collection run for
    HTTP connection pooling.
  * Each worker thread paces itself with SLEEP_BETWEEN_REQ between its
    own requests; aggregate request rate is bounded at roughly
    `concurrency / SLEEP_BETWEEN_REQ` requests/second.
  * Retries/backoff/timeouts (see polite_get()) are unchanged.

-----------------------------------------------------------------------------
OUTPUT FORMAT
-----------------------------------------------------------------------------
Sharded JSONL (~max_docs_per_shard documents per shard file), one shard
sequence per source_name:

    hindi/data/raw/manual/news/dainik_jagran_00000.jsonl, _00001.jsonl, ...
    hindi/data/raw/manual/news/hindustan_00000.jsonl, ...
    hindi/data/raw/manual/news/abp_00000.jsonl, ...

Each line is one document record using the SAME common schema fields as
every other manual/collect_*.py collector (manual_common.DocumentMetadata)
PLUS the article text itself and a deterministic doc_id:

    {
      "doc_id": "...", "text": "...",
      "data_source": "manual", "source_type": "news",
      "source_name": "dainik_jagran" | "hindustan" | "abp" | ...,
      "title": "...", "source_url": "...", "language": "hi",
      "document_type": "html_article",
      "collection_date": "YYYY-MM-DD",
      "original_filename": "", "local_filename": "",
      "license_or_access_basis": "...",
      "extraction_method": "...", "extraction_status": "success"
    }

license_or_access_basis records ONLY what is actually verified (the
robots.txt check above) — never an invented "open license"/"public
domain" claim.

Discovery persistence lives alongside the shards, under a `_discovery/`
subdirectory of the source's output_dir:

    hindi/data/raw/manual/news/_discovery/dainik_jagran_urls.jsonl
    hindi/data/raw/manual/news/_discovery/dainik_jagran_state.json
    hindi/data/raw/manual/news/_discovery/dainik_jagran_rejected.jsonl
    ... (one triple per source)

-----------------------------------------------------------------------------
RESUMABILITY
-----------------------------------------------------------------------------
At startup, every existing shard for a source_name is scanned once to
build an in-memory `source_url -> already collected` set so already-
collected articles are skipped without re-fetching them (source_url
dedup, unchanged). The discovery store adds a SECOND resumable layer on
top: URLs already in `<source>_urls.jsonl` are never re-added (in-memory
dedup set loaded once at startup), sitemap documents already marked
CLOSED in `<source>_state.json` are never re-fetched, and a dated-pattern
source's backward-walk frontier is persisted so the next run continues
the walk instead of restarting from today. URLs that failed extraction
for being too short are recorded in `<source>_rejected.jsonl` so they are
never re-fetched either (a transient fetch failure is NOT recorded here
and so is naturally retried on the next run).

--overwrite deletes and re-collects only the source_name(s) actually being
run in that invocation (shards AND discovery state) — never any other
source, never wikipedia.

Usage:
    python hindi/scripts/manual/03_collect_news.py \\
        --config hindi/configs/hindi_data.yaml

    python hindi/scripts/manual/03_collect_news.py \\
        --config hindi/configs/hindi_data.yaml --source dainik_jagran

    # Discovery-only: walk sitemaps/historical index and grow the
    # persisted URL inventory, but download ZERO article bodies. Reports
    # exactly how many historical URLs are actually discoverable before
    # any article-body network traffic happens (requirement: dry-run
    # reports discoverable volume up front). This DOES perform network
    # I/O against sitemap documents (a deliberate, documented deviation
    # from manual_common's usual "--dry-run = zero network calls"
    # convention, since discovery-volume reporting is meaningless without
    # actually walking the sitemaps).
    python hindi/scripts/manual/03_collect_news.py \\
        --config hindi/configs/hindi_data.yaml --dry-run

    python hindi/scripts/manual/03_collect_news.py \\
        --config hindi/configs/hindi_data.yaml --smoke-test

    # Override concurrency for this run only (default: 8, or
    # manual.news.collection.concurrency from hindi_data.yaml):
    python hindi/scripts/manual/03_collect_news.py \\
        --config hindi/configs/hindi_data.yaml --concurrency 4

    # Override historical discovery window / budgets for this run only:
    python hindi/scripts/manual/03_collect_news.py \\
        --config hindi/configs/hindi_data.yaml --source abp \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --max-discovered-urls 50000 --max-sitemap-documents 400
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

sys.path.insert(0, os.path.dirname(__file__))
from manual_common import (  # noqa: E402
    NEWS_SOURCE_NAMES, build_arg_parser, collection_date_now, load_config,
    probe_url, setup_logger,
)

SOURCE_TYPE = "news"

USER_AGENT = (
    "LMA-IDV-HindiCorpus/1.0 (educational/research corpus collection; "
    "see hindi/scripts/manual/03_collect_news.py)"
)
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQ = 0.25
MIN_ARTICLE_CHARS = 300
MAX_FETCH_RETRIES = 3
DEFAULT_MAX_SITEMAP_DOCUMENTS = 60
DEFAULT_MAX_DISCOVERED_URLS = 20_000
DEFAULT_CONSECUTIVE_EMPTY_DAYS_TO_STOP = 14
DEFAULT_CLOSED_AFTER_DAYS = 3
# A dated daily sitemap with this few (or fewer) <url> entries is treated
# as "empty" for stop_when_no_older_sitemaps purposes — some publishers
# return HTTP 200 with a near-empty <urlset> instead of a 404 for dates
# outside their real archive (see abp's historical.note in hindi_data.yaml).
EMPTY_DAY_URL_THRESHOLD = 2

# Bounded worker-pool size for concurrent article fetching (see module
# docstring "CONCURRENT ARTICLE FETCHING"). Configurable via
# manual.news.collection.concurrency / manual.news.<source>.concurrency /
# --concurrency — this is only the fallback when none of those are set.
DEFAULT_CONCURRENCY = 8

_JUNK_CLASS_HINTS = (
    "nav", "menu", "footer", "header", "sidebar", "related", "comment",
    "advert", "ad-", "share", "social", "widget", "breadcrumb", "tag",
    "subscribe", "newsletter",
)

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Per-source license_or_access_basis strings — each states ONLY the
# verified robots.txt evidence, never a guessed content license. See the
# module docstring's "SOURCE VERIFICATION" section for the full evidence.
_LICENSE_BASIS_TEMPLATE = (
    "{host} robots.txt (User-agent: *) is permissive for this source's "
    "article paths and does not name-disallow this project's collection "
    "approach (verified live 2026-08-18). Articles remain copyrighted by "
    "the publisher; this records ONLY that automated collection is not "
    "disallowed, not a content license grant."
)


# ---------------------------------------------------------------------------
# Sharded JSONL writer (unchanged)
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
# HTTP helpers (retry + rate limiting) — unchanged
# ---------------------------------------------------------------------------

def polite_get(
    url: str, logger, session: Optional[requests.Session] = None,
) -> Optional[requests.Response]:
    """Unchanged retry/backoff/timeout semantics. *session*, when given,
    reuses pooled HTTP connections (see _build_session()) instead of
    opening a fresh one per call — the User-Agent is then already set on
    the session, so it isn't passed again per-request."""
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            if session is not None:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
            else:
                resp = requests.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT,
                )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None  # genuinely missing, not a transient error
            logger.warning(f"GET {url} -> HTTP {resp.status_code} (attempt {attempt}/{MAX_FETCH_RETRIES})")
        except requests.RequestException as exc:
            logger.warning(f"GET {url} failed: {exc} (attempt {attempt}/{MAX_FETCH_RETRIES})")
        if attempt < MAX_FETCH_RETRIES:
            time.sleep(2 ** attempt)
    return None


def _build_session(pool_size: int) -> requests.Session:
    """One persistent Session per source-collection run, shared read-only
    (only .get() calls) across worker threads for HTTP connection
    pooling. requests.Session is safe to share across threads under this
    usage pattern (each thread only calls .get(), never mutates the
    session) — see requests' own thread-safety notes."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Date helpers (dated-pattern / static-index discovery)
# ---------------------------------------------------------------------------

def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _parse_lastmod_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


_LH_BUCKET_RE = re.compile(r"/sitemap/(\d{4})/([a-z]{3})-(\d+)\.xml$", re.IGNORECASE)
_ABP_DATED_RE = re.compile(r"news-(\d{2})-(\d{2})-(\d{4})\.xml$")


def _parse_abp_dated_url(url: str) -> Optional[date]:
    m = _ABP_DATED_RE.search(url)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    try:
        return date(int(yyyy), int(mm), int(dd))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Persistent discovery store
# ---------------------------------------------------------------------------

class DiscoveryStore:
    """Persists ONE source's discovered-URL inventory + discovery
    frontier state across runs, under <out_dir>/_discovery/. This is what
    makes discovery resumable: URLs already known are never re-added
    (`urls` set, loaded once at startup), sitemap documents already fully
    mined are marked CLOSED in `state` and never re-fetched, and a
    dated-pattern backward walk's frontier date is persisted so the next
    run continues it rather than restarting from today.
    """

    def __init__(self, discovery_dir: Path, source_name: str):
        self.dir = Path(discovery_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.source_name = source_name
        self.urls_path = self.dir / f"{source_name}_urls.jsonl"
        self.state_path = self.dir / f"{source_name}_state.json"
        self.rejected_path = self.dir / f"{source_name}_rejected.jsonl"

        self.urls: List[str] = []
        self._url_set: Set[str] = set()
        self.rejected: Set[str] = set()
        self.state: Dict[str, Any] = {
            "processed_sitemap_docs": {},  # doc_url -> {"closed": bool, "processed_at": iso}
            "dated_frontier_oldest": None,  # "YYYY-MM-DD" — oldest date already probed
        }
        self._load()

    def _load(self) -> None:
        if self.urls_path.exists():
            with open(self.urls_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = rec.get("url")
                    if url and url not in self._url_set:
                        self._url_set.add(url)
                        self.urls.append(url)
        if self.rejected_path.exists():
            with open(self.rejected_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = rec.get("url")
                    if url:
                        self.rejected.add(url)
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.state.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def has_url(self, url: str) -> bool:
        return url in self._url_set

    def add_url(self, url: str, sitemap_doc: str) -> bool:
        """Returns True if *url* was newly added, False if it was
        already in the inventory (duplicate)."""
        if url in self._url_set:
            return False
        self._url_set.add(url)
        self.urls.append(url)
        with open(self.urls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "url": url, "sitemap_doc": sitemap_doc,
                "discovered_at": datetime.utcnow().isoformat() + "Z",
            }, ensure_ascii=False) + "\n")
        return True

    def mark_rejected(self, url: str, reason: str) -> None:
        if url in self.rejected:
            return
        self.rejected.add(url)
        with open(self.rejected_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "url": url, "reason": reason,
                "attempted_at": datetime.utcnow().isoformat() + "Z",
            }, ensure_ascii=False) + "\n")

    def is_doc_closed(self, doc_url: str) -> bool:
        entry = self.state["processed_sitemap_docs"].get(doc_url)
        return bool(entry and entry.get("closed"))

    def mark_doc_processed(self, doc_url: str, closed: bool) -> None:
        self.state["processed_sitemap_docs"][doc_url] = {
            "closed": closed, "processed_at": datetime.utcnow().isoformat() + "Z",
        }

    def get_dated_frontier_oldest(self) -> Optional[date]:
        return _parse_iso_date(self.state.get("dated_frontier_oldest"))

    def set_dated_frontier_oldest(self, d: date) -> None:
        self.state["dated_frontier_oldest"] = d.isoformat()
        self.save_state()

    def save_state(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def delete(self) -> int:
        n = 0
        for p in (self.urls_path, self.state_path, self.rejected_path):
            if p.exists():
                p.unlink()
                n += 1
        return n


@dataclass
class DiscoveryStats:
    source_name: str
    sitemap_documents_discovered: int = 0
    sitemap_documents_fetched: int = 0
    urls_discovered: int = 0
    duplicate_urls: int = 0
    urls_already_collected: int = 0
    new_urls_available: int = 0
    articles_fetched: int = 0
    articles_kept: int = 0
    articles_rejected: int = 0
    fetch_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sitemap document parsing (generic, namespace-agnostic)
# ---------------------------------------------------------------------------

def _local_tag(elem_tag: str) -> str:
    """Strip an XML namespace URI from an ElementTree tag."""
    return elem_tag.rsplit("}", 1)[-1] if "}" in elem_tag else elem_tag


def _parse_sitemap_document(xml_bytes: bytes) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """Parse one sitemap XML document. Returns (kind, entries) where kind
    is "urlset" (entries are (article_url, lastmod)) or "sitemapindex"
    (entries are (child_sitemap_url, lastmod)), or ("unknown", []) if the
    document isn't parseable/recognized — never raises on malformed XML."""
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return "unknown", []

    kind = _local_tag(root.tag)
    if kind not in ("urlset", "sitemapindex"):
        return "unknown", []

    entries: List[Tuple[str, Optional[str]]] = []
    child_tag = "url" if kind == "urlset" else "sitemap"
    for child in root:
        if _local_tag(child.tag) != child_tag:
            continue
        loc, lastmod = None, None
        for grandchild in child:
            gtag = _local_tag(grandchild.tag)
            if gtag == "loc" and grandchild.text:
                loc = grandchild.text.strip()
            elif gtag == "lastmod" and grandchild.text:
                lastmod = grandchild.text.strip()
        if loc:
            entries.append((loc, lastmod))
    return kind, entries


def _fetch_sitemap_doc(url: str, logger) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """Fetch + parse one sitemap document. Always sequential/single-
    request (discovery deliberately does not use the article-fetch
    thread pool — see module docstring)."""
    resp = polite_get(url, logger)
    time.sleep(SLEEP_BETWEEN_REQ)
    if resp is None:
        logger.warning(f"sitemap document unreachable, skipping: {url}")
        return "unknown", []
    return _parse_sitemap_document(resp.content)


# ---------------------------------------------------------------------------
# Discovery: sitemap_rolling
# ---------------------------------------------------------------------------

def discover_rolling(
    sitemap_urls: List[str],
    store: DiscoveryStore,
    max_sitemap_documents: int,
    max_discovered_urls: int,
    logger,
    source_name: str,
) -> DiscoveryStats:
    """Walk a fixed list of (possibly index) sitemap URLs, unbounded in
    HISTORY (there is none to bound — this is the rolling-window-only
    mechanism) but bounded in per-run sitemap-document/URL budget.
    Already-known URLs are simply not re-added (DiscoveryStore dedup)."""
    stats = DiscoveryStats(source_name=source_name)
    queue: List[str] = list(sitemap_urls)
    seen_docs: Set[str] = set()

    while queue and stats.sitemap_documents_fetched < max_sitemap_documents \
            and stats.new_urls_available < max_discovered_urls:
        doc_url = queue.pop(0)
        if doc_url in seen_docs:
            continue
        seen_docs.add(doc_url)
        stats.sitemap_documents_discovered += 1
        kind, entries = _fetch_sitemap_doc(doc_url, logger)
        stats.sitemap_documents_fetched += 1
        if kind == "sitemapindex":
            queue.extend(loc for loc, _ in entries)
            stats.sitemap_documents_discovered += len(entries)
        elif kind == "urlset":
            for loc, _ in entries:
                stats.urls_discovered += 1
                if store.add_url(loc, doc_url):
                    stats.new_urls_available += 1
                else:
                    stats.duplicate_urls += 1
                if stats.new_urls_available >= max_discovered_urls:
                    break
        else:
            logger.warning(f"[{source_name}] {doc_url} not a recognizable sitemap document; skipping.")

    return stats


# ---------------------------------------------------------------------------
# Discovery: sitemap_index_dated (e.g. abp — news-DD-MM-YYYY.xml)
# ---------------------------------------------------------------------------

def discover_dated(
    index_url: str,
    dated_pattern: str,
    store: DiscoveryStore,
    start_date: date,
    end_date: date,
    max_sitemap_documents: int,
    max_discovered_urls: int,
    stop_when_no_older_sitemaps: bool,
    consecutive_empty_days_to_stop: int,
    closed_after_days: int,
    logger,
    source_name: str,
) -> DiscoveryStats:
    stats = DiscoveryStats(source_name=source_name)
    today = date.today()

    def _ingest_doc(doc_url: str, doc_date: Optional[date]) -> int:
        """Fetch+parse one dated sitemap document, add its URLs to the
        store, mark it closed if its date is old enough to never change
        again. Returns the number of <url> entries found (0 = empty)."""
        stats.sitemap_documents_discovered += 1
        kind, entries = _fetch_sitemap_doc(doc_url, logger)
        stats.sitemap_documents_fetched += 1
        n = 0
        if kind == "urlset":
            n = len(entries)
            for loc, _ in entries:
                stats.urls_discovered += 1
                if store.add_url(loc, doc_url):
                    stats.new_urls_available += 1
                else:
                    stats.duplicate_urls += 1
        if doc_date is not None and (today - doc_date).days >= closed_after_days:
            store.mark_doc_processed(doc_url, closed=True)
        return n

    # Step 1: walk whatever the index itself currently lists (real,
    # exposed URLs — no derivation needed for these).
    kind, entries = _fetch_sitemap_doc(index_url, logger)
    stats.sitemap_documents_discovered += 1
    stats.sitemap_documents_fetched += 1
    index_dates: List[date] = []
    if kind == "sitemapindex":
        for loc, lastmod in entries:
            d = _parse_abp_dated_url(loc) or _parse_lastmod_date(lastmod)
            if d is not None:
                index_dates.append(d)
            if store.is_doc_closed(loc):
                continue
            if d is not None and not (start_date <= d <= end_date):
                continue
            if stats.new_urls_available >= max_discovered_urls or \
                    stats.sitemap_documents_fetched >= max_sitemap_documents:
                break
            _ingest_doc(loc, d)
    elif kind == "urlset":
        # Some sources might expose the index itself as a flat urlset;
        # treat every entry as an article URL directly.
        for loc, _ in entries:
            stats.urls_discovered += 1
            if store.add_url(loc, index_url):
                stats.new_urls_available += 1
            else:
                stats.duplicate_urls += 1

    # Step 2: continue the SAME verified filename template backward from
    # the resumable frontier (or the oldest date the index listed, on a
    # first run) down to start_date, subject to per-run budgets and
    # stop_when_no_older_sitemaps.
    frontier = store.get_dated_frontier_oldest()
    if frontier is None:
        frontier = min(index_dates) if index_dates else today
    cursor = frontier - timedelta(days=1)
    consecutive_empty = 0

    while cursor >= start_date \
            and stats.sitemap_documents_fetched < max_sitemap_documents \
            and stats.new_urls_available < max_discovered_urls:
        doc_url = dated_pattern.format(dd=f"{cursor.day:02d}", mm=f"{cursor.month:02d}", yyyy=cursor.year)
        if store.is_doc_closed(doc_url):
            cursor -= timedelta(days=1)
            continue
        n = _ingest_doc(doc_url, cursor)
        store.set_dated_frontier_oldest(cursor)
        if n <= EMPTY_DAY_URL_THRESHOLD:
            consecutive_empty += 1
        else:
            consecutive_empty = 0
        cursor -= timedelta(days=1)
        if stop_when_no_older_sitemaps and consecutive_empty >= consecutive_empty_days_to_stop:
            logger.info(
                f"[{source_name}] stopping backward walk: {consecutive_empty} consecutive "
                f"near-empty daily sitemaps (last probed date: {(cursor + timedelta(days=1)).isoformat()})."
            )
            break

    return stats


# ---------------------------------------------------------------------------
# Discovery: sitemap_index_static (e.g. hindustan — sitemap-index.xml)
# ---------------------------------------------------------------------------

def discover_static_index(
    index_url: str,
    rolling_sitemap_urls: List[str],
    store: DiscoveryStore,
    start_date: date,
    end_date: date,
    max_sitemap_documents: int,
    max_discovered_urls: int,
    closed_after_days: int,
    logger,
    source_name: str,
) -> DiscoveryStats:
    stats = DiscoveryStats(source_name=source_name)
    today = date.today()

    # Rolling near-real-time sitemap first (cheap: 1 document, catches
    # the very latest articles that may not be in a "closed" bucket yet).
    for doc_url in rolling_sitemap_urls:
        if stats.sitemap_documents_fetched >= max_sitemap_documents:
            break
        stats.sitemap_documents_discovered += 1
        kind, entries = _fetch_sitemap_doc(doc_url, logger)
        stats.sitemap_documents_fetched += 1
        if kind == "urlset":
            for loc, _ in entries:
                stats.urls_discovered += 1
                if store.add_url(loc, doc_url):
                    stats.new_urls_available += 1
                else:
                    stats.duplicate_urls += 1

    # The real, already-populated historical index — every child is a
    # URL the publisher itself exposes; none are derived/guessed.
    kind, entries = _fetch_sitemap_doc(index_url, logger)
    stats.sitemap_documents_discovered += 1
    stats.sitemap_documents_fetched += 1
    if kind != "sitemapindex":
        logger.warning(f"[{source_name}] {index_url} was not a sitemapindex; historical backfill skipped this run.")
        return stats

    # Newest-first is how these indexes are typically published; process
    # in that order so a budget-limited run makes progress on the most
    # recently-closed (and thus most valuable) buckets first.
    for loc, lastmod in entries:
        if stats.sitemap_documents_fetched >= max_sitemap_documents or \
                stats.new_urls_available >= max_discovered_urls:
            break
        if store.is_doc_closed(loc):
            continue
        bucket_date = _parse_lastmod_date(lastmod)
        m = _LH_BUCKET_RE.search(loc)
        if bucket_date is None and m:
            year = int(m.group(1))
            month = _MONTH_ABBR.get(m.group(2).lower())
            if month:
                bucket_date = date(year, month, 1)
        if bucket_date is not None and not (start_date <= bucket_date <= end_date):
            # Buckets are listed newest-first; once we're past end_date
            # going forward or before start_date going backward we can
            # stop scanning entirely rather than checking every entry.
            if bucket_date < start_date:
                break
            continue

        stats.sitemap_documents_discovered += 1
        child_kind, child_entries = _fetch_sitemap_doc(loc, logger)
        stats.sitemap_documents_fetched += 1
        if child_kind == "urlset":
            for url, url_lastmod in child_entries:
                stats.urls_discovered += 1
                if store.add_url(url, loc):
                    stats.new_urls_available += 1
                else:
                    stats.duplicate_urls += 1
        close_reference = bucket_date or today
        if (today - close_reference).days >= closed_after_days:
            store.mark_doc_processed(loc, closed=True)

    return stats


# ---------------------------------------------------------------------------
# Discovery dispatch
# ---------------------------------------------------------------------------

def discover_one_source(
    source_name: str,
    src_cfg: Dict[str, Any],
    store: DiscoveryStore,
    hist_cfg: Dict[str, Any],
    max_sitemap_documents: int,
    logger,
) -> DiscoveryStats:
    discovery_cfg = src_cfg.get("discovery") or {}
    dtype = discovery_cfg.get("type", "sitemap_rolling")

    start_date = _parse_iso_date(hist_cfg.get("start_date")) or date(2000, 1, 1)
    end_date = _parse_iso_date(hist_cfg.get("end_date")) or date.today()
    max_discovered_urls = hist_cfg.get("max_discovered_urls", DEFAULT_MAX_DISCOVERED_URLS)
    stop_when_no_older_sitemaps = hist_cfg.get("stop_when_no_older_sitemaps", True)
    consecutive_empty_days_to_stop = hist_cfg.get(
        "consecutive_empty_days_to_stop", DEFAULT_CONSECUTIVE_EMPTY_DAYS_TO_STOP)
    closed_after_days = hist_cfg.get("closed_after_days", DEFAULT_CLOSED_AFTER_DAYS)
    historical_enabled = hist_cfg.get("enabled", True)

    if dtype == "sitemap_index_dated" and historical_enabled:
        return discover_dated(
            index_url=discovery_cfg.get("index_url", (src_cfg.get("sitemap_urls") or [None])[0]),
            dated_pattern=discovery_cfg["dated_pattern"],
            store=store, start_date=start_date, end_date=end_date,
            max_sitemap_documents=max_sitemap_documents,
            max_discovered_urls=max_discovered_urls,
            stop_when_no_older_sitemaps=stop_when_no_older_sitemaps,
            consecutive_empty_days_to_stop=consecutive_empty_days_to_stop,
            closed_after_days=closed_after_days,
            logger=logger, source_name=source_name,
        )
    if dtype == "sitemap_index_static" and historical_enabled:
        return discover_static_index(
            index_url=discovery_cfg["index_url"],
            rolling_sitemap_urls=discovery_cfg.get("rolling_sitemap_urls")
                or src_cfg.get("sitemap_urls") or [],
            store=store, start_date=start_date, end_date=end_date,
            max_sitemap_documents=max_sitemap_documents,
            max_discovered_urls=max_discovered_urls,
            closed_after_days=closed_after_days,
            logger=logger, source_name=source_name,
        )

    # sitemap_rolling, OR a historical-capable type with
    # historical_discovery.enabled: false for this run — fall back to
    # walking only the configured (rolling) sitemap_urls.
    sitemap_urls = src_cfg.get("sitemap_urls") or []
    if not sitemap_urls:
        logger.warning(f"[{source_name}] no sitemap_urls configured — nothing to discover.")
        return DiscoveryStats(source_name=source_name)
    return discover_rolling(
        sitemap_urls=sitemap_urls, store=store,
        max_sitemap_documents=max_sitemap_documents,
        max_discovered_urls=max_discovered_urls,
        logger=logger, source_name=source_name,
    )


# ---------------------------------------------------------------------------
# Extraction (generic <article>/<p> heuristic — unchanged)
# ---------------------------------------------------------------------------

def _looks_like_junk_container(tag) -> bool:
    ident = " ".join(tag.get("class", []) + [tag.get("id", "")]).lower()
    return any(h in ident for h in _JUNK_CLASS_HINTS)


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for tag in list(soup.find_all(["div", "section", "ul", "aside", "form"])):
        if tag.parent is not None and _looks_like_junk_container(tag):
            tag.decompose()

    article = soup.find("article")
    if article is not None:
        text = " ".join(p.get_text(separator=" ") for p in article.find_all("p"))
        if len(text.strip()) >= MIN_ARTICLE_CHARS:
            return text
    return " ".join(p.get_text(separator=" ") for p in soup.find_all("p"))


def clean_article_text(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and len(ln) > 3]
    return "\n".join(lines).strip()


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(separator=" ").strip()
    return ""


# ---------------------------------------------------------------------------
# Per-source collection: discovery, then fetch
# ---------------------------------------------------------------------------

def _fetch_and_extract_one(
    source_name: str,
    url: str,
    license_basis: str,
    session: requests.Session,
    logger,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Runs in a worker thread: fetch + extract + validate exactly ONE
    article. Never writes to the shard file and never mutates shared
    state — returns (url, record_or_None, status) for the MAIN thread to
    act on. status is "success" / "fetch_failed" / "too_short"."""
    resp = polite_get(url, logger, session=session)
    time.sleep(SLEEP_BETWEEN_REQ)  # per-thread pacing, unchanged politeness value
    if resp is None:
        return url, None, "fetch_failed"
    raw = extract_article_text(resp.text)
    cleaned = clean_article_text(raw)
    if len(cleaned) < MIN_ARTICLE_CHARS:
        return url, None, "too_short"
    title = extract_title(resp.text) or url
    record = {
        "doc_id": make_doc_id(source_name, url),
        "text": cleaned,
        "data_source": "manual",
        "source_type": SOURCE_TYPE,
        "source_name": source_name,
        "title": title,
        "source_url": url,
        "language": "hi",
        "document_type": "html_article",
        "collection_date": collection_date_now(),
        "original_filename": "",
        "local_filename": "",
        "license_or_access_basis": license_basis,
        "extraction_method": "bs4_generic_article_or_paragraph_extraction",
        "extraction_status": "success",
    }
    return url, record, "success"


def collect_one_source(
    source_name: str,
    src_cfg: Dict[str, Any],
    hist_cfg: Dict[str, Any],
    writer: JsonlShardWriter,
    already_collected: Set[str],
    store: DiscoveryStore,
    max_sitemap_documents: int,
    logger,
    concurrency: int = DEFAULT_CONCURRENCY,
    discovery_only: bool = False,
) -> DiscoveryStats:
    """Runs discovery (growing *store*'s persisted inventory), then — unless
    *discovery_only* — fetches up to src_cfg['max_articles'] URLs that are
    in the inventory but not yet collected/permanently-rejected. Returns
    combined discovery+fetch statistics for this invocation."""
    base_url = src_cfg.get("base_url", "")
    host = base_url.split("//", 1)[-1].rstrip("/")
    max_articles = src_cfg.get("max_articles", 2000)
    license_basis = _LICENSE_BASIS_TEMPLATE.format(host=host)

    stats = discover_one_source(source_name, src_cfg, store, hist_cfg, max_sitemap_documents, logger)
    stats.urls_already_collected = sum(1 for u in store.urls if u in already_collected)
    pending = [u for u in store.urls if u not in already_collected and u not in store.rejected]
    stats.new_urls_available = len(pending)

    logger.info(
        f"[{source_name}] discovery: {stats.sitemap_documents_fetched} sitemap doc(s) fetched, "
        f"{stats.urls_discovered} URL(s) seen ({stats.duplicate_urls} duplicate), "
        f"inventory={len(store.urls)} total / {stats.urls_already_collected} already collected / "
        f"{len(pending)} pending fetch"
    )

    if discovery_only:
        return stats

    to_fetch = pending[:max_articles]
    session = _build_session(concurrency)
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_fetch_and_extract_one, source_name, url, license_basis, session, logger)
                for url in to_fetch
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    url, record, status = future.result()
                except Exception as exc:  # a single article must never abort the batch
                    logger.warning(f"[{source_name}] worker raised {exc!r}; counted as failed.")
                    stats.articles_fetched += 1
                    stats.fetch_failures += 1
                    continue

                stats.articles_fetched += 1
                if status == "success" and record is not None:
                    writer.write(record)
                    already_collected.add(url)
                    stats.articles_kept += 1
                elif status == "too_short":
                    store.mark_rejected(url, "too_short")
                    stats.articles_rejected += 1
                else:
                    stats.fetch_failures += 1

                if stats.articles_fetched % 50 == 0:
                    elapsed = time.time() - t0
                    rate = stats.articles_fetched / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"  [{source_name}] {stats.articles_fetched} fetched, {stats.articles_kept} kept | "
                        f"{rate:.1f} articles/s | concurrency={concurrency}"
                    )
    finally:
        session.close()

    elapsed = time.time() - t0
    rate = stats.articles_fetched / elapsed if elapsed > 0 else 0.0
    logger.info(f"{source_name.upper()} collection complete")
    logger.info(f"  Sitemap docs fetched: {stats.sitemap_documents_fetched}")
    logger.info(f"  URLs discovered:      {stats.urls_discovered} ({stats.duplicate_urls} duplicate)")
    logger.info(f"  Inventory total:      {len(store.urls)}")
    logger.info(f"  Fetched:              {stats.articles_fetched}")
    logger.info(f"  Kept:                 {stats.articles_kept}")
    logger.info(f"  Rejected (too short): {stats.articles_rejected}")
    logger.info(f"  Failed:               {stats.fetch_failures}")
    logger.info(f"  Elapsed:              {elapsed:.1f}s")
    logger.info(f"  Average:              {rate:.2f} articles/s")
    logger.info(f"  Concurrency:          {concurrency}")

    return stats


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _news_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("manual", {}).get("news", {})


def _resolve_out_dir(config: Dict[str, Any], repo_root: Path) -> Path:
    news_cfg = _news_cfg(config)
    return repo_root / news_cfg.get("output_dir", "hindi/data/raw/manual/news")


def _configured_sources(news_cfg: Dict[str, Any]) -> List[str]:
    """Sources actually present in config, in NEWS_SOURCE_NAMES order (a
    closed, checkable set) — not every name in NEWS_SOURCE_NAMES need be
    present in a given config."""
    return [name for name in NEWS_SOURCE_NAMES if name in news_cfg]


def _resolve_concurrency(news_cfg: Dict[str, Any], src_cfg: Dict[str, Any], args) -> int:
    """Precedence (highest first): --concurrency CLI override > this
    source's own `concurrency` key > manual.news.collection.concurrency >
    DEFAULT_CONCURRENCY."""
    cli_override = getattr(args, "concurrency", None)
    if cli_override is not None:
        return cli_override
    if "concurrency" in src_cfg:
        return src_cfg["concurrency"]
    return news_cfg.get("collection", {}).get("concurrency", DEFAULT_CONCURRENCY)


def _resolve_historical_cfg(news_cfg: Dict[str, Any], src_cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge manual.news.historical_discovery (global defaults) with any
    per-source `historical:` override, then apply CLI overrides for this
    run only. CLI overrides never persist back to hindi_data.yaml."""
    merged = dict(news_cfg.get("historical_discovery", {}))
    src_historical = src_cfg.get("historical", {})
    if "supported" in src_historical and not src_historical["supported"]:
        merged["enabled"] = False
    if getattr(args, "start_date", None):
        merged["start_date"] = args.start_date
    if getattr(args, "end_date", None):
        merged["end_date"] = args.end_date
    if getattr(args, "max_discovered_urls", None) is not None:
        merged["max_discovered_urls"] = args.max_discovered_urls
    return merged


def _resolve_max_sitemap_documents(news_cfg: Dict[str, Any], args) -> int:
    if getattr(args, "max_sitemap_documents", None) is not None:
        return args.max_sitemap_documents
    return news_cfg.get("max_sitemap_documents", DEFAULT_MAX_SITEMAP_DOCUMENTS)


# ---------------------------------------------------------------------------
# Dry-run (discovery-only) / smoke-test
# ---------------------------------------------------------------------------

def run_dry_run(config: Dict[str, Any], repo_root: Path, args, logger) -> None:
    """Performs REAL discovery network I/O (sitemap documents only) and
    grows the persisted URL inventory, but fetches ZERO article bodies —
    see module docstring for why this deviates from the usual
    zero-network --dry-run convention."""
    news_cfg = _news_cfg(config)
    out_dir = _resolve_out_dir(config, repo_root)
    discovery_dir = out_dir / "_discovery"
    max_sitemap_documents = _resolve_max_sitemap_documents(news_cfg, args)

    logger.info("=" * 70)
    logger.info("DRY RUN — news collector — discovery only, ZERO article bodies fetched")
    logger.info("=" * 70)
    logger.info(f"news.enabled:              {news_cfg.get('enabled', True)}")
    logger.info(f"Output dir:                {out_dir}")
    logger.info(f"max_sitemap_documents:     {max_sitemap_documents} (this run)")

    sources = [args.source] if args.source else _configured_sources(news_cfg)
    summary: Dict[str, Any] = {"generated_at": collection_date_now(), "mode": "dry_run_discovery"}
    for name in sources:
        src_cfg = news_cfg.get(name, {})
        hist_cfg = _resolve_historical_cfg(news_cfg, src_cfg, args)
        already = load_collected_urls(out_dir, name) if out_dir.exists() else set()
        store = DiscoveryStore(discovery_dir, name)
        logger.info("-" * 70)
        logger.info(f"[{name}] enabled:               {src_cfg.get('enabled', False)}")
        logger.info(f"[{name}] discovery.type:         {(src_cfg.get('discovery') or {}).get('type', 'sitemap_rolling')}")
        logger.info(f"[{name}] historical enabled:     {hist_cfg.get('enabled', True)}")
        logger.info(f"[{name}] historical start_date:  {hist_cfg.get('start_date')}")
        logger.info(f"[{name}] historical end_date:    {hist_cfg.get('end_date') or 'today'}")
        logger.info(f"[{name}] already collected:      {len(already):,} URL(s) across existing shards")
        logger.info(f"[{name}] inventory before run:   {len(store.urls):,} URL(s)")

        if not src_cfg.get("enabled", False):
            logger.info(f"[{name}] disabled_reason: {src_cfg.get('disabled_reason', '(none given)')}")
            continue

        stats = discover_one_source(name, src_cfg, store, hist_cfg, max_sitemap_documents, logger)
        stats.urls_already_collected = sum(1 for u in store.urls if u in already)
        pending = [u for u in store.urls if u not in already and u not in store.rejected]
        stats.new_urls_available = len(pending)
        logger.info(f"[{name}] sitemap docs fetched:   {stats.sitemap_documents_fetched}")
        logger.info(f"[{name}] URLs discovered (run):  {stats.urls_discovered} ({stats.duplicate_urls} duplicate)")
        logger.info(f"[{name}] inventory after run:    {len(store.urls):,} URL(s) total")
        logger.info(f"[{name}] already collected:      {stats.urls_already_collected:,}")
        logger.info(f"[{name}] NEW, available to fetch: {stats.new_urls_available:,}")
        summary[name] = stats.to_dict()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "news_discovery_dry_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("DRY RUN complete — discovery inventory persisted, no article bodies fetched.")
    logger.info(f"Summary saved to {summary_path}")
    logger.info("=" * 70)


def run_smoke_test(config: Dict[str, Any], args, logger) -> None:
    news_cfg = _news_cfg(config)
    sources = [args.source] if args.source else _configured_sources(news_cfg)

    logger.info("=" * 70)
    logger.info("SMOKE TEST — news collector — connectivity check only")
    logger.info("=" * 70)

    any_unreachable_enabled = False
    for name in sources:
        src_cfg = news_cfg.get(name, {})
        enabled = src_cfg.get("enabled", False)
        base_url = src_cfg.get("base_url", "")
        if not base_url:
            logger.info(f"[{name}] enabled={enabled} — no base_url configured, skipping probe.")
            continue
        info = probe_url(base_url, logger, headers={"User-Agent": USER_AGENT})
        logger.info(f"[{name}] enabled={enabled} reachable={info.get('reachable')} "
                    f"(status={info.get('status_code')})")
        if enabled and not info.get("reachable"):
            any_unreachable_enabled = True

    if any_unreachable_enabled:
        logger.error("SMOKE TEST FAILED — one or more ENABLED news sources unreachable.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("SMOKE TEST PASSED — all enabled news sources reachable. No articles collected.")
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
    discovery_dir = out_dir / "_discovery"
    max_sitemap_documents = _resolve_max_sitemap_documents(news_cfg, args)
    max_docs_per_shard = news_cfg.get("max_docs_per_shard", 20_000)

    sources = [args.source] if args.source else _configured_sources(news_cfg)
    summary: Dict[str, Any] = {"generated_at": collection_date_now()}

    for name in sources:
        src_cfg = news_cfg.get(name, {})
        if not src_cfg.get("enabled", False):
            logger.info(f"manual.news.{name}.enabled is false — skipping "
                        f"(disabled_reason: {src_cfg.get('disabled_reason', 'not given')}).")
            continue

        if args.max_articles is not None:
            src_cfg = dict(src_cfg)
            src_cfg["max_articles"] = args.max_articles

        if args.overwrite:
            n = delete_existing_shards(out_dir, name)
            n += DiscoveryStore(discovery_dir, name).delete()
            logger.info(f"[{name}] --overwrite: deleted {n} existing shard/discovery file(s).")

        hist_cfg = _resolve_historical_cfg(news_cfg, src_cfg, args)
        writer = JsonlShardWriter(out_dir, prefix=name, max_docs_per_shard=max_docs_per_shard)
        already = load_collected_urls(out_dir, name)
        store = DiscoveryStore(discovery_dir, name)
        concurrency = _resolve_concurrency(news_cfg, src_cfg, args)
        logger.info("=" * 70)
        logger.info(f"{name.upper()} — discovering + collecting articles (concurrency={concurrency})")
        logger.info("=" * 70)
        stats = collect_one_source(
            name, src_cfg, hist_cfg, writer, already, store, max_sitemap_documents, logger,
            concurrency=concurrency,
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
            f"[{name}] {stats.to_dict()} "
            f"new_docs_this_run={new_docs_this_run} "
            f"total_docs_all_time={total_docs_all_time} "
            f"inventory_total={len(store.urls)}"
        )
        summary[name] = {
            **stats.to_dict(),
            "new_docs_this_run": new_docs_this_run,
            "total_docs_all_time": total_docs_all_time,
            "inventory_total": len(store.urls),
        }

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
        "Collect Hindi manual-corpus news documents from configurable, "
        "individually enable/disable-able sources. Discovery (growing a "
        "persisted historical URL inventory) is separate from article "
        "fetching — see module docstring for the full architecture."
    )
    parser.add_argument("--source", type=str, choices=list(NEWS_SOURCE_NAMES), default=None,
                         help="Collect only this one source (default: every source "
                              "present in manual.news, enabled or not — disabled "
                              "sources are skipped and reported, never silently).")
    parser.add_argument("--max-articles", type=int, default=None,
                         help="Override max_articles (the FETCH-phase cap: how many "
                              "pending discovered URLs to download article bodies for "
                              "this run) for whichever source(s) are collected. Must "
                              "be a positive integer. Omit to use each source's YAML default.")
    parser.add_argument("--concurrency", type=int, default=None,
                         help="Override the number of concurrent ARTICLE-FETCH worker "
                              "threads for this run (overrides "
                              "manual.news.collection.concurrency and any per-source "
                              "override). Must be a positive integer. Default: "
                              f"{DEFAULT_CONCURRENCY}. This does NOT affect discovery, "
                              "which is always sequential — keep this conservative.")
    parser.add_argument("--start-date", type=str, default=None,
                         help="Override historical_discovery.start_date (YYYY-MM-DD) "
                              "for this run only. Does not persist to hindi_data.yaml.")
    parser.add_argument("--end-date", type=str, default=None,
                         help="Override historical_discovery.end_date (YYYY-MM-DD) "
                              "for this run only. Does not persist to hindi_data.yaml.")
    parser.add_argument("--max-discovered-urls", type=int, default=None,
                         help="Override historical_discovery.max_discovered_urls (the "
                              "per-run cap on NEW URLs added to the persisted discovery "
                              "inventory) for this run only.")
    parser.add_argument("--max-sitemap-documents", type=int, default=None,
                         help="Override manual.news.max_sitemap_documents (the per-run "
                              "cap on sitemap XML documents fetched) for this run only.")
    args = parser.parse_args()

    if args.max_articles is not None and args.max_articles <= 0:
        parser.error(f"--max-articles must be a positive integer, got {args.max_articles}")
    if args.concurrency is not None and args.concurrency <= 0:
        parser.error(f"--concurrency must be a positive integer, got {args.concurrency}")
    if args.max_discovered_urls is not None and args.max_discovered_urls <= 0:
        parser.error(f"--max-discovered-urls must be a positive integer, got {args.max_discovered_urls}")
    if args.max_sitemap_documents is not None and args.max_sitemap_documents <= 0:
        parser.error(f"--max-sitemap-documents must be a positive integer, got {args.max_sitemap_documents}")
    for flag, value in (("--start-date", args.start_date), ("--end-date", args.end_date)):
        if value is not None:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                parser.error(f"{flag} must be YYYY-MM-DD, got {value!r}")

    config = load_config(args.config)
    logger = setup_logger(f"03_collect_{SOURCE_TYPE}", config)
    repo_root = Path(args.config).resolve().parent.parent.parent

    if args.dry_run:
        run_dry_run(config, repo_root, args, logger)
        return

    if args.smoke_test:
        run_smoke_test(config, args, logger)
        return

    run_collect(config, repo_root, args, logger)


if __name__ == "__main__":
    main()
