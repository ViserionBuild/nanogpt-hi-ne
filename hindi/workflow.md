# Phase 1 — Building Two Monolingual Corpora and Their Tokenizers

**Model H — Hindi (higher-resource) · Model L — Nepali (lower-resource)**
Shubham Kumar Sunny(2025201025)

---

## Where this ended up

|  | Hindi (Model H) | Nepali (Model L) |
|---|---:|---:|
| **Training tokens** | **503,783,232** | **505,739,399** |
| Documents | 1,654,946 | 2,428,442 |
| Words | 392,059,676 | 354,256,373 |
| Characters | 2,013,671,207 | 2,381,634,424 |
| UTF-8 bytes | 5,087,159,391 | 6,344,866,742 |
| **Manually collected** | **20.0041%** | **20.0007%** |
| Tokenizer | SentencePiece unigram, 16,000 pieces | SentencePiece unigram, 16,000 pieces |
| Fertility (tokens per word) | 1.2480 | 1.4318 |
| Characters per token | 4.0362 | 4.6975 |
| Unknown-token rate | 0.0000% | 0.0000% |

Both corpora clear the 500M-token target and both clear the 20% manual
requirement. Every token count is exact — the corpus was encoded with the
tokenizer that ships with it, not estimated from a characters-per-token ratio.

Every figure in this report is regenerated from the corpus by
`scripts/make_figures.py`, and `scripts/check_report_numbers.py` re-reads each
headline number out of the generated files and fails if the prose has drifted
from them.

---

## Choosing the two languages

**Hindi** is the higher-resource choice. It has more public text than any other
Indian language, which shows up concretely here: three of its four public
sources each hit an 800M-character collection cap without running dry — the cap
was ours, not theirs — and the crawler found enough independent news sites to
gather 609M characters of its own.

**Nepali** comes from the permitted lower-resource list, and it is genuinely
thinner. Its Wikipedia holds 37.7M characters against Hindi's 248.5M — 15% as
much, from the same kind of source. Its news sites republish each other so
heavily that near-duplicate detection removes 4.8× the share of documents it
removes from Hindi. Reaching the manual quota took two crawl sessions rather
than one. But it is not so scarce that 500M tokens is out of reach — which
matters, because a Model L that could not be built would have turned the
comparison into a study of a failed corpus rather than a study of a
low-resource one.

Both are written in **Devanagari**, and that is deliberate. Sharing a script
removes writing system as a confounding variable, so any difference between
Model H and Model L in later phases traces back to how much data was available,
not to the fact that two different alphabets were being modelled.

### What the two languages are like

Both are Indo-Aryan and both are written in **Devanagari**, an abugida: each
consonant carries an inherent /a/, and other vowels are written as dependent
signs (*matras*) attached above, below or beside it. Consonant clusters are
written as conjuncts joined by a virama (*halant*, `्`). Two consequences matter
for this project.

First, **the same visible glyph can have more than one Unicode encoding** —
precomposed nukta forms against base-plus-nukta sequences, for instance. Without
NFC normalisation the tokenizer learns two separate pieces for what a reader
sees as one letter, wasting vocabulary and fragmenting words. That is why NFC
runs before every other filter.

Second, **a character is not a keystroke and not a glyph**. `क्ष` is three code
points. Byte-level fallback therefore has to operate on UTF-8 bytes, not on
characters, if it is to guarantee losslessness — which is exactly what
`byte_fallback` does, and why the flag emoji in the worked examples costs eight
byte pieces rather than one.

**Hindi** is the third most spoken language in the world and India's primary
official language. It is inflectional with SOV word order, uses postpositions
rather than prepositions, and marks split ergativity with *ने* on the subject of
perfective transitive clauses. Verb compounding is pervasive (*करना*, *होना*,
*देना* as light verbs). Its measured profile here: **5.14 characters per word**,
237 words per document.

**Nepali** is on the permitted lower-resource list. It shares the script and a
great deal of vocabulary with Hindi, but its verbal morphology is markedly more
synthetic: person, number and honorificity agreement attach directly to the
stem, so a single orthographic word can carry what Hindi would spread across
several (*गरिरहनुभएको थियो*). Its postpositions differ too — *लाई*, *बाट*,
*देखि*, *सँग*.

That morphology is visible in the measurements rather than merely asserted:

|  | Hindi | Nepali |
|---|---:|---:|
| Characters per word | 5.14 | **6.72** |
| Words per document | 237 | 146 |
| Distinct word forms in the train split | 2,439,235 | **3,682,562** |
| Tokens per word, 16k unigram | 1.2480 | **1.4318** |
| Pieces covering 50% of all tokens | 242 | **536** |

Nepali words are 31% longer, and the corpus contains **51% more distinct word
forms** while holding 9% fewer word tokens overall. Longer, more varied words are
harder to cover with a fixed vocabulary, which is why Nepali needs more tokens
per word and why its token distribution is flatter. This is the clearest
statement of what makes Model L the harder of the two, and it is separate from
the question of how much data exists.


---

## How the corpus was built

Five stages, run in this order for each language independently. Nothing is
shared between the two: separate directories, separate source declarations,
separate tokenizers, separate statistics.

```
   COLLECT                    our own crawler ─┐
                              our own OCR ─────┼──> data/raw/manual/
                              HF downloader ───┴──> data/raw/downloaded/
                                                          │
   CLEAN                                                  ▼
                              normalise · filter · deduplicate
                                                          │
                                          data/clean/{manual,downloaded}.jsonl.gz
                                                          │
   BUDGET                                                 ▼
                              select a subset that hits ~500M tokens
                              while keeping manual ≥ 20%
                                                          │
   SPLIT                                                  ▼
                              96 / 2 / 2, at document level, seeded
                                                          │
                                          data/splits/{train,val,test}.jsonl.gz
                                                          │
   TOKENIZE                                               ▼
                              fit SentencePiece on the train split alone
                                                          │
                                          tokenizer/sp_<lang>_16000_uni.model
```

### What each stage does

**Collect.** Two origins, kept in separate directories so the manual share is
visible in the filesystem rather than only in a statistic. Manual means we
gathered it: `common/collect/crawler.py` and `parallel_crawler.py` fetch news
and literature pages, obeying `robots.txt` per host including `Crawl-delay`,
rate-limiting per host and backing off on 429 and 5xx responses; page bodies are
extracted with `trafilatura` so navigation and footers are discarded.
`common/collect/ocr.py` runs Tesseract over Devanagari page images rendered by
`pdftoppm`. Downloaded means an existing public release, streamed through the
Hugging Face datasets API so no dataset is ever materialised in full.

Every document carries thirteen provenance fields from the moment it is written
— `doc_id`, `source_id`, `source_url`, `collector`, `license`, `collected_at`
and the size counters — so any document in the final corpus can be traced back
to where it came from.

**Clean.** Filters run in a fixed order, cheapest first, so expensive checks
only see text that has already survived. Unicode is normalised to NFC before
anything else, because Devanagari can encode the same character several ways and
without normalisation the tokenizer would learn duplicate pieces for what a
reader sees as one letter.

**Budget.** After cleaning there is more text than the target needs, but not in
the right proportions. The 20% manual rule caps the whole corpus at five times
the manual token count, so surplus downloaded text is simply unusable. The
budgeter solves for the split directly rather than sampling and hoping.

**Split.** At document level, never mid-document, with a seeded shuffle so the
split is reproducible. Deduplication runs *before* splitting — doing it after
would leave copies of the same article in both train and test and quietly
inflate every Phase 2 evaluation score.

**Tokenize.** Fitted on the train split only. This is not a formality: the
tokenizer is part of the model, and a vocabulary fitted on validation text has
already seen the evaluation data.

### Running it in three passes

The obvious implementation holds the cleaned corpus in memory to shuffle and
budget it. Measured over 120,000 documents, that projected to roughly 30 GB on
an 8 GB machine. The pipeline instead makes three passes over disk-backed
intermediates — clean and hash, then select, then write splits — and never holds
document text in memory. Measured peak resident memory: **1.23 GB**.

---

## Repository layout

Organised by language, with each language directory self-contained, as the brief
requires. Anything shared between the two lives in `common/` and takes the
language as a parameter — there is no language-specific logic outside `hindi/`
and `nepali/`.

```
common/                        shared library — no language-specific logic
├── collect/
│   ├── crawler.py             breadth-first crawler: robots.txt per host,
│   │                          per-host rate limiting, exponential backoff,
│   │                          resumable frontier
│   ├── parallel_crawler.py    multi-host driver over crawler.py
│   ├── ocr.py                 Tesseract over pdftoppm page images
│   ├── hf_download.py         streaming downloader for public corpora
│   ├── hf_sources.py          which public dataset maps to which source_id
│   └── ia_books.py            Internet Archive fetcher for OCR input
├── data/
│   ├── schema.py              the Document record — 13 provenance fields + text
│   ├── registry.py            reads sources.yaml; cross-checks it against the data
│   ├── clean.py               NFC, boilerplate, length and script filters
│   ├── pipeline.py            the three-pass clean → budget → split driver
│   ├── budget.py              solves the ≥20% manual constraint
│   ├── split.py               seeded, document-level 96/2/2 partitioning
│   ├── cross_dedup.py         removes documents shared between the two corpora
│   ├── tokens.py              exact token counting, with tokenizer fingerprinting
│   ├── token_stats.py         coverage, rank/frequency, unused pieces
│   ├── stats.py               the corpus ledger
│   ├── status.py              live collection progress against a target
│   ├── writer.py              sharded gzip JSONL writer
│   ├── package.py             bundles a language workspace for transfer
│   └── repair.py              recovers a shard truncated by an unclean kill
├── tokenizer/
│   └── train.py               SentencePiece training, sampling and fertility
├── utils/
│   ├── paths.py               THE definition of the workspace layout
│   ├── io.py, logging.py, seed.py, config.py
├── model/  train/  eval/  attention/     Phase 2 and 3 scaffold (empty in Phase 1)

hindi/   nepali/                per-language workspaces, identical in shape
├── configs/                   crawl seed lists, with excluded hosts and why
├── data/
│   ├── sources.yaml           human-authored record of every source and licence
│   ├── raw/                   as collected, untouched                    (Drive)
│   │   ├── manual/            our own crawler and OCR — the ≥20% quota
│   │   └── downloaded/        public corpus releases
│   ├── clean/                 cleaned and deduplicated                   (Drive)
│   │   ├── manual.jsonl.gz
│   │   ├── downloaded.jsonl.gz
│   │   └── index/             which documents survived, and their token counts
│   ├── splits/                train / val / test — what training consumes (Drive)
│   ├── stats/                 generated evidence tables                  (in git)
│   │   ├── corpus_ledger.{json,md}    docs, words, chars, bytes, tokens
│   │   ├── budget_plan.md             the 20% decision, shown as arithmetic
│   │   ├── pipeline_report.json       raw scanned → kept, with every drop reason
│   │   ├── token_frequency_*.json     coverage, rank/frequency, unused pieces
│   │   └── progress/                  per-source collector snapshots     (Drive)
│   └── ocr_input/books/       source PDFs staged for OCR                 (Drive)
├── tokenizer/                 sp_<lang>_<8000|16000>_<uni|bpe>.{model,vocab}
│   ├── fertility_*.json       chars/token, tokens/word, <unk>, round-trip
│   └── train_sample_*.txt     the cached training extract       (not in git)
├── logs/                      kept in the repository — the brief requires them
│   ├── collect/               one log per source, plus crawler stdout
│   ├── pipeline/              cleaning, budgeting and splitting runs
│   ├── tokenizer/             every tokenizer training run
│   └── audit/                 per-URL crawl trails                       (Drive)
├── model/  train/  eval/  finetune/  checkpoints/    Phase 2 and 3 scaffold

report/
├── phase1/
│   ├── phase1_report.md            this document — results and evidence
│   ├── phase1_methodology.md       why each decision was made
│   ├── tokenizer_examples.md       four sentences through all eight tokenizers
│   ├── word_frequency.json         word statistics behind figures 9 and 10
│   ├── figures/                    the ten generated figures
│   └── archive/                    superseded snapshots, clearly marked
└── phase2/  phase3/                figure directories for later phases

scripts/                       grader-facing tools
├── verify_setup.py            environment, corpus and tokenizer health check
├── show_stats.py              every statistic, including the leakage proof
├── check_report_numbers.py    fails if the reports drift from the data
├── try_tokenizer.py           tokenize any sentence, any model, interactively
├── make_figures.py            regenerates every figure in this report
└── make_drive_bundle.sh       builds the Drive upload

tests/
└── test_pipeline_layout.py    end-to-end pipeline self-test on a synthetic
                               corpus; runs without any real data present

.tessdata/                     Tesseract hin/nep language data, for OCR
requirements.txt               pinned dependencies
README.md                      reproduction steps and the Drive links
```

**Why the split between `common/` and the language directories.** Every
language-specific fact — which sites to crawl, which public corpora to pull,
what the token target is — lives in `<lang>/configs/` or `<lang>/data/sources.yaml`.
Every algorithm lives in `common/` and is called with a language directory as an
argument. That is what makes the two pipelines genuinely independent rather than
one pipeline with a flag: running Hindi cannot read Nepali's data, because it is
never given the path.

---


## Stage 1 — Collection

### Hindi, as collected

| Source | Origin | Documents | Words | Characters |
|---|---|---:|---:|---:|
| `own_crawl_hi_news` | manual | 177,528 | 104,449,556 | 608,798,943 |
| `fineweb2_hi` | downloaded | 305,161 | 158,703,418 | 800,003,397 |
| `indiccorp_v2_hi` | downloaded | 1,674,926 | 158,206,874 | 800,000,162 |
| `sangraha_hi` | downloaded | 367,809 | 155,433,836 | 800,016,174 |
| `wikipedia_hi` | downloaded | 125,766 | 45,202,958 | 248,503,087 |
| **Manual total** | | **177,528** | **104,449,556** | **608,798,943** |
| **Downloaded total** | | **2,473,662** | **517,547,086** | **2,648,522,820** |
| **Raw total** | | **2,651,190** | **621,996,642** | **3,257,321,763** |

Manual text is 18.69% of the raw characters collected.

### Nepali, as collected

| Source | Origin | Documents | Words | Characters |
|---|---|---:|---:|---:|
| `own_crawl_ne_news` | manual | 259,502 | 113,781,476 | 800,011,025 |
| `ocr_ne_books` | manual | 1,358 | 359,995 | 2,224,047 |
| `fineweb2_ne` | downloaded | 356,884 | 118,468,083 | 800,002,430 |
| `indiccorp_v2_ne` | downloaded | 2,091,711 | 119,650,020 | 800,000,119 |
| `sangraha_ne` | downloaded | 413,638 | 119,333,746 | 800,000,410 |
| `wikipedia_ne` | downloaded | 24,282 | 5,615,860 | 37,655,671 |
| **Manual total** | | **260,860** | **114,141,471** | **802,235,072** |
| **Downloaded total** | | **2,886,515** | **363,067,709** | **2,437,658,630** |
| **Raw total** | | **3,147,375** | **477,209,180** | **3,239,893,702** |

Manual text is 24.76% of the raw characters collected — a deliberately larger
margin than Hindi's, because Nepali loses far more to near-duplication in the
next stage.

![Collection funnel](figures/fig1_collection_funnel.png)

Two things are worth noticing in these tables. Wikipedia is the clearest
illustration of what "low-resource" means: 248M characters in Hindi against 37M
in Nepali, from the same kind of source. And `indiccorp_v2` contributes far more
*documents* than the other downloaded sources for roughly the same character
count, because its documents are much shorter.

Duplicates rejected at write time, before the corpus was even assembled: 32,360
Hindi and 39,758 Nepali.

---

## Stage 2 — Cleaning and deduplication

Nine filters, in this order:

1. **Language guard** — a document whose `lang` field disagrees with the
   workspace is dropped outright, as a contamination check.
2. **Unicode NFC normalisation.**
3. **URL and boilerplate stripping.**
4. **Whitespace collapse.**
5. **Minimum length** — 200 characters after cleaning.
6. **Devanagari script ratio** — at least 70% of letters must be Devanagari.
7. **Exact deduplication** — SHA-256 over the normalised text.
8. **Near-duplicate detection** — each document is cut into 8-word shingles,
   each shingle hashed with blake2b, and the 16 smallest hashes kept as its
   signature. A document sharing at least half its signature with one already
   seen is dropped. Signatures shared by more than 64 documents are treated as
   boilerplate and ignored, so a common navigation phrase cannot make unrelated
   articles look like copies of each other.
9. **Cross-corpus deduplication** — steps 7 and 8 look within one language. The
   brief also requires that the two corpora share no documents, and nothing was
   checking it. See below.

### What cleaning removed

| Reason | Hindi | Share of raw | Nepali | Share of raw |
|---|---:|---:|---:|---:|
| Near-duplicate | 23,325 | 0.88% | **132,561** | **4.21%** |
| Low script ratio | 27,739 | 1.05% | 19,321 | 0.61% |
| Exact duplicate | 13,088 | 0.49% | 2,932 | 0.09% |
| Too short | 1,729 | 0.07% | 2,686 | 0.09% |
| Wrong language | 0 | 0.00% | 0 | 0.00% |
| **Survived** | **2,585,309** | **97.5%** | **2,989,875** | **95.0%** |

![What cleaning removed](figures/fig4_cleaning_drops.png)

The asymmetry in the top row is the single clearest measurement of the
difference between the two languages. Nepali loses 4.8× more documents to
near-duplication than Hindi, because its roughly two dozen news sites republish
each other's wire copy heavily, where Hindi's larger outlets produce more of
their own. That one number is why the Nepali crawl needed a second session to
reach the manual quota.

### What the cleaned text looks like

Two views of the corpus that do not depend on any tokenizer, so they describe
the text itself rather than a choice made about it.

![The most frequent words in each corpus](figures/fig9_top_words.png)

Both lists are function words — postpositions, the copula, conjunctions — which
is what ordinary running prose should produce. Nothing in either top thirty is
boilerplate (`Read more`, `Copyright`, a site name), which is the check this
figure exists to perform: navigation text that survives extraction shows up here
immediately.

The shapes differ in a way that matches the morphology. Hindi's `के` alone is
4.03% of all word tokens, and its top three account for 10.0%. Nepali's most
frequent word, `र`, is only 1.99%, and its top three reach 4.9%. Nepali spreads
the same quantity of text over more distinct forms because its inflection is
carried inside the word rather than by separate postpositions — the same fact
that later makes its tokenizer need more pieces per word.

![Word clouds of the two training corpora](figures/fig10_word_clouds.png)

| | Hindi | Nepali |
|---|---:|---:|
| Word tokens in the train split | 362,409,794 | 329,034,532 |
| Distinct word forms | 2,439,235 | **3,682,562** |
| Most frequent word's share | 4.03% (`के`) | 1.99% (`र`) |

Nepali has **51% more distinct word forms from 9% fewer word tokens**. That
single comparison is the clearest statement of why the low-resource model is the
harder one, and it is a property of the language rather than of the data volume.

### Documents shared between the two corpora

Hashing every document of both corpora and intersecting found **106 documents
that were byte-identical across Hindi and Nepali**.

They are not a collection bug. No `doc_id` and no `source_id` is shared, so both
crawls found them independently — the two languages use the same script and
share a great deal of vocabulary, so a short quotation, a list of place names or
a Sanskrit line can legitimately appear in both.

**Both copies were removed, not one.** If identical text sits in a corpus
labelled `hi` and one labelled `ne`, at least one of those labels is wrong and
nothing in the data says which; keeping either would knowingly leave a
mislabelled document in a corpus described as monolingual. At 106 documents out
of 4.1 million the cost is far below the noise in the token target, and the
alternative is a guess.

| Removed from | Hindi | Nepali |
|---|---:|---:|
| train | 101 | 102 |
| validation | 2 | 2 |
| test | 3 | 2 |
| **Total** | **106** | **106** |

This runs after splitting and before the shipping tokenizer is trained, so no
tokenizer is ever fitted on text that is about to be removed. Re-running the
check now reports zero shared documents.

### The cleaned pool

Stored as one file per origin, so the manual/downloaded division survives into
the intermediate:

|  | Hindi | Nepali |
|---|---|---|
| `clean/manual.jsonl.gz` | 154,011 documents | 241,162 documents |
| `clean/downloaded.jsonl.gz` | 2,454,623 documents | 2,881,274 documents |

Documents keep a single global index spanning both files, stored in
`clean/index/`, so the two streams merge back into their original order. That is
what lets a re-split reproduce the shipped corpus exactly.

---

## Stage 3 — The 20% manual rule

The rule binds harder than it first appears. If manual text must be at least 20%
of the final corpus, then

```
    total  ≤  manual ÷ 0.20  =  5 × manual
```

The corpus can never be larger than five times what was manually collected, no
matter how much downloaded text is available. Both languages had far more
downloaded text than they could use; the manual pool, not the public corpora,
set the ceiling.

|  | Hindi | Nepali |
|---|---:|---:|
| Target requested | 505,000,000 | 507,000,000 |
| Manual budgeted | 101,000,000 | 101,400,000 |
| Downloaded budgeted | 404,000,000 | 405,600,000 |
| Feasible | yes | yes |
| Limiting factor | target met | target met |
| **Achieved after recount** | **503,783,232 (20.0041%)** | **505,739,399 (20.0007%)** |

Both targets sit deliberately above the 500M floor. The corpus is *selected*
using one tokenizer and *recounted* with its retrained successor, and that drift
has been measured at up to −0.21%. An exact 500M target would leave no room for
it.

### The final corpus, by origin

**Hindi**

| Origin | Documents | Words | Characters | Tokens | Share |
|---|---:|---:|---:|---:|---:|
| Manual | 137,914 | 77,669,017 | 407,559,765 | 100,777,521 | **20.0041%** |
| Downloaded | 1,517,032 | 314,390,659 | 1,606,111,442 | 403,005,711 | 79.9959% |
| **Total** | **1,654,946** | **392,059,676** | **2,013,671,207** | **503,783,232** | 100% |

**Nepali**

| Origin | Documents | Words | Characters | Tokens | Share |
|---|---:|---:|---:|---:|---:|
| Manual | 177,927 | 71,792,743 | 486,398,245 | 101,151,534 | **20.0007%** |
| Downloaded | 2,250,515 | 282,463,630 | 1,895,236,179 | 404,587,865 | 79.9993% |
| **Total** | **2,428,442** | **354,256,373** | **2,381,634,424** | **505,739,399** | 100% |

![Manual versus downloaded share](figures/fig3_manual_quota.png)

Note how much of the collected pool goes unused. Hindi keeps 63.4% of its
cleaned documents and Nepali 77.8% — the rest is downloaded text that would have
pushed the manual share below 20%.

### Which sources the final tokens came from

| Source | Hindi tokens | Share | Nepali tokens | Share |
|---|---:|---:|---:|---:|
| `own_crawl_*_news` (manual) | 100,777,521 | 20.00% | 100,609,595 | 19.89% |
| `ocr_ne_books` (manual) | — | — | 541,939 | 0.11% |
| `fineweb2_*` | 120,328,111 | 23.88% | 137,073,974 | 27.10% |
| `indiccorp_v2_*` | 120,310,187 | 23.88% | 132,531,349 | 26.21% |
| `sangraha_*` | 124,624,548 | 24.74% | 128,589,313 | 25.43% |
| `wikipedia_*` | 37,742,865 | 7.49% | 6,393,229 | 1.26% |

![Corpus composition by source](figures/fig2_corpus_composition.png)

Wikipedia's asymmetry survives all the way to the finished corpus: 7.49% of
Hindi against 1.26% of Nepali.

---

## Stage 4 — Splitting

Split at document level, never mid-document, with a seeded shuffle (seed 42) so
the split is reproducible from the same inputs.

**Hindi**

| Split | Documents | Share | Tokens | Characters |
|---|---:|---:|---:|---:|
| train | 1,588,748 | 96.00% | 483,776,641 | 1,933,722,585 |
| validation | 33,099 | 2.00% | 9,861,236 | 39,477,067 |
| test | 33,099 | 2.00% | 10,145,355 | 40,471,555 |

**Nepali**

| Split | Documents | Share | Tokens | Characters |
|---|---:|---:|---:|---:|
| train | 2,331,304 | 96.00% | 485,551,056 | 2,286,615,069 |
| validation | 48,568 | 2.00% | 10,044,186 | 47,292,156 |
| test | 48,570 | 2.00% | 10,144,157 | 47,727,199 |

96/2/2 rather than the more usual 80/10/10 because the corpus is large: 2% of
Nepali is still 48,570 documents and 10.1M tokens, which is far more than enough
to estimate validation loss stably, and every document held back is one the
model cannot learn from. Two properties are verified rather than assumed — no
document straddles two splits, and no document *text* appears in more than one
split.

### Manual text is spread across all three splits

The shuffle is over the whole selected corpus, so manual and downloaded
documents land in every split in roughly their corpus proportion:

| Split | Manual docs | Downloaded docs | Manual words | Manual share of words |
|---|---:|---:|---:|---:|
| **Hindi** train | 132,393 | 1,456,355 | 74,635,229 | 19.82% |
| Hindi validation | 2,809 | 30,290 | 1,556,262 | 20.22% |
| Hindi test | 2,712 | 30,387 | 1,477,526 | 18.77% |
| **Nepali** train | 170,863 | 2,160,441 | 68,912,644 | 20.26% |
| Nepali validation | 3,486 | 45,082 | 1,418,189 | 20.15% |
| Nepali test | 3,578 | 44,992 | 1,461,910 | 20.58% |

This is a deliberate choice, and the alternative is tempting: putting *all*
manual text in the training split would raise the manual share there to about
25% and make the quota easier to satisfy. It would also make validation and test
sets that are purely public web text, and therefore not representative of the
corpus the model was trained on. Held-out data exists to estimate performance on
the training distribution; skewing its composition breaks that. The quota is
defined over the whole corpus, and it is met over the whole corpus.

### Disk footprint

| Artifact | Hindi | Nepali |
|---|---:|---:|
| `splits/train.jsonl.gz` | 1,169 MB | 1,426 MB |
| `splits/val.jsonl.gz` | 24.0 MB | 29.6 MB |
| `splits/test.jsonl.gz` | 24.6 MB | 29.8 MB |
| `clean/` (both origins) | 2.83 GB | 2.97 GB |
| `raw/` (both origins) | 2.04 GB | 2.08 GB |

All corpus files are gzipped JSON Lines, one document per line, read natively by
every module in the codebase. The cleaned pool is larger than the raw pool
despite holding fewer documents, because it is written at gzip level 1 — it is a
working intermediate where write speed matters more than size, and recompressing
it at level 6 would bring it to roughly 1.96 GB for Hindi.

---

## Stage 5 — The tokenizers

| Setting | Value | Reason |
|---|---|---|
| Library | SentencePiece 0.2.2 | no pretrained tokenizer is downloaded or adapted anywhere |
| Algorithm | unigram | scores whole candidate segmentations against a global objective, where BPE replays a fixed merge list |
| Vocabulary | 16,000 | forced by the 25M-parameter budget |
| `character_coverage` | 0.9995 | leaves the rarest 0.05% of characters to byte fallback rather than spending slots on them |
| `byte_fallback` | on | makes the mapping lossless |
| `max_sentence_length` | 65,536 | one document is one line; the 4192-byte default was silently discarding half the sample |
| Fitted on | the train split only | fitting on validation text leaks evaluation data into the vocabulary |

Throughout this report, `␣` stands for SentencePiece's word-boundary marker
(U+2581, written `▁` in the vocabulary files), so `␣के` is the piece "के
occurring at the start of a word".

**Eight tokenizers exist** — two languages × two vocabulary sizes × two
algorithms. One ships per language; the other three per language are the
evidence behind the two design choices.

|  | Hindi | Nepali |
|---|---|---|
| Shipped | `sp_hi_16000_uni.model` | `sp_ne_16000_uni.model` |
| Comparisons | `sp_hi_8000_uni`, `sp_hi_16000_bpe`, `sp_hi_8000_bpe` | `sp_ne_8000_uni`, `sp_ne_16000_bpe`, `sp_ne_8000_bpe` |

### How all eight perform

Measured on the same 20,000 held-out validation documents per language — text no
tokenizer was trained on.

| Language | Vocab | Algorithm | Chars/token | Bytes/token | Tokens/word | Byte-fallback | `<unk>` | Round-trip |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Hindi | 16,000 | **unigram (shipped)** | 4.0362 | 10.2436 | **1.2480** | 0.1908% | 0 | exact |
| Hindi | 16,000 | BPE | 3.9839 | 10.1110 | 1.2644 | 0.1883% | 0 | exact |
| Hindi | 8,000 | unigram | 3.7121 | 9.4212 | 1.3570 | 0.1755% | 0 | exact |
| Hindi | 8,000 | BPE | 3.6383 | 9.2340 | 1.3845 | 0.1720% | 0 | exact |
| Nepali | 16,000 | **unigram (shipped)** | 4.6975 | 12.5395 | **1.4318** | 0.2798% | 0 | exact |
| Nepali | 16,000 | BPE | 4.6393 | 12.3842 | 1.4497 | 0.2763% | 0 | exact |
| Nepali | 8,000 | unigram | 4.2094 | 11.2367 | 1.5978 | 0.2507% | 0 | exact |
| Nepali | 8,000 | BPE | 4.1190 | 10.9952 | 1.6328 | 0.2453% | 0 | exact |

![Tokenizer fertility and vocabulary composition](figures/fig6_tokenizer.png)

**Bytes per token** is the figure that matters for the bits-per-byte comparison
in Phase 2, and it is where the two languages differ most: 10.24 for Hindi
against 12.54 for Nepali. **Byte-fallback** stays under 0.3% of all tokens
everywhere — the 256 reserved slots are doing real work without costing
sequence length, and `<unk>` is zero for all eight.

Nepali needs more tokens per word than Hindi at every setting. That is not a
defect in the tokenizer — it is Nepali's richer inflection showing up as a
measurement, and it is the same fact that reappears in the token-frequency
statistics as a flatter distribution.

### What the vocabularies are made of

Percentages are of the *learned* pieces. Byte fallback reserves 259 slots in
every model — `<unk>`, `<s>`, `</s>` and the 256 `<0xNN>` byte pieces — leaving
15,741 learned pieces at 16,000 and 7,741 at 8,000.

| Language | Vocab | Algorithm | Devanagari | Latin | Digits | Other | Word-initial |
|---|---:|---|---:|---:|---:|---:|---:|
| Hindi | 16,000 | unigram | 89.1% | 7.2% | 3.2% | 0.5% | 12,314 |
| Hindi | 16,000 | BPE | 89.8% | 6.2% | 3.4% | 0.6% | 11,104 |
| Hindi | 8,000 | unigram | 89.7% | 6.3% | 3.3% | 0.7% | 5,946 |
| Hindi | 8,000 | BPE | 90.9% | 5.2% | 3.0% | 0.9% | 5,261 |
| Nepali | 16,000 | unigram | 97.1% | 2.1% | 0.5% | 0.3% | 12,675 |
| Nepali | 16,000 | BPE | 97.8% | 1.6% | 0.3% | 0.3% | 11,552 |
| Nepali | 8,000 | unigram | 97.2% | 1.9% | 0.4% | 0.5% | 6,145 |
| Nepali | 8,000 | BPE | 97.5% | 1.7% | 0.3% | 0.5% | 5,326 |

Two things stand out. Hindi carries **3.4× more Latin vocabulary than Nepali**
(7.2% against 2.1%), because Hindi news uses far more English acronyms and brand
names. And unigram learns **11% more word-initial pieces than BPE** at the same
size — 12,314 against 11,104 for Hindi 16k — which is the mechanism behind its
better fertility: more whole words earn a slot of their own.

---

## How much text the tokenizer actually saw

A fair question, and the honest answer is: a sample, not the corpus.

|  | Hindi | Nepali |
|---|---:|---:|
| Train split — the eligible pool | 1,588,748 docs · 1,933,722,585 chars | 2,331,304 docs · 2,286,615,069 chars |
| Extract cap (`max_bytes`) | 400 MB | 400 MB |
| Sources drawn from | 5 of 5 | 6 of 6 |
| Documents extracted | 125,703 | 150,106 |
| Discarded as too long | 193 (0.15%) | 121 (0.08%) |
| **Documents used for training** | **125,510** | **149,985** |
| **Characters used for training** | **149,440,900** | **145,303,708** |
| **Share of the train split** | **7.73%** | **6.35%** |
| **Share of the whole corpus** | **7.42%** | **6.10%** |
| Distinct characters seen | 155 | 127 |
| Character coverage achieved | 0.999512 | 0.999517 |

So roughly **7% of the corpus for Hindi and 6% for Nepali**. All four tokenizers
of a language are trained on the identical extract — it is written once to
`<lang>/tokenizer/train_sample_<code>.txt` and cached with a stamp recording the
split's size and modification time — so the 8k-versus-16k and unigram-versus-BPE
comparisons each differ in exactly one variable.

### Why a sample is enough, and where it is not

Fitting a vocabulary is a frequency-statistics problem, and it is a different
question from training a model:

|  | What it learns | What it sees |
|---|---|---|
| Tokenizer (Phase 1) | which 16,000 substrings deserve a slot | a ~150M character sample |
| Model (Phase 2) | the language itself | **all ~500M tokens** |

At 149M characters, every one of the 15,741 learned pieces still has thousands
of observations — the token-frequency statistics below confirm it directly, and
the exact token counts in this report come from encoding *every* document, not
the sample.

### Where the sample comes from

The extract is drawn **proportionally from every source**, not read top-down.
`measure_sources()` walks the split once to size each source, then a second pass
takes each source's share:

| Hindi source | Sampled | Share | | Nepali source | Sampled | Share |
|---|---:|---:|---|---|---:|---:|
| `own_crawl_hi_news` (manual) | 79.28 MB | **19.8%** | | `own_crawl_ne_news` (manual) | 81.18 MB | **20.3%** |
| `indiccorp_v2_hi` | 99.54 MB | 24.9% | | `ocr_ne_books` (manual) | 0.28 MB | 0.1% |
| `sangraha_hi` | 97.61 MB | 24.4% | | `fineweb2_ne` | 108.27 MB | 27.1% |
| `fineweb2_hi` | 94.78 MB | 23.7% | | `indiccorp_v2_ne` | 105.14 MB | 26.3% |
| `wikipedia_hi` | 28.79 MB | 7.2% | | `sangraha_ne` | 100.77 MB | 25.2% |
| | | | | `wikipedia_ne` | 4.35 MB | 1.1% |

Each source's share of the sample tracks its share of the train split, so the
manual crawl is represented at very nearly its 20% corpus weight.

### This was not always true

Two defects, both found by measurement and both now fixed:

|  | Before | After |
|---|---|---|
| Sources in the Hindi sample | 1 of 5 (FineWeb-2 only) | **5 of 5** |
| Manual crawl's share of the sample | 0% | **19.8%** Hindi, **20.3%** Nepali |
| Discarded as too long | 50.9% Hindi, 44.3% Nepali | **0.15%**, **0.08%** |
| Characters trained on | 31.4M Hindi, 35.5M Nepali | **149.4M**, **145.3M** |
| Hindi fertility | 1.2579 | **1.2480** |
| Nepali fertility | 1.4433 | **1.4318** |

The first defect: the extract read the split from the top until it hit the byte
cap, and the split is written in source-block order, so it never reached past
the first source. The second: SentencePiece skips any line longer than
`max_sentence_length`, and at the 4192-byte default that silently discarded
about half the extract and biased what remained toward short documents.

All eight tokenizers improved. The most visible consequence appears in the
worked examples below — Hindi's Devanagari numerals gained vocabulary slots,
because the manually crawled news that writes numbers that way is now in the
sample.

---

## Token-frequency statistics

Fertility says how *many* tokens a text costs. This says **which** tokens they
are — and it is the check that fertility cannot perform: a 16,000-piece
vocabulary in which 3,000 pieces never occur is really a 13,000-piece
vocabulary, and every fertility number would look identical.

Counted by `common/data/token_stats.py` over the **entire train split** with
each language's shipping tokenizer, so these describe the exact text Phase 2
will consume.

| | Hindi (16k unigram) | Nepali (16k unigram) |
|---|---:|---:|
| Documents counted (train split) | 1,588,748 | 2,331,304 |
| Token occurrences counted | 483,776,641 | 485,551,056 |
| Vocabulary size | 16,000 | 16,000 |
| Pieces that occur at least once | 15,873 | 15,894 |
| **Pieces never used** | **127** | **106** |
| Pieces used exactly once | 1 | 2 |
| Mean occurrences per used piece | 30,478 | 30,549 |
| Median occurrences per used piece | 6,806 | 8,699 |
| `<unk>` occurrences | 0 | 0 |

**Not one learned piece is wasted.** 99.21% of Hindi pieces and 99.34% of Nepali pieces occur in
running text, and the breakdown of those that do not settles what kind of
slot is going unused:

| Never-used pieces, by kind | Hindi | Nepali |
|---|---:|---:|
| Byte fallback (`<0xNN>`) / control | 127 | 106 |

**All 127 unused Hindi slots and all 106 unused Nepali slots are byte-fallback pieces** —
byte values that never occur in that corpus. They are not waste; they are the
reserve that makes the mapping lossless for input the corpus has never seen
(see byte fallback, below). Every learned piece is used. Only 1 Hindi and 2 Nepali pieces occur
a single time, so there is no long tail to reclaim either.

### Coverage — how top-heavy the distribution is

| Share of all token occurrences | Hindi: pieces needed | Nepali: pieces needed |
|---|---:|---:|
| Top pieces covering 50% of all tokens | 242 | 536 |
| … 80% | 2,624 | 3,641 |
| … 90% | 5,691 | 6,879 |
| … 95% | 8,773 | 9,810 |
| … 99% | 13,420 | 13,939 |

Half of all Hindi text is just **242 pieces** — 1.5% of the vocabulary. Nepali needs **536**, 3.4%.
Nepali's distribution is flatter: its richer inflection spreads the same
quantity of text over more distinct pieces, which is the same fact that shows
up in the fertility table as a higher fertility (flatter head, more pieces per word).

### Which pieces, by script

Vocabulary *slots* and running *text* are different questions. Latin holds 8.0%
of the Hindi vocabulary (above) but far less of the text:

| Category | Hindi tokens | Hindi share | Nepali tokens | Nepali share |
|---|---:|---:|---:|---:|
| Devanagari | 430,181,034 | 88.92% | 457,536,075 | 94.23% |
| Latin (English fragments) | 14,783,910 | 3.06% | 3,825,185 | 0.79% |
| Digits | 8,259,187 | 1.71% | 769,751 | 0.16% |
| Byte fallback / control | 1,826,055 | 0.38% | 1,271,540 | 0.26% |
| Punctuation and other | 28,726,455 | 5.94% | 22,148,505 | 4.56% |

### The most frequent pieces

A sanity check with a clear expectation: the top of a well-fitted vocabulary
should be function words, case markers and inflectional endings — never
content words, and never fragments.

| Rank | Hindi piece | Occurrences | Share | Nepali piece | Occurrences | Share |
|---:|---|---:|---:|---|---:|---:|
| 1 | `␣के` | 14,741,630 | 3.05% | `␣।` | 13,999,058 | 2.88% |
| 2 | `।` | 13,436,464 | 2.78% | `,` | 9,904,719 | 2.04% |
| 3 | `␣में` | 11,150,288 | 2.30% | `को` | 8,935,950 | 1.84% |
| 4 | `␣है` | 10,617,116 | 2.19% | `␣र` | 6,936,488 | 1.43% |
| 5 | `,` | 9,861,199 | 2.04% | `।` | 6,934,588 | 1.43% |
| 6 | `␣की` | 8,811,544 | 1.82% | `मा` | 6,825,082 | 1.41% |
| 7 | `␣से` | 6,776,011 | 1.40% | `␣छ` | 6,412,476 | 1.32% |
| 8 | `␣को` | 6,608,217 | 1.37% | `का` | 4,891,741 | 1.01% |
| 9 | `␣और` | 6,250,120 | 1.29% | `ले` | 4,850,664 | 1.00% |
| 10 | `␣का` | 5,310,516 | 1.10% | `␣पनि` | 3,287,202 | 0.68% |
| 11 | `.` | 4,518,537 | 0.93% | `लाई` | 3,004,970 | 0.62% |
| 12 | `␣पर` | 4,147,043 | 0.86% | `␣छन्` | 2,221,762 | 0.46% |

Both lists are exactly what they should be —
postpositions, the copula, the danda `।`, and the comma. Nothing in either top
twelve is a content word, which is the signal that the vocabulary is fitted to
the language rather than to a topic in the sample.

![Token-frequency statistics](figures/fig8_token_frequency.png)

The left panel is the rank/frequency curve on log-log axes. A Zipfian
distribution appears as a straight line, and both languages are close to one
over three decades — the expected shape for natural-language text, and evidence
that neither corpus is dominated by boilerplate. The right panel is the same
data as cumulative coverage.

Full output, including the 25 most frequent pieces and the downsampled curve:
`<lang>/data/stats/token_frequency_<code>.json`.

---

## Choosing the vocabulary size

The choice is forced by the **25M parameter budget, including embeddings**. With
tied embeddings the embedding table costs `vocabulary × d_model`, and at
`d_model = 384`:

| Vocabulary | Embedding at `d=384` | Share of 25M | Embedding at `d=512` | Share of 25M |
|---:|---:|---:|---:|---:|
| 8,000 | 3.07M | 12.3% | 4.10M | 16.4% |
| **16,000** | **6.14M** | **24.6%** | **8.19M** | **32.8%** |
| 24,000 | 9.22M | 36.9% | 12.29M | 49.2% |
| 32,000 | 12.29M | 49.2% | 16.38M | 65.5% |

At `d=384` a 16,000-piece vocabulary leaves about 18.9M parameters for the
transformer stack — roughly ten layers. A 32,000-piece vocabulary leaves 12.7M,
or seven, and at `d=512` it consumes **65.5% of the entire budget**, which would
leave two or three layers. The vocabulary would be most of the model.

### The choice depends on two architecture decisions, so both are stated

This table is only valid under the configuration Phase 2 will actually use, and
the numbers move a long way if either assumption changes:

| Assumption | Chosen here | Effect on the table |
|---|---|---|
| **Tied embeddings** — input embedding matrix reused as the output projection | yes | Untying doubles every embedding cost to `2 × V × d`, so 16,000 at `d=512` would consume **65.5%** rather than 32.8% |
| **`d_model = 384`** | yes | At `d=512` every embedding cost rises by a third |

Tying is standard for models this small and is what makes 16,000 affordable:
untied at `d=512`, a 16,000-piece vocabulary really would crowd out the
transformer, and 8,000 would be the right answer instead. Stating the
assumption matters more than the conclusion — the same arithmetic supports a
different vocabulary size under a different architecture, and a reader should be
able to check which one they are looking at.

The parameter saving from tying is also not free of consequences: input and
output representations are forced to share a space. For a ~25M model trained on
500M tokens that is a good trade, and it is the configuration this vocabulary
size was chosen for.

32,000 is unusable here, and no 32,000-piece tokenizer ships. Between 8k and
16k, measured on the real corpora:

|  | Hindi 8k | **Hindi 16k** | Nepali 8k | **Nepali 16k** |
|---|---:|---:|---:|---:|
| Characters per token | 3.7121 | **4.0362** | 4.2094 | **4.6975** |
| Tokens per word | 1.3570 | **1.2480** | 1.5978 | **1.4318** |
| Sequences longer by | +8.7% | — | +11.6% | — |

![Vocabulary trade-off](figures/fig5_vocab_tradeoff.png)

Halving the vocabulary returns 3.07M parameters — 12.3% of the budget, worth
roughly two extra transformer layers — and costs 8.7% longer Hindi sequences and
11.6% longer Nepali ones. Since attention is quadratic in sequence length that
compute is not free either, and **Nepali is hurt about a third harder than
Hindi**, which is the wrong direction: Model L is the one that can least afford
to lose context. 16k also gives each used piece roughly 30,500 mean occurrences,
so no slot survives on thin evidence.

> **A trap worth naming.** A smaller vocabulary produces *more tokens from the
> same text*, so one could shrink the vocabulary until the 500M target is met.
> That would be gaming the metric rather than collecting data. The measurement
> rules it out anyway: even at 8,000, the Nepali corpus at that time yielded
> only 475.7M — still short. More data was needed regardless, so vocabulary size
> was settled purely on modelling grounds.

---

## Choosing the algorithm

|  | BPE | **Unigram (chosen)** |
|---|---|---|
| Method | greedily merges the most frequent adjacent pair | starts large and **prunes** to maximise corpus likelihood |
| Objective | local, greedy | global, probabilistic |
| Can weigh competing segmentations | no | yes |
| Subword regularization | no | available for Phase 2 |

Both algorithms were trained at both vocabulary sizes for both languages, on the
identical sample and with byte fallback on, so the algorithm is the only
variable.

|  | Hindi 16k | Hindi 8k | Nepali 16k | Nepali 8k |
|---|---:|---:|---:|---:|
| BPE tokens per word | 1.2644 | 1.3845 | 1.4497 | 1.6328 |
| **Unigram tokens per word** | **1.2480** | **1.3570** | **1.4318** | **1.5978** |
| **BPE needs more tokens by** | **+1.31%** | **+2.03%** | **+1.25%** | **+2.19%** |

![Unigram versus BPE](figures/fig7_algorithms.png)

Unigram wins all four, by 1.25–2.19%. One pattern holds robustly: **the gap
widens as the vocabulary shrinks** — Hindi +1.31% → +2.03%, Nepali +1.25% →
+2.19%. With fewer slots, the ability to weigh alternative segmentations matters
more than replaying a fixed merge list, and it matters most exactly where the
vocabulary is tightest.

What does *not* hold is a clean split between the two languages. At 8,000 the
Nepali gap is the wider of the two (2.19% against 2.03%); at 16,000 they are
level inside the measurement's own noise (1.25% against 1.31%). An earlier run
showed Nepali ahead at both sizes, and retraining on a corrected sample reversed
the smaller difference — which is the honest reading of a 0.06 percentage-point
gap: it is not a finding.

Published comparisons put the two algorithms within about 1% of each other, so
the margin here is slightly wider and consistently in unigram's favour. Unigram
also keeps **subword regularization** available for Phase 2 — sampling
alternative segmentations as data augmentation, which matters most for the
low-resource model.

---

## Byte fallback, and the bug it fixed

A subword tokenizer has a fixed vocabulary, so it needs an answer for characters
it has never seen. There are two:

|  | An unknown character becomes | Cost | Consequence |
|---|---|---|---|
| `<unk>` (the default) | one unknown token | free | **the character is gone** |
| `byte_fallback` | its UTF-8 bytes, as `<0x00>`–`<0xFF>` | 256 slots | nothing is ever lost |

The first version of these tokenizers used the default, and the damage was not
the harmless tail one might assume. Because each tokenizer had been fitted on
text from a single source — the sampling defect described above, since fixed —
**each learned only the numeral system that source happened to use**:

```
Hindi   trained on text using ASCII digits       ->  ०१२३४५६७८९  ALL LOST
Nepali  trained on text using Devanagari digits  ->  ASCII 3,4,5,6,7,8,9  LOST
```

A Hindi date or price written in Devanagari numerals was destroyed outright:

```
in :  २०८१ सालमा ५०० जना
out:   ⁇  सालमा  ⁇  जना
```

Measured on 20,000 held-out documents per language with the pre-change
tokenizers: **0.2496%** of Hindi tokens and **0.2399%** of Nepali tokens were
`<unk>`, affecting 12.1% and 13.9% of documents.

**Fertility alone would never have revealed this** — it *rewards* the loss,
because discarded text costs no tokens. Catching it took two direct
measurements, now recorded on every training run in `tokenizer/fertility_*.json`:
the count of `<unk>` ids, and whether each document decodes back to its input.

|  | Before | After |
|---|---|---|
| Unknown-token rate | 0.2496% Hindi, 0.2399% Nepali | **0.0000%** |
| Held-out documents containing `<unk>` | 2,417 Hindi, 2,789 Nepali | **0** |
| Documents decoding back exactly | 87.9% Hindi, 86.1% Nepali | **20,000 of 20,000** |
| Cost in fertility | — | +0.6% Hindi, +0.4% Nepali |

Checked exhaustively as well as on held-out data: encoding **every document of
both final corpora** — 4.1M documents, 1.01 billion tokens — yields zero `<unk>`.

This matters beyond tidiness. A model cannot learn to read or produce text its
tokenizer cannot represent, and Phase 2's bits-per-byte comparison between Model
H and Model L is only meaningful if every byte of the evaluation text is
representable by both.

One transformation does remain: SentencePiece's own `nmt_nfkc` normalizer, which
collapses whitespace runs and folds compatibility characters — `…` becomes `...`
— in 2.7% of Hindi and 1.7% of Nepali held-out documents. That is declared
preprocessing, applied identically at training and inference, not lost data.

---

## Worked examples

Six sentences — three per language — each through all four of that language's
tokenizers, reporting every parameter the tokenizer is judged on: token count,
fertility, characters and bytes per token, byte-fallback pieces, unknown tokens,
and whether the text survives a decode.

The three per language are chosen to stress different things: ordinary news
prose, mixed script with numerals and an emoji, and a formal register heavy in
compounds. The same sentences run through *both* languages' models are in
[`tokenizer_examples.md`](tokenizer_examples.md).

Reproduce any of these:

```bash
python scripts/try_tokenizer.py --lang hindi --vocab all --algo all "<sentence>"
```

#### Hindi 1 — everyday news prose

> नेपाल और भारत के प्रधानमंत्री ने काठमांडू में अंतर्राष्ट्रीय सम्मेलन का उद्घाटन किया ।

13 words · 86 characters · 232 UTF-8 bytes

| Tokenizer | Tokens | Tokens/word | Chars/token | Bytes/token | Byte pieces | `<unk>` | Round-trip |
|---|---:|---:|---:|---:|---:|---:|---|
| **16k unigram** *(shipped)* | **14** | **1.077** | 6.143 | 16.571 | 0 | 0 | exact |
| 16k BPE | 17 | 1.308 | 5.059 | 13.647 | 0 | 0 | exact |
| 8k unigram | 18 | 1.385 | 4.778 | 12.889 | 0 | 0 | exact |
| 8k BPE | 18 | 1.385 | 4.778 | 12.889 | 0 | 0 | exact |

**16k unigram**

`␣नेपाल` `␣और` `␣भारत` `␣के` `␣प्रधानमंत्री` `␣ने` `␣काठमांडू` `␣में` `␣अंतर्राष्ट्रीय` `␣सम्मेलन` `␣का` `␣उद्घाटन` `␣किया` `␣।`

**16k BPE**

`␣नेपाल` `␣और` `␣भारत` `␣के` `␣प्रधानमंत्री` `␣ने` `␣काठ` `म` `ांड` `ू` `␣में` `␣अंतर्राष्ट्रीय` `␣सम्मेलन` `␣का` `␣उद्घाटन` `␣किया` `␣।`

**8k unigram**

`␣नेपाल` `␣और` `␣भारत` `␣के` `␣प्रधानमंत्री` `␣ने` `␣का` `ठ` `मा` `ंड` `ू` `␣में` `␣अंतर्राष्ट्रीय` `␣सम्मेलन` `␣का` `␣उद्घाटन` `␣किया` `␣।`

**8k BPE**

`␣नेपाल` `␣और` `␣भारत` `␣के` `␣प्रधानमंत्री` `␣ने` `␣का` `ठ` `म` `ांड` `ू` `␣में` `␣अंतर्राष्ट्रीय` `␣सम्मेलन` `␣का` `␣उद्घाटन` `␣किया` `␣।`

#### Hindi 2 — numerals, currency, English and an emoji

> सरकार ने २०८१ में ५,००० करोड़ रुपये का बजट पेश किया (Budget 2024) 🇮🇳

14 words · 68 characters · 154 UTF-8 bytes

| Tokenizer | Tokens | Tokens/word | Chars/token | Bytes/token | Byte pieces | `<unk>` | Round-trip |
|---|---:|---:|---:|---:|---:|---:|---|
| **16k unigram** *(shipped)* | **30** | **2.143** | 2.267 | 5.133 | 8 | 0 | exact |
| 16k BPE | 31 | 2.214 | 2.194 | 4.968 | 8 | 0 | exact |
| 8k unigram | 31 | 2.214 | 2.194 | 4.968 | 8 | 0 | exact |
| 8k BPE | 32 | 2.286 | 2.125 | 4.812 | 8 | 0 | exact |

**16k unigram**

`␣सरकार` `␣ने` `␣२०` `८` `१` `␣में` `␣५` `,` `०००` `␣करोड़` `␣रुपये` `␣का` `␣बजट` `␣पेश` `␣किया` `␣(` `B` `ud` `get` `␣2024` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xAE>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>`

**16k BPE**

`␣सरकार` `␣ने` `␣२०` `८` `१` `␣में` `␣५` `,` `०००` `␣करोड़` `␣रुपये` `␣का` `␣बजट` `␣पेश` `␣किया` `␣(` `B` `ud` `g` `et` `␣2024` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xAE>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>`

**8k unigram**

`␣सरकार` `␣ने` `␣२०` `८` `१` `␣में` `␣५` `,` `०००` `␣करोड़` `␣रुपये` `␣का` `␣बजट` `␣पेश` `␣किया` `␣(` `B` `ud` `g` `et` `␣2024` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xAE>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>`

**8k BPE**

`␣सरकार` `␣ने` `␣२०` `८` `१` `␣में` `␣५` `,` `००` `०` `␣करोड़` `␣रुपये` `␣का` `␣बजट` `␣पेश` `␣किया` `␣(` `B` `ud` `g` `et` `␣2024` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xAE>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>`

#### Hindi 3 — Sanskrit-derived compounds and conjuncts

> पुरातात्विक उत्खननों से प्राप्त साक्ष्यों के पुनर्स्थापन हेतु अंतर्राष्ट्रीय सहयोग अपरिहार्य है ।

12 words · 97 characters · 267 UTF-8 bytes

| Tokenizer | Tokens | Tokens/word | Chars/token | Bytes/token | Byte pieces | `<unk>` | Round-trip |
|---|---:|---:|---:|---:|---:|---:|---|
| **16k unigram** *(shipped)* | **22** | **1.833** | 4.409 | 12.136 | 0 | 0 | exact |
| 16k BPE | 23 | 1.917 | 4.217 | 11.609 | 0 | 0 | exact |
| 8k unigram | 25 | 2.083 | 3.880 | 10.680 | 0 | 0 | exact |
| 8k BPE | 26 | 2.167 | 3.731 | 10.269 | 0 | 0 | exact |

**16k unigram**

`␣पुरा` `ता` `त्व` `िक` `␣उत्खनन` `ों` `␣से` `␣प्राप्त` `␣साक्ष्य` `ों` `␣के` `␣पुनर्` `स्थापन` `␣हेतु` `␣अंतर्राष्ट्रीय` `␣सहयोग` `␣अपर` `ि` `हार` `्य` `␣है` `␣।`

**16k BPE**

`␣पुरा` `ता` `त्विक` `␣उत्` `खन` `नों` `␣से` `␣प्राप्त` `␣सा` `क्ष` `्यों` `␣के` `␣पुनर्` `स्थापन` `␣हेतु` `␣अंतर्राष्ट्रीय` `␣सहयोग` `␣अप` `रि` `हार` `्य` `␣है` `␣।`

**8k unigram**

`␣` `पुरा` `ता` `त्व` `िक` `␣उत्` `ख` `न` `नों` `␣से` `␣प्राप्त` `␣साक्ष्य` `ों` `␣के` `␣पुनर्` `स्थापन` `␣हेतु` `␣अंतर्राष्ट्रीय` `␣सहयोग` `␣अ` `परि` `हार` `्य` `␣है` `␣।`

**8k BPE**

`␣पु` `रा` `ता` `त्व` `िक` `␣उत्` `खन` `नों` `␣से` `␣प्राप्त` `␣सा` `क्ष` `्यों` `␣के` `␣पुनर्` `स्था` `पन` `␣हेतु` `␣अंतर्राष्ट्रीय` `␣सहयोग` `␣अप` `रि` `हार` `्य` `␣है` `␣।`

#### Nepali 1 — everyday news prose

> नेपाल र भारतका प्रधानमन्त्रीले काठमाडौंमा अन्तर्राष्ट्रिय सम्मेलनको उद्घाटन गर्नुभयो ।

9 words · 86 characters · 240 UTF-8 bytes

| Tokenizer | Tokens | Tokens/word | Chars/token | Bytes/token | Byte pieces | `<unk>` | Round-trip |
|---|---:|---:|---:|---:|---:|---:|---|
| **16k unigram** *(shipped)* | **11** | **1.222** | 7.818 | 21.818 | 0 | 0 | exact |
| 16k BPE | 10 | 1.111 | 8.600 | 24.000 | 0 | 0 | exact |
| 8k unigram | 11 | 1.222 | 7.818 | 21.818 | 0 | 0 | exact |
| 8k BPE | 11 | 1.222 | 7.818 | 21.818 | 0 | 0 | exact |

**16k unigram**

`␣नेपाल` `␣र` `␣भारतका` `␣प्रधानमन्त्रीले` `␣काठमाडौंमा` `␣अन्तर्राष्ट्रिय` `␣सम्मेलन` `को` `␣उद्घाटन` `␣गर्नुभयो` `␣।`

**16k BPE**

`␣नेपाल` `␣र` `␣भारतका` `␣प्रधानमन्त्रीले` `␣काठमाडौंमा` `␣अन्तर्राष्ट्रिय` `␣सम्मेलनको` `␣उद्घाटन` `␣गर्नुभयो` `␣।`

**8k unigram**

`␣नेपाल` `␣र` `␣भारतका` `␣प्रधानमन्त्रीले` `␣काठमाडौंमा` `␣अन्तर्राष्ट्रिय` `␣सम्मेलन` `को` `␣उद्घाटन` `␣गर्नुभयो` `␣।`

**8k BPE**

`␣नेपाल` `␣र` `␣भारतका` `␣प्रधानमन्त्रीले` `␣काठमाडौंमा` `␣अन्तर्राष्ट्रिय` `␣सम्मेल` `नको` `␣उद्घाटन` `␣गर्नुभयो` `␣।`

#### Nepali 2 — numerals, currency, English and an emoji

> सरकारले २०८१ मा ५,००० करोड रुपैयाँको बजेट पेस गर्‍यो (Budget 2024) 🇳🇵

13 words · 69 characters · 161 UTF-8 bytes

| Tokenizer | Tokens | Tokens/word | Chars/token | Bytes/token | Byte pieces | `<unk>` | Round-trip |
|---|---:|---:|---:|---:|---:|---:|---|
| **16k unigram** *(shipped)* | **29** | **2.231** | 2.379 | 5.552 | 9 | 0 | exact |
| 16k BPE | 28 | 2.154 | 2.464 | 5.750 | 9 | 0 | exact |
| 8k unigram | 31 | 2.385 | 2.226 | 5.194 | 9 | 0 | exact |
| 8k BPE | 31 | 2.385 | 2.226 | 5.194 | 9 | 0 | exact |

**16k unigram**

`␣सरकारले` `␣२०८१` `␣मा` `␣५` `,` `०००` `␣करोड` `␣रुपैयाँको` `␣बजेट` `␣पेस` `␣गर्‍यो` `␣(` `<0x42>` `ud` `g` `et` `␣20` `2` `4` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB5>`

**16k BPE**

`␣सरकारले` `␣२०८१` `␣मा` `␣५` `,` `०००` `␣करोड` `␣रुपैयाँको` `␣बजेट` `␣पेस` `␣गर्‍यो` `␣(` `<0x42>` `ud` `g` `et` `␣202` `4` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB5>`

**8k unigram**

`␣सरकारले` `␣२०८१` `␣मा` `␣५` `,` `०००` `␣करोड` `␣रुपैयाँ` `को` `␣बजेट` `␣पेस` `␣गर्‍यो` `␣(` `<0x42>` `u` `d` `g` `et` `␣20` `2` `4` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB5>`

**8k BPE**

`␣सरकारले` `␣२०८१` `␣मा` `␣५` `,` `०००` `␣करोड` `␣रुपैयाँको` `␣बजेट` `␣पेस` `␣गर्` `‍यो` `␣(` `<0x42>` `u` `d` `g` `et` `␣20` `2` `4` `)` `␣` `<0xF0>` `<0x9F>` `<0x87>` `<0xB3>` `<0xF0>` `<0x9F>` `<0x87>` `<0xB5>`

#### Nepali 3 — honorific verb morphology and agglutination

> विद्यालयका शिक्षकहरूले विद्यार्थीहरूलाई पढाइरहनुभएको थियो र अभिभावकहरूसँग छलफल गर्नुभयो ।

9 words · 89 characters · 249 UTF-8 bytes

| Tokenizer | Tokens | Tokens/word | Chars/token | Bytes/token | Byte pieces | `<unk>` | Round-trip |
|---|---:|---:|---:|---:|---:|---:|---|
| **16k unigram** *(shipped)* | **15** | **1.667** | 5.933 | 16.600 | 0 | 0 | exact |
| 16k BPE | 14 | 1.556 | 6.357 | 17.786 | 0 | 0 | exact |
| 8k unigram | 16 | 1.778 | 5.562 | 15.562 | 0 | 0 | exact |
| 8k BPE | 15 | 1.667 | 5.933 | 16.600 | 0 | 0 | exact |

**16k unigram**

`␣विद्यालयका` `␣शिक्षकहरू` `ले` `␣विद्यार्थीहरू` `लाई` `␣पढ` `ाइरहन` `ुभएको` `␣थियो` `␣र` `␣अभिभावक` `हरूसँग` `␣छलफल` `␣गर्नुभयो` `␣।`

**16k BPE**

`␣विद्यालयका` `␣शिक्षक` `हरूले` `␣विद्यार्थी` `हरूलाई` `␣पढाइ` `रहनुभएको` `␣थियो` `␣र` `␣अभिभावक` `हरूसँग` `␣छलफल` `␣गर्नुभयो` `␣।`

**8k unigram**

`␣विद्यालय` `का` `␣शिक्षक` `हरूले` `␣विद्यार्थीहरू` `लाई` `␣पढ` `ाइरहन` `ुभएको` `␣थियो` `␣र` `␣अभिभावक` `हरूसँग` `␣छलफल` `␣गर्नुभयो` `␣।`

**8k BPE**

`␣विद्यालयका` `␣शिक्षक` `हरूले` `␣विद्यार्थी` `हरूलाई` `␣पढाइ` `रह` `नुभएको` `␣थियो` `␣र` `␣अभिभावक` `हरूसँग` `␣छलफल` `␣गर्नुभयो` `␣।`

### What the six examples show

**Register moves fertility more than any design choice does.** The same shipped
Hindi tokenizer needs 1.077 tokens per word on news prose and 1.833 on
Sanskrit-derived compounds — a 70% swing. Every algorithm and vocabulary-size
difference in this report is under 3%. What a tokenizer costs depends far more
on what you feed it than on how it was built, which is worth remembering when
reading any single fertility number.

**Neither algorithm wins every sentence.** Unigram keeps काठमांडू whole where
16k BPE shatters it into four pieces (Hindi 1). BPE keeps सम्मेलनको whole where
unigram splits it (Nepali 1), and BPE edges Nepali 3 by one token. Averaged over
20,000 held-out documents unigram wins all four comparisons; on a single
sentence either can. The report's algorithm choice rests on the average, not on
cherry-picked sentences.

**Unigram's splits are morphologically principled.** In Nepali 3 it segments
`␣शिक्षकहरू` + `ले` and `␣विद्यार्थीहरू` + `लाई` — noun-plus-plural, then the
case marker on its own. That generalises to every other noun taking those
markers. This is what the global objective buys: not always fewer tokens, but
splits that correspond to something real in the language.

**Vocabulary size shows up exactly where you would expect.** काठमांडू survives at
16k and shatters at 8k. `पुनर्स्थापन` costs 2 pieces at 16k and 3 at 8k. The
8,000-piece models are never catastrophically worse — they simply spend more
tokens on the less frequent words, which is the trade-off the parameter budget
forced.

**Byte fallback appears only where it should.** Zero byte pieces in all four
pure-Devanagari sentences; 8–9 in the mixed-script ones, all from the flag emoji
and ASCII letters. Every one of the 24 tokenizations round-trips exactly — the
text that goes in is the text that comes back.

**Each tokenizer is specialised to its own language.** Three Nepali words through
the Hindi tokenizer:

| Word | Nepali tokenizer | Hindi tokenizer |
|---|---:|---:|
| `काठमाडौंमा` (in-Kathmandu) | **1 token** | 6 tokens |
| `प्रधानमन्त्रीले` (prime-minister-ERG) | **1 token** | 2 tokens |
| `गर्नुभयो` (did-HON) | **1 token** | 4 tokens |

A shared vocabulary would have handed one language a systematic advantage and
made any Model H versus Model L comparison meaningless.

---

## Verification

Everything below was run against the shipped corpus.

**Independent recount**, ignoring the ledger and recomputing from the split
files with the shipped tokenizer:

|  | Recomputed | Ledger |  |
|---|---:|---:|---|
| Hindi tokens | 503,783,232 | 503,783,232 | match |
| Hindi documents | 1,654,946 | 1,654,946 | match |
| Hindi characters | 2,013,671,207 | 2,013,671,207 | match |
| Hindi bytes | 5,087,159,391 | 5,087,159,391 | match |
| Nepali tokens | 505,739,399 | 505,739,399 | match |
| Nepali documents | 2,428,442 | 2,428,442 | match |
| Nepali characters | 2,381,634,424 | 2,381,634,424 | match |
| Nepali bytes | 6,344,866,742 | 6,344,866,742 | match |

**Everything else that was checked:**

| Check | Hindi | Nepali |
|---|---|---|
| At least 500M tokens | 503,783,232 | 505,739,399 |
| At least 20% manual | 20.0041% | 20.0007% |
| `<unk>` over the entire corpus | 0 | 0 |
| Duplicate `doc_id`s across splits | 0 | 0 |
| Wrong-language documents | 0 | 0 |
| train ∩ test / train ∩ val / val ∩ test | 0 / 0 / 0 | 0 / 0 / 0 |
| Documents shared with the other language's corpus | 0 | 0 |
| Sources declared but unused, or used but undeclared | none | none |
| Ledger regenerates byte-identically | yes | yes |
| All eight tokenizer models load at their stated size | yes | yes |
| Every figure has a title, both axis labels and a legend | 10 of 10 | 10 of 10 |
| Headline numbers match the generated files | pass | pass |
| Pipeline self-test (`tests/`) | pass | pass |

### The two vocabularies are not shared

Required explicitly by the brief, so it is measured rather than asserted:

|  | Result |
|---|---|
| Model files | two distinct files — sha256 `097893fe249ab05c…` (Hindi) against `17d2e824c44920d1…` (Nepali) |
| Pieces appearing in both vocabularies | 5,984 of 16,000 (37.4%) — *strings* both languages genuinely use |
| Pieces at the **same id** in both | 260 — the reserved and byte pieces, which are positional by construction |
| Hindi-only / Nepali-only pieces | 10,016 each |

A 37.4% string overlap is what two Devanagari languages sharing vocabulary looks
like, not what a shared tokenizer looks like. Outside the 259 reserved slots the
two models assign different ids to different pieces, and neither was initialised
from the other.

### Commands a marker can run

```bash
python scripts/verify_setup.py                      # environment, data, all 8 tokenizers
python scripts/show_stats.py --leakage              # every statistic, plus the leakage proof
python scripts/check_report_numbers.py              # this report against the generated files
PYTHONPATH=. python tests/test_pipeline_layout.py   # pipeline self-test, needs no corpus
PYTHONPATH=. python -m common.data.cross_dedup hindi:hi nepali:ne   # expect 0 shared
python scripts/try_tokenizer.py --lang both --vocab all --algo all "…"
python scripts/make_figures.py                      # regenerate all ten figures
```

---

## What went wrong, and how it was caught

Every one of these was found by measurement, not by inspection.

| # | Problem | How it surfaced | Fix |
|---|---|---|---|
| 1 | Two Hindi crawlers ran at once — 27% duplicate documents, one corrupted shard | scanned shards for repeated text hashes | repaired the shard; single-process rule |
| 2 | Crawler wandered from `bbc.com/hindi` into English BBC | measured Devanagari ratio per host | path-scoped seeds |
| 3 | `ne.wikisource.org` served English project text | mean Devanagari ratio of 0.01 for that host | removed from seeds |
| 4 | Nepali crawl ran dry at ~72M tokens | live host counter fell to 4 of 7 | widened seeds 7 → 23; throughput ~3M → ~13.6M tokens/hour |
| 5 | Downloader could not be paused safely | tested signal handling on a live process | flag-based signal handler |
| 6 | Pipeline needed ~30 GB RAM on an 8 GB machine | measured RSS over 120,000 documents | three passes with disk-backed intermediates; peak 1.23 GB |
| 7 | Token counts 6% (Hindi) and 26% (Nepali) optimistic | trained real tokenizers and measured | `--sp-model` for exact counting; both corpora re-sized |
| 8 | A 32k vocabulary would consume half a 25M model | computed the parameter budget explicitly | reduced to 16,000 |
| 9 | Nepali landed 19,686 tokens under the 20% quota | the ledger's own quota check failed | safety margin in the budget solver |
| 10 | 146 MB of crawler resume state would have been committed | inspected what `git add -A` would stage | corrected `.gitignore` |
| 11 | **Each tokenizer silently destroyed a numeral system** — 0.25% `<unk>`, 12–14% of documents | counted `<unk>` ids and decoded every held-out document back | enabled `byte_fallback`; rebuilt both corpora |
| 12 | Cache recorded *that* a tokenizer was used, not *which* | hit while fixing 11 — retraining overwrites the model in place | fingerprint the model file; `--recount-tokens` |
| 13 | Raw shards sorted by path put all downloaded documents before all manual ones | pipeline self-test asserted split order | sort by `(source, shard)` |
| 14 | 106 documents were byte-identical across the two corpora | hashed both corpora and intersected | removed from both sides |
| 15 | Every tokenizer was fitted on one source, and half the extract was discarded | checked which `source_id`s reached the training extract | stratified sampling; raised `max_sentence_length` |
| 16 | Embedding-budget percentages did not match their own arithmetic | recomputed while regenerating the figure | corrected in report and figure |

---

## Known limitations

**Machine-translated content cannot be ruled out.** Sangraha's explicitly
synthetic split was excluded by choice, but FineWeb-2, IndicCorp and Wikipedia
are web and community sources, and low-resource Wikipedias are known to contain
machine translation. There is no reliable detector at this scale. The manual
sources carry the lowest risk.

**OCR text is noisier than crawled text.** Devanagari OCR makes matra-reordering
errors. OCR is 0.11% of the Nepali corpus, so the effect is small, but real.

**English fragments occupy 7.2% of the Hindi vocabulary** and 3.06% of the
tokens it emits. Nepali is cleaner at 2.1% and 0.79%. Removing Latin entirely
would be wrong — real Hindi news contains English acronyms — but it is a genuine
impurity in a corpus described as monolingual.

**Hindi's manual share is entirely web crawl, with no OCR.** Nepali's mixes
crawl with OCR over 86 Devanagari PDFs; Hindi's crawler cleared the quota on its
own. The brief accepts either method, so this is a difference in register
coverage rather than a shortfall — Hindi's manual text is all contemporary web
prose, where Nepali's also contains edited book prose.

**The two languages used slightly different targets** — 505M and 507M, both
deliberately above the floor to absorb tokenizer drift.

**The tokenizers see a sample, not the corpus** — about 7% of Hindi and 6% of
Nepali. Representativeness is no longer the problem it was, but volume is a
deliberate choice: raising the cap costs training time and moves every token
count again, for a change the measured fertility says is small.

**Token counts depend on the tokenizer.** Every token figure here is measured
with the 16k unigram models. A different vocabulary would produce different
counts from identical text — which is the entire lesson of the vocabulary-size
section.

---

## Reproducing this

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/verify_setup.py

# 1. collect — manual
PYTHONPATH=. python -m common.collect.parallel_crawler --lang-dir nepali --lang ne \
    --source-id own_crawl_ne_news --seeds nepali/configs/seeds_ne.txt

# 2. collect — downloaded
PYTHONPATH=. python -m common.collect.hf_download --lang-dir nepali --lang ne

# 3. clean, deduplicate, budget, split
PYTHONPATH=. python -m common.data.pipeline nepali --lang ne --target-tokens 507000000

# 4. fit the tokenizer on the resulting train split
PYTHONPATH=. python -m common.tokenizer.train --lang-dir nepali --lang ne

# 5. re-run with exact counts, then retrain and recount
PYTHONPATH=. python -m common.data.pipeline nepali --lang ne \
    --sp-model nepali/tokenizer/sp_ne_16000_uni.model \
    --target-tokens 507000000 --reuse-clean --recount-tokens
PYTHONPATH=. python -m common.tokenizer.train --lang-dir nepali --lang ne
PYTHONPATH=. python -m common.data.stats nepali --lang ne \
    --sp-model nepali/tokenizer/sp_ne_16000_uni.model

# 6. token-frequency statistics and every figure
PYTHONPATH=. python -m common.data.token_stats nepali --lang ne \
    --sp-model nepali/tokenizer/sp_ne_16000_uni.model
python scripts/make_figures.py
```

Steps 4 and 5 are the tokenizer loop: the shipping tokenizer must be fitted on
the *final* train split, and the ledger recounted with it. It terminates because
the second retrain moves the ratio by about 0.1% — far too little to matter once
a margin is included.

---

## Requirements, clause by clause

### Language selection

| Clause | Where it is met |
|---|---|
| Model H: a non-English Indian language with abundant public text, briefly justified | Hindi — see *Choosing the two languages* |
| Model L: exactly one of Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, **Nepali**, Sindhi | Nepali |

### Dataset requirements

| Clause | Where it is met |
|---|---|
| Each model has its own monolingual corpus | separate trees, separate `sources.yaml`, separate statistics |
| ~500M training tokens per language, after tokenization | 503.78M and 505.74M, counted by encoding every document |
| At least 20% of final tokens manually collected | 20.0041% and 20.0007% |
| Manual means OCR, scraping or typed text you gather yourself | own crawler and own OCR — `common/collect/` |
| Report the manual versus downloaded token split | *Stage 3*, and `data/stats/corpus_ledger.md` |
| Do not share documents across corpora | 106 found and removed from both sides; now 0 |
| Do not concatenate languages | no code path reads both trees into one corpus |
| Separate train/validation/test splits per language | *Stage 4* |
| Separate statistics per language | `<lang>/data/stats/` |
| Collect sources | *Stage 1* |
| Remove invalid and duplicate data | *Stage 2* — nine filters |
| Normalize Unicode | NFC, before every other filter |
| Create splits | 96/2/2, document-level, seeded |
| Report corpus statistics — size, sources, cleaning steps, manual fraction | stages 1 to 3 |

### Tokenizer training

| Clause | Where it is met |
|---|---|
| A separate tokenizer trained from scratch per language | `hindi/tokenizer/`, `nepali/tokenizer/`, each fitted on its own train split |
| BPE, SentencePiece or Unigram | SentencePiece; both unigram and BPE trained and compared |
| Pretrained tokenizers not permitted | nothing downloaded or adapted; `spm.SentencePieceTrainer` on our own text |
| Vocabulary size in the tens of thousands | 16,000, with 8,000 measured as the alternative |
| Choose the size using fertility or unknown-token rate on held-out text | measured on the validation split, which no tokenizer saw |
| Vocabularies must not be shared | verified — see *Verification* |
| Report vocabulary size | *Stage 5* |
| Report token-frequency statistics | *Token-frequency statistics* |
| Report average characters per token | *Stage 5*, all eight tokenizers |
| Report tokenization examples | *Worked examples* |
| Report unknown-token statistics | zero, over every document of both corpora |

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts, both languages | `common/collect/` |
| 2 | Dataset preprocessing pipelines | `common/data/` |
| 3 | Per-language dataset statistics reports | `<lang>/data/stats/` |
| 4 | Per-language train/validation/test splits | `<lang>/data/splits/` (on Drive) |
| 5 | Tokenizer training code, both languages | `common/tokenizer/train.py` |
| 6 | Vocabulary files, one per language | `<lang>/tokenizer/sp_<code>_16000_uni.vocab`, plus three comparisons each |
| 7 | Tokenizer model files, one per language | `<lang>/tokenizer/sp_<code>_16000_uni.model`, plus three comparisons each |

---

*Every figure in this report is generated by `scripts/make_figures.py` from
`data/stats/*.json` and the tokenizer fertility files. No number here is
hand-entered, and `scripts/check_report_numbers.py` fails if any of them drifts
from the files on disk.*