# Phase 1 Report — Data Collection and Tokenizer Construction

*(Fill in once real data collection is complete — this is a template with
the required sections stubbed in.)*

## 1. Language Selection
- **Model H (higher-resource):** Hindi — justification: ...
- **Model L (lower-resource):** Nepali — justification: ...

## 2. Corpus Statistics
| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Total tokens | | |
| Manual-collection tokens | | |
| Manual fraction | | (must be ≥ 20%) |
| Public sources used | | |
| Manual sources used | | |
| Train / Val / Test sizes | | |

(Paste/embed the JSON from `hindi/data/stats/corpus_statistics.json` and
`nepali/data/stats/corpus_statistics.json` here, plus cleaning-step notes.)

## 3. Tokenizer Statistics
| | Model H | Model L |
|---|---|---|
| Vocab size | | |
| Avg chars/token | | |
| UNK rate | | |
| Example tokenization | | |

## 4. Shortfall Justification (Model L only, if applicable)
If Nepali did not reach ~500M tokens, explain the shortfall here with the
exact token count achieved.
