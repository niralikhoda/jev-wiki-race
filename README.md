# Jev Wiki Race

One Wikipedia race, three decision APIs, side by side.

Start on one article, reach another by clicking links only. The same start, the same
destination and the same rules go to three navigators: TypeSafe's **Jev**, **Claude**
and **GPT**. You watch each one choose its links live, with the probabilities it
assigned, the time it took and what it cost.

This is a Python port of [davext/classifier-wiki-race](https://github.com/davext/classifier-wiki-race),
extended from one model to a three-way comparison.

## Why this exists

Jev is a decision model. It never writes text. You give it a state and typed
questions, and it returns a probability for every allowed answer. Claude and GPT are
chat models that can be asked to do the same job through structured output.

A Wikipedia race is a clean way to compare them, because every hop is one small
decision with a fixed answer space: which of these links is closer to the destination?

The interesting difference is not who wins. It is where each model's confidence comes
from.

| Navigator | How it answers | Where confidence comes from |
| --- | --- | --- |
| Jev | Two `Choice` questions in one request | Computed from the probability distribution the model returns |
| Claude | One structured-output call | A number the model writes about itself. The API exposes no probabilities |
| GPT | One JSON-schema call with logprobs on | Measured from token logprobs, with the self-reported number kept alongside |

## What happens in a race

1. **Destination preload.** Fetch the destination article to learn its canonical
   title, and keep the first 320 characters of its intro as a summary.
2. **Current article.** Fetch the article as rendered HTML, strip infoboxes,
   navigation boxes, tables and references, and collect every body link.
3. **Candidate filter.** Plain string matching, no model. Links whose text matches a
   keyword from the destination go first, the rest follow in page order, cut at 28.
4. **Decision.** The navigator receives the current page, the destination and its
   summary, the last twelve pages visited and the 28 candidates. It answers two
   questions: which operation (CLICK, DONE, BLOCKED) and which link.
5. **Move.** Fetch the chosen article. If it is the destination, the race is won.
   If it loops back to a visited page, try again. Otherwise repeat from step 2.
6. **Fallbacks.** If the model declines but still gives some link 15 percent or more,
   follow it. Otherwise backtrack one page. Stop after 25 decisions.

The model never reads the article body. It chooses among link titles, using the
destination summary as its yardstick.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/niralikhoda/jev-wiki-race.git
cd jev-wiki-race
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and add the keys for the lanes you want to run. Any lane without a key
will report an error and leave the others untouched.

```text
TYPESAFE_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

Keys: [TypeSafe](https://console.typesafe.ai/keys),
[Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/api-keys).
`.env` is gitignored. Keys are read by the local Python process only and never reach
the browser.

## Run the page

```bash
.venv/bin/python web/server.py
```

Open http://127.0.0.1:8765.

- Type a start and a destination, or press **Random pair** for a connected pair.
- Each lane has a model picker and its own **Start** button. Races run one at a time
  so Wikipedia never sees parallel bursts.
- Click a lane to watch it. The stage renders the Wikipedia article the lane is on,
  tints the links it was offered, marks the one it clicked, and shows the operation
  and link probabilities plus the exact state JSON sent to the model.
- **Watch delay** pauses after each decision so you can follow along. Set it to 0 for
  a timed run.
- The table at the bottom fills in as lanes finish, with a glossary underneath.

Set `PORT` to use a different port.

## Run from the command line

```bash
.venv/bin/python race.py --start "Chess" --dest "Sabarmati Ashram" --navigator jev claude gpt
```

```bash
.venv/bin/python race.py --random --navigator jev claude --claude-model claude-haiku-4-5
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--start`, `--dest` | Article titles or Wikipedia URLs | required unless `--random` |
| `--random` | Pick a connected random pair | off |
| `--navigator` | One or more of `jev`, `claude`, `gpt` | `jev` |
| `--claude-model` | Any Claude model id | `claude-opus-5` |
| `--claude-effort` | `low`, `medium`, `high` | `low` |
| `--gpt-model` | Any GPT model id | `gpt-4.1-mini` |
| `--max-steps` | Decision cap per race | 25 |
| `--no-save` | Skip the JSON record in `results/` | off |

## What we measured

Chess to Sabarmati Ashram, 26 September 2026. All models took the same route: Chess,
History of India, Mahatma Gandhi, Sabarmati Ashram.

| Model | Hops | Summed API time | Cost |
| --- | --- | --- | --- |
| jev-latest | 3 | 1.6 s | $0.00044 |
| gpt-4.1-mini | 3 | 4.5 s | $0.0032 |
| claude-haiku-4-5 | 3 | 4.6 s | $0.0096 |
| claude-sonnet-5 | 3 | 7.8 s | $0.010 |
| claude-opus-5 | 3 | 7.5 s | $0.065 |

Compare on API time. Wall time includes Wikipedia fetches and rate-limit back-off.
Costs are computed from list prices at the time and will drift.

This is one route, not a benchmark. On an easy route every model agrees. The runs
worth keeping are the ones where they do not.

## Two findings worth knowing

**Logprobs collapse under a JSON schema.** GPT's measured confidence came out at 1.00
on every hop while the model itself reported 0.90 to 0.95. That is not certainty. With
a schema enforced, decoding is constrained, the answer tokens are near-deterministic,
and their logprobs go to zero. The common advice to read logprobs instead of asking
the model stops working once structured outputs are on. GPT-5 models refuse logprobs
entirely, so they fall back to self-reported confidence.

**The three confidence numbers are not comparable.** One is a distribution statistic,
one is self-reported, one is a collapsed logprob. The page shows them per lane and
says so in its glossary. Do not rank models by that column.

## Limits

- The keyword filter in step 3 does real work. It pushes relevant links to the top
  before any model looks. A stricter test would send 28 links in page order.
- Wikipedia rate-limits bursts with HTTP 429. The client spaces requests and backs
  off, which can add tens of seconds of wall time.
- English Wikipedia only.
- Jev cannot explain a choice, reason across steps, or do arithmetic. The race hides
  that, because each hop is a single comparison over names.

## Project layout

```text
wiki_race/
  wikipedia.py    MediaWiki client, link extraction, random pairs, 429 back-off
  candidates.py   keyword scoring and the 28-link cap
  prompts.py      rules and state shared by every navigator
  navigators.py   Jev, Claude and GPT behind one interface
  runner.py       the race loop, backtracking, event hook
race.py           command line runner and comparison table
web/server.py     local server, one race at a time, keys stay server-side
web/index.html    the page
env_loader.py     dependency-free .env loader
```

Adding a fourth model means one new class in `navigators.py` with a `decide` method
that returns a `Decision`. Nothing else changes.

## Credits

The race idea, the rules text and the loop design come from
[davext/classifier-wiki-race](https://github.com/davext/classifier-wiki-race) by
David Harvey, MIT licensed. Its copyright and licence notice are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Jev is a model by [TypeSafe AI](https://typesafe.ai). Docs: https://docs.typesafe.ai
