# Citeworthy

AI Citation Optimizer — measure and improve how often **fool.com.au** content is
selected as a source by AI answer engines (Google AI Mode/Overviews, Gemini,
Perplexity, ChatGPT search) for high-value investing queries.

See [`CITEWORTHY_SPEC.md`](./CITEWORTHY_SPEC.md) for the full build specification.

## Status

Built in the milestone sequence from §14 of the spec. **Milestone 1 (scaffold)**
is complete: repo layout, config, data models, SQLite schema, and a CLI skeleton
with every command wired or stubbed. Ranking, tracking, and the edit loop land in
later milestones.

## Setup

```bash
uv sync                      # install dependencies
cp .env.example .env         # add ANTHROPIC_API_KEY (others optional)
uv run citeworthy init       # create config.yaml + citeworthy.db
```

## CLI

```
citeworthy init                          # create config.yaml, .env.example, db
citeworthy queries add|list|rm           # manage target query set
citeworthy rank <query_id> [--all]       # v1: build set, run tournament, report
citeworthy optimize <query_id>           # v2: edit loop on a ranked query
citeworthy track run [--engine ...]      # v3: one tracking pass (cron target)
citeworthy report <query_id> | --all     # markdown / portfolio report
citeworthy costs                         # token spend summary from logged calls
```

`init`, `queries`, and `costs` are functional now; `rank`, `optimize`,
`track run`, and `report` print which milestone implements them.

## Tests

```bash
uv run pytest
```

No live API calls in tests — fixtures live in `tests/fixtures/`.
