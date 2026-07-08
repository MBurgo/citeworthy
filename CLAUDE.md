# Citeworthy
Internal tool: measures and improves fool.com.au citation share in AI answers.
- Python 3.11+, uv, pydantic v2, typer, SQLite. Run tests with `uv run pytest`.
- NEVER make live API calls in tests; use fixtures in tests/fixtures/.
- All LLM calls go through src/citeworthy/ranker/judge.py or editor/ — never
  inline API calls elsewhere. Every call is logged to the db with token counts.
- Respect the budget caps in config.yaml; abort, don't overspend.
- The judge prompt JSON contract in §6.3 of CITEWORTHY_SPEC.md is load-bearing;
  changing fields requires updating models.py, db.py, and diagnose.py together.
- Compliance gate (editor/compliance.py) must run on every generated variant.
  No exceptions, including in tests of the loop (use a stub checker).
- Verify current model names/pricing at https://docs.claude.com before
  hardcoding; same for Perplexity and Gemini response schemas.

## Build status
Built per the milestone sequence in §14 of CITEWORTHY_SPEC.md.
- [x] Milestone 1: Scaffold repo, config, models, db, CLI skeleton (all commands stubbed).
- [ ] Milestone 2: Competitive set builder + chunker, with fixtures and tests.
- [ ] Milestone 3: Judge client + tournament + Bradley–Terry + bootstrap.
- [ ] Milestone 4: `rank` end-to-end + report. **Human review checkpoint.**
- [ ] Milestone 5: Tracker (`truth/` + `track run` + share stats).
- [ ] Milestone 6: Edit loop + compliance gate.
- [ ] Milestone 7: Portfolio report + judge-vs-reality correlation.

Modules under serp/, ranker/, editor/, truth/, extract.py, and report.py are
scaffolded stubs — they define the interfaces but raise NotImplementedError
until their milestone lands.
