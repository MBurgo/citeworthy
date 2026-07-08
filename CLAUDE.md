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
- [x] Milestone 2: Competitive set builder + chunker, with fixtures and tests.
- [x] Milestone 3: Judge client + tournament + Bradley–Terry + bootstrap.
- [x] Milestone 4: `rank` end-to-end + report. **Human review checkpoint (awaiting a real run).**
- [x] Milestone 5: Tracker (`truth/` + `track run` + share stats).
- [x] Milestone 6: Edit loop + compliance gate.
- [ ] Milestone 7: Portfolio report + judge-vs-reality correlation.

serp/serpapi.py and extract.py are implemented as of Milestone 2; ranker/judge.py,
tournament.py, and bradley_terry.py as of Milestone 3; pipeline.py, report.py,
ranker/diagnose.py, and the `rank`/`report` CLI commands as of Milestone 4;
truth/ (perplexity, gemini, share), tracker.py, and the `track run` command as of
Milestone 5; editor/ (variants, compliance, loop) and the `optimize` command as
of Milestone 6. Only report.py's `--all` portfolio summary + judge-vs-reality
correlation remain (Milestone 7).
