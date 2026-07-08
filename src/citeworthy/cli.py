"""Citeworthy CLI (typer entrypoint).

Command surface per §7 of CITEWORTHY_SPEC.md:

    citeworthy init
    citeworthy queries add|list|rm
    citeworthy rank <query_id> [--all]
    citeworthy optimize <query_id>
    citeworthy track run [--engine ...]
    citeworthy report <query_id> | --all
    citeworthy costs

Milestone 1 wires `init`, `queries`, and `costs` to real (config/db) plumbing;
`rank`, `optimize`, `track`, and `report` are stubs that explain which milestone
implements them.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import db
from .config import DEFAULT_CONFIG_PATH, load_config

app = typer.Typer(
    name="citeworthy",
    help="AI Citation Optimizer for fool.com.au.",
    no_args_is_help=True,
    add_completion=False,
)
queries_app = typer.Typer(help="Manage the target query set.", no_args_is_help=True)
track_app = typer.Typer(help="Ground-truth citation tracker (v3).", no_args_is_help=True)
app.add_typer(queries_app, name="queries")
app.add_typer(track_app, name="track")

console = Console()

# Load .env once at import so api_key() lookups work across commands (§9).
load_dotenv()


def _not_yet(command: str, milestone: str) -> None:
    console.print(
        f"[yellow]`{command}` is not implemented yet.[/yellow] "
        f"It lands with [bold]{milestone}[/bold] (see §14 of CITEWORTHY_SPEC.md)."
    )
    raise typer.Exit(code=2)


# --- init ------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config.yaml."),
) -> None:
    """Create config.yaml, .env.example, and the SQLite database."""
    # config.yaml
    if DEFAULT_CONFIG_PATH.exists() and not force:
        console.print(f"[dim]config.yaml already exists — leaving it in place.[/dim]")
    else:
        # Persist the validated defaults so the file always round-trips.
        console.print(f"[green]config.yaml ready.[/green]")

    # .env.example
    env_example = Path(".env.example")
    if not env_example.exists():
        env_example.write_text(
            "ANTHROPIC_API_KEY=\nSERPAPI_API_KEY=\n"
            "PERPLEXITY_API_KEY=\nGEMINI_API_KEY=\n",
            encoding="utf-8",
        )
        console.print("[green].env.example created.[/green]")
    else:
        console.print("[dim].env.example already exists.[/dim]")

    # db
    path = db.init_db()
    console.print(f"[green]Database initialised at {path}.[/green]")

    if not Path(".env").exists():
        console.print(
            "[yellow]No .env found.[/yellow] Copy .env.example to .env and add "
            "your ANTHROPIC_API_KEY to run `rank`."
        )


# --- queries ---------------------------------------------------------------


@queries_app.command("add")
def queries_add(
    id: str = typer.Argument(..., help="Slug, e.g. best-asx-dividend-shares."),
    text: str = typer.Option(..., "--text", help="The query text."),
    tier: str = typer.Option("money", "--tier", help="money | supporting."),
    our_url: str | None = typer.Option(None, "--our-url", help="Page we're optimising."),
) -> None:
    """Add or update a target query."""
    from .models import TargetQuery

    q = TargetQuery(id=id, text=text, tier=tier, our_url=our_url)  # type: ignore[arg-type]
    conn = db.connect()
    try:
        db.upsert_query(conn, q)
    finally:
        conn.close()
    console.print(f"[green]Saved query[/green] [bold]{q.id}[/bold] ({q.tier}).")


@queries_app.command("list")
def queries_list() -> None:
    """List all target queries."""
    conn = db.connect()
    try:
        rows = db.list_queries(conn)
    finally:
        conn.close()

    if not rows:
        console.print("[dim]No queries yet. Add one with `citeworthy queries add`.[/dim]")
        return

    table = Table(title="Target queries")
    table.add_column("id", style="bold")
    table.add_column("text")
    table.add_column("tier")
    table.add_column("our_url")
    for q in rows:
        table.add_row(q.id, q.text, q.tier, str(q.our_url) if q.our_url else "—")
    console.print(table)


@queries_app.command("rm")
def queries_rm(
    id: str = typer.Argument(..., help="Query slug to remove."),
) -> None:
    """Remove a target query."""
    conn = db.connect()
    try:
        removed = db.remove_query(conn, id)
    finally:
        conn.close()
    if removed:
        console.print(f"[green]Removed[/green] {id}.")
    else:
        console.print(f"[yellow]No query with id[/yellow] {id}.")
        raise typer.Exit(code=1)


# --- rank (v1) -------------------------------------------------------------


@app.command()
def rank(
    query_id: str = typer.Argument(None, help="Query slug (omit with --all)."),
    all: bool = typer.Option(False, "--all", help="Rank every query."),
    no_robots: bool = typer.Option(
        False, "--no-robots", help="Skip robots.txt checks (use only on sites you own)."
    ),
) -> None:
    """v1: build the competitive set, run the tournament, and report."""
    import uuid
    from datetime import datetime, timezone

    import httpx

    from .config import api_key
    from .extract import DEFAULT_UA
    from .models import Run
    from .pipeline import rank_query
    from .ranker.judge import AnthropicTransport, Judge
    from .ranker.tournament import BudgetExceeded
    from .serp.serpapi import SerpApiProvider

    config = load_config()

    conn = db.connect()
    try:
        if all:
            queries = db.list_queries(conn)
        elif query_id:
            q = db.get_query(conn, query_id)
            queries = [q] if q else []
            if not q:
                console.print(f"[red]No query with id[/red] {query_id}.")
                raise typer.Exit(code=1)
        else:
            console.print("[red]Provide a query id or --all.[/red]")
            raise typer.Exit(code=1)

        if not queries:
            console.print("[yellow]No queries to rank.[/yellow]")
            raise typer.Exit(code=1)

        anthropic_key = api_key("ANTHROPIC_API_KEY")
        if not anthropic_key:
            console.print("[red]ANTHROPIC_API_KEY is not set.[/red] The judge cannot run.")
            raise typer.Exit(code=1)
        serp_key = api_key("SERPAPI_API_KEY")
        if config.serp.provider == "serpapi" and not serp_key:
            console.print("[red]SERPAPI_API_KEY is not set.[/red] Cannot build the SERP set.")
            raise typer.Exit(code=1)

        provider = SerpApiProvider(serp_key or "")
        judge = Judge(
            AnthropicTransport(anthropic_key),
            model=config.judge.model,
            temperature=config.judge.temperature,
            show_domains=config.judge.show_domains,
        )
        client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": DEFAULT_UA},
            follow_redirects=True,
        )

        run_id = uuid.uuid4().hex[:12]
        db.create_run(
            conn,
            Run(
                run_id=run_id,
                kind="rank",
                started_at=datetime.now(timezone.utc),
                config_hash=config.config_hash(),
                git_sha=_git_sha(),
            ),
        )

        try:
            for q in queries:
                console.print(f"\n[bold]Ranking[/bold] {q.id} — “{q.text}”…")
                try:
                    out = rank_query(
                        q, config,
                        provider=provider, judge=judge, client=client,
                        run_id=run_id, conn=conn, respect_robots=not no_robots,
                    )
                except BudgetExceeded as exc:
                    console.print(f"[red]Budget cap hit:[/red] {exc}")
                    raise typer.Exit(code=3)
                _print_rank_summary(out)
        finally:
            client.close()
    finally:
        conn.close()


def _print_rank_summary(out) -> None:
    """Concise rich terminal summary of a completed rank (§3 reporting)."""
    cs = out.candidate_set
    for w in cs.warnings:
        console.print(f"  [yellow]![/yellow] {w}")

    table = Table(title=f"{out.query.id} — top passages")
    table.add_column("rank", justify="right")
    table.add_column("domain")
    table.add_column("ours", justify="center")
    table.add_column("sel. prob", justify="right")
    table.add_column("95% CI", justify="right")
    by_rank = sorted(out.rank_results, key=lambda r: r.rank)
    passages = {p.id: p for p in cs.passages}
    for r in by_rank[:8]:
        p = passages[r.passage_id]
        table.add_row(
            str(r.rank),
            p.domain,
            "✓" if p.is_ours else "",
            f"{r.selection_prob * 100:.1f}%",
            f"{r.ci_low * 100:.1f}–{r.ci_high * 100:.1f}%",
        )
    console.print(table)

    dr = out.tournament.disagreement_rate
    warn = "  [yellow]⚠ weak judge signal[/yellow]" if out.tournament.weak_signal else ""
    console.print(
        f"  {out.tournament.n_calls} judge calls · "
        f"${out.total_cost_usd:.4f} · disagreement {dr * 100:.0f}%{warn}"
    )
    if out.report_path:
        console.print(f"  [green]Report:[/green] {out.report_path}")


def _git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - git may be absent; run is still valid
        return None


# --- optimize (v2) ---------------------------------------------------------


@app.command()
def optimize(
    query_id: str = typer.Argument(..., help="Query slug to optimise."),
) -> None:
    """v2: run the edit loop (hill-climb with a compliance gate) on a ranked query."""
    _not_yet("optimize", "Milestone 6")


# --- track (v3) ------------------------------------------------------------


@track_app.command("run")
def track_run(
    engine: list[str] = typer.Option(
        None, "--engine", help="Limit to specific engines (perplexity, gemini_grounded)."
    ),
) -> None:
    """v3: one ground-truth tracking pass (cron target)."""
    import uuid
    from datetime import datetime, timezone

    from .config import api_key
    from .models import Run
    from .tracker import TrackBudgetExceeded, run_track
    from .truth.gemini import GeminiClient
    from .truth.perplexity import PerplexityClient

    config = load_config()
    conn = db.connect()
    try:
        queries = db.list_queries(conn)
        if not queries:
            console.print("[yellow]No queries to track. Add some with "
                          "`citeworthy queries add`.[/yellow]")
            raise typer.Exit(code=1)

        wanted = engine or config.truth.engines
        clients = []
        for name in wanted:
            if name == "perplexity":
                key = api_key("PERPLEXITY_API_KEY")
                if not key:
                    console.print("[yellow]Skipping perplexity: PERPLEXITY_API_KEY not set.[/yellow]")
                    continue
                clients.append(PerplexityClient(key, model=config.truth.perplexity_model))
            elif name == "gemini_grounded":
                key = api_key("GEMINI_API_KEY")
                if not key:
                    console.print("[yellow]Skipping gemini_grounded: GEMINI_API_KEY not set.[/yellow]")
                    continue
                clients.append(GeminiClient(key, model=config.truth.gemini_model))
            else:
                console.print(f"[yellow]Unknown engine '{name}' — skipping.[/yellow]")

        if not clients:
            console.print("[red]No engines available (missing API keys).[/red]")
            raise typer.Exit(code=1)

        run_id = uuid.uuid4().hex[:12]
        db.create_run(
            conn,
            Run(
                run_id=run_id,
                kind="track",
                started_at=datetime.now(timezone.utc),
                config_hash=config.config_hash(),
                git_sha=_git_sha(),
            ),
        )
        console.print(
            f"[bold]Tracking[/bold] {len(queries)} queries × "
            f"{len(clients)} engine(s) × {config.truth.samples_per_engine} samples…"
        )
        try:
            result = run_track(
                queries, clients, config, run_id=run_id, conn=conn,
                budget_cap=config.budget.max_usd_per_track_run,
            )
        except TrackBudgetExceeded as exc:
            console.print(f"[red]Budget cap hit:[/red] {exc}")
            raise typer.Exit(code=3)

        _print_track_summary(result, config.our_domain)
    finally:
        conn.close()


def _print_track_summary(result, our_domain: str) -> None:
    """Per-run citation-share summary (a single run is noise — §6.6)."""
    for w in result.warnings:
        console.print(f"  [yellow]{w}[/yellow]")

    for s in result.stats:
        if s.n_samples == 0:
            console.print(f"[dim]{s.query_id} · {s.engine}: no samples collected.[/dim]")
            continue
        ours = s.our_share
        console.print(
            f"\n[bold]{s.query_id}[/bold] · {s.engine} "
            f"({s.n_samples} samples)"
        )
        console.print(
            f"  our share (`{our_domain}`): {ours.share * 100:.0f}% "
            f"(Wilson {ours.ci_low * 100:.0f}–{ours.ci_high * 100:.0f}%)"
        )
        dominators = ", ".join(
            f"{d.domain} {d.share * 100:.0f}%" for d in s.top_domains[:3]
        )
        if dominators:
            console.print(f"  dominating: {dominators}")

    console.print(
        f"\n[dim]{result.n_samples} samples · ${result.total_cost_usd:.4f} · "
        f"{result.n_failures} failures. A single run is noise — trends need "
        f"3+ weekly runs with non-overlapping Wilson intervals (§6.6).[/dim]"
    )


# --- report ----------------------------------------------------------------


@app.command()
def report(
    query_id: str = typer.Argument(None, help="Query slug (omit with --all)."),
    all: bool = typer.Option(False, "--all", help="Portfolio summary across queries."),
    show: bool = typer.Option(False, "--show", help="Print the report to the terminal."),
) -> None:
    """Render a markdown report for a query's most recent rank run."""
    from . import report as report_mod

    if all:
        _not_yet("report --all", "Milestone 7")

    if not query_id:
        console.print("[red]Provide a query id or --all.[/red]")
        raise typer.Exit(code=1)

    conn = db.connect()
    try:
        try:
            md, path = report_mod.render_query_report(
                conn, query_id, our_domain=load_config().our_domain
            )
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
    finally:
        conn.close()

    console.print(f"[green]Report written:[/green] {path}")
    if show:
        console.print(md)


# --- costs -----------------------------------------------------------------


@app.command()
def costs() -> None:
    """Token spend summary from logged LLM calls."""
    conn = db.connect()
    try:
        summary = db.cost_summary(conn)
    finally:
        conn.close()

    if not summary:
        console.print("[dim]No LLM calls logged yet.[/dim]")
        return

    table = Table(title="Token spend")
    table.add_column("model", style="bold")
    table.add_column("calls", justify="right")
    table.add_column("input tok", justify="right")
    table.add_column("output tok", justify="right")
    table.add_column("est. USD", justify="right")
    total = 0.0
    for row in summary:
        total += row["est_usd"] or 0.0
        table.add_row(
            row["model"],
            str(row["calls"]),
            f"{row['input_tokens']:,}",
            f"{row['output_tokens']:,}",
            f"${row['est_usd']:.4f}",
        )
    console.print(table)
    console.print(f"[bold]Total: ${total:.4f}[/bold]")


# Touch load_config so a bad config.yaml fails fast at startup via callback.
@app.callback()
def _main(ctx: typer.Context) -> None:  # pragma: no cover - thin wiring
    ctx.obj = load_config()


if __name__ == "__main__":  # pragma: no cover
    app()
