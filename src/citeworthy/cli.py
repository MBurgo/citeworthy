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
) -> None:
    """v1: build the competitive set, run the tournament, and report."""
    _not_yet("rank", "Milestone 4")


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
    _not_yet("track run", "Milestone 5")


# --- report ----------------------------------------------------------------


@app.command()
def report(
    query_id: str = typer.Argument(None, help="Query slug (omit with --all)."),
    all: bool = typer.Option(False, "--all", help="Portfolio summary across queries."),
) -> None:
    """Render a markdown report for a query, or a portfolio summary."""
    _not_yet("report", "Milestone 4")


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
