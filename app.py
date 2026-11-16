"""
app.py - Supply Chain Analytics Inventory Decision Agent Application

Interactive CLI Application powered by Rich.
Provides real-time decision analysis for:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock & Reorder Point (Continuous Review)
4. Fixed-Period Review (Periodic Review / Order-Up-To)
5. Newsvendor Single-Period Optimization

Architecture Pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

import sys
import argparse
from typing import Optional, Dict, Any

def is_streamlit_running() -> bool:
    """Detect whether script is running inside an active Streamlit runtime."""
    try:
        import streamlit.runtime
        return streamlit.runtime.exists()
    except Exception:
        return False

# Seamless bridge: if invoked via `streamlit run app.py` (e.g. on Streamlit Cloud)
if is_streamlit_running():
    import web_app
    web_app.run_app()


from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.prompt import Prompt

from inventory_agent import InventoryDecisionAgent, AgentResponse


console = Console()
agent = InventoryDecisionAgent()


# ---------------------------------------------------------------------------
# Pre-loaded Business Scenarios
# ---------------------------------------------------------------------------

SAMPLE_SCENARIOS = [
    {
        "id": "1",
        "title": "Automotive Parts Distribution (EOQ Analysis)",
        "question": (
            "We distribute automotive brake pads and sell 12,000 units annually. "
            "Placing an order costs $25 each time, and our inventory holding cost is $0.50 per unit per year. "
            "Our supplier has a 5-day lead time over 250 operating days. "
            "What is our optimal batch size and when should we reorder?"
        ),
        "context": None,
    },
    {
        "id": "2",
        "title": "Electronics Hardware Retail (Volume Quantity Discounts)",
        "question": (
            "Our annual demand for HDMI splitters is 1,000 units, and placing a purchase order costs $40. "
            "Our inventory carrying cost rate is 20% of unit price. "
            "The vendor offers a tiered price structure: "
            "1 to 499 units at $10.00 each, and 500 or more units at $8.00 each. "
            "Should we order larger batches to capture the discount tier?"
        ),
        "context": {
            "price_breaks": [
                {"min_qty": 0, "max_qty": 499, "unit_price": 10.0},
                {"min_qty": 500, "max_qty": None, "unit_price": 8.0},
            ]
        },
    },
    {
        "id": "3",
        "title": "Hospital Pharmaceutical Supply (Safety Stock & ROP under Joint Uncertainty)",
        "question": (
            "We manage critical surgical kits with an average daily demand of 50 units and standard deviation of 10 units. "
            "Supplier replenishment lead time averages 4 days with standard deviation of 1.5 days. "
            "Hospital protocol mandates a 98% Cycle Service Level. "
            "What safety stock buffer and reorder point (ROP) should we establish?"
        ),
        "context": None,
    },
    {
        "id": "4",
        "title": "Grocery FMCG Replenishment (Fixed-Period Review / Order-Up-To)",
        "question": (
            "Our regional distribution hub reviews bakery inventory every 7 days. "
            "Delivery lead time is 3 days. Mean daily demand is 20 units with std dev 5 units. "
            "We target a 95% service level. "
            "Right now we have 40 units in warehouse stock and 20 units in transit pipeline. "
            "What is our target Order-Up-To level S, and how many units should we order today?"
        ),
        "context": None,
    },
    {
        "id": "5",
        "title": "Fashion Apparel & Seasonal Goods (Newsvendor Optimization)",
        "question": (
            "We are preparing our winter designer coat collection. Each coat costs $40 to procure and retails for $100. "
            "Unsold coats at the end of winter are liquidated at an off-price salvage markdown of $10 each. "
            "Seasonal demand averages 500 coats with standard deviation of 100 coats. "
            "How many units should we purchase before the season begins?"
        ),
        "context": None,
    },
]


# ---------------------------------------------------------------------------
# UI Rendering Functions
# ---------------------------------------------------------------------------

def display_banner():
    banner_text = Text()
    banner_text.append("SUPPLY CHAIN ANALYTICS INVENTORY DECISION AGENT\n", style="bold cyan")
    banner_text.append("Autonomous Operations Research & Inventory Decision Support\n", style="white")
    banner_text.append("Pipeline: ", style="bold yellow")
    banner_text.append("Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain", style="italic green")

    console.print(Panel(banner_text, border_style="bright_blue", expand=False))


def display_pipeline_progress(stage_name: str, detail: str):
    console.print(f" [bold green]>[/bold green] [dim]{stage_name}:[/dim] [cyan]{detail}[/cyan]")


def render_agent_response(response: AgentResponse):
    console.print()
    # 1. Pipeline Execution Summary
    pipe_table = Table(title="Agent Pipeline Audit Trail", border_style="dim")
    pipe_table.add_column("Pipeline Stage", style="bold yellow", width=22)
    pipe_table.add_column("Agent Action & Outcome", style="white")

    pipe_table.add_row("1. Customer Question", f'"{response.question[:90]}..."' if len(response.question) > 90 else f'"{response.question}"')
    pipe_table.add_row("2. Model Identification", f"[bold green]{response.identified_model}[/bold green] (Confidence: {response.confidence * 100:.1f}%)")

    input_summary = ", ".join([f"{k}={v}" for k, v in response.extracted_inputs.items() if not isinstance(v, list)])
    pipe_table.add_row("3. Input Extraction", input_summary if input_summary else "Provided via Context")

    if response.missing_inputs:
        pipe_table.add_row("   Missing Inputs", f"[bold red]{', '.join(response.missing_inputs)}[/bold red]")

    pipe_table.add_row("4. Tool Selected", f"[magenta]{response.tool_selected or 'None'}[/magenta]")
    pipe_table.add_row("5. Python Calculation", "[green]Completed (Pure Python & Scipy)[/green]" if response.calculation_result else "[red]Pending Missing Parameters[/red]")

    verif_color = "green" if response.verification_passed else "bold red"
    pipe_table.add_row("6. Verification", f"[{verif_color}]{'Passed' if response.verification_passed else 'Flagged'}[/{verif_color}]")
    pipe_table.add_row("7. Explanation", "[cyan]Executive Briefing Synthesized[/cyan]")

    console.print(pipe_table)
    console.print()

    # 2. Executive Recommendation Callout Panel
    console.print(
        Panel(
            response.executive_summary,
            title="[bold green]Executive Supply Chain Recommendation[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # 3. Verification Details
    if response.verification_notes:
        verif_text = "\n".join([f"• {note}" for note in response.verification_notes])
        console.print(
            Panel(
                verif_text,
                title="[bold blue]Operational Verification & Sanity Audit[/bold blue]",
                border_style="blue",
                padding=(0, 2),
            )
        )

    # 4. Detailed Breakdown & Mathematical Steps
    console.print()
    console.print(Markdown(response.detailed_explanation))

    # 5. Step-by-Step Mathematical Trace
    if response.mathematical_trace:
        trace_table = Table(title="Deterministic Mathematical Calculation Trace", border_style="dim", show_lines=True)
        trace_table.add_column("Step #", justify="center", width=8, style="cyan")
        trace_table.add_column("Operations Research Formula & Substitution", style="white")

        for idx, step in enumerate(response.mathematical_trace, 1):
            trace_table.add_row(f"Step {idx}", step)

        console.print(trace_table)

    # 6. Strategic Insights
    if response.strategic_insights:
        insights_panel = "\n\n".join([f"• {insight}" for insight in response.strategic_insights])
        console.print()
        console.print(
            Panel(
                insights_panel,
                title="[bold yellow]Managerial Trade-Offs & Strategic Guidance[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )


def run_interactive_mode():
    """Main interactive terminal loop for natural-language queries."""
    display_banner()

    while True:
        console.print("\n[bold cyan]Choose an option:[/bold cyan]")
        console.print("  [bold 1-5][/bold 1-5] Run pre-loaded business scenarios (EOQ, Discount, ROP, Periodic, Newsvendor)")
        console.print("  [bold C][/bold C]   Enter a Custom natural-language inventory question")
        console.print("  [bold A][/bold A]   Run All scenarios in batch")
        console.print("  [bold Q][/bold Q]   Quit Application")

        choice = Prompt.ask("\nEnter selection", default="1").strip().upper()

        if choice == "Q":
            console.print("[dim]Exiting Inventory Decision Agent. Goodbye![/dim]")
            break

        elif choice in ["1", "2", "3", "4", "5"]:
            scenario = SAMPLE_SCENARIOS[int(choice) - 1]
            console.print(f"\n[bold yellow]=== Scenario {scenario['id']}: {scenario['title']} ===[/bold yellow]\n")
            console.print(f"[white]Question:[/white] {scenario['question']}\n")

            display_pipeline_progress("Stage 1", "Ingesting customer query...")
            display_pipeline_progress("Stage 2-7", "Executing reasoning and calculation pipeline...")

            response = agent.process_question(scenario["question"], context=scenario["context"])
            render_agent_response(response)

        elif choice == "A":
            for scenario in SAMPLE_SCENARIOS:
                console.print(f"\n[bold yellow]=== Scenario {scenario['id']}: {scenario['title']} ===[/bold yellow]\n")
                console.print(f"[white]Question:[/white] {scenario['question']}\n")
                response = agent.process_question(scenario["question"], context=scenario["context"])
                render_agent_response(response)
                console.print("\n" + "=" * 70 + "\n")

        elif choice == "C":
            console.print("\n[bold cyan]Enter your inventory question in plain English:[/bold cyan]")
            console.print("[dim]e.g. 'We sell 8,000 units per year. Placing an order costs $60 and holding cost is $3 per unit/year. What is our EOQ?'[/dim]\n")
            user_q = Prompt.ask("Question").strip()
            if not user_q:
                continue

            display_pipeline_progress("Stage 1", "Ingesting custom customer query...")
            display_pipeline_progress("Stage 2-7", "Processing through decision agent...")

            response = agent.process_question(user_q)
            render_agent_response(response)

        else:
            console.print("[red]Invalid selection. Please choose 1-5, C, A, or Q.[/red]")


def main():
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent CLI")
    parser.add_argument("--demo", action="store_true", help="Run all 5 scenario benchmarks automatically")
    parser.add_argument("--query", type=str, help="Submit a natural language inventory question directly")
    parser.add_argument("--web", action="store_true", help="Launch the interactive Streamlit web dashboard")
    args = parser.parse_args()

    if args.web:
        import subprocess
        console.print("[bold green]Starting Streamlit Interactive Web Application...[/bold green]")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    elif args.demo:
        display_banner()
        for scenario in SAMPLE_SCENARIOS:
            console.print(f"\n[bold yellow]=== Scenario {scenario['id']}: {scenario['title']} ===[/bold yellow]\n")
            console.print(f"[white]Question:[/white] {scenario['question']}\n")
            response = agent.process_question(scenario["question"], context=scenario["context"])
            render_agent_response(response)
            console.print("\n" + "=" * 70 + "\n")
    elif args.query:
        display_banner()
        response = agent.process_question(args.query)
        render_agent_response(response)
    else:
        run_interactive_mode()


if __name__ == "__main__":
    if not is_streamlit_running():
        main()
