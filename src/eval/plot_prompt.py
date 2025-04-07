# src/visualizations/prompt_comparison.py
"""Create a bar chart comparing different prompting approaches."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from structlog import getLogger

logger = getLogger(__name__)


def create_prompt_comparison_chart(output_dir: Path = Path("writeup/figures")):
    """Create a bar chart comparing different prompting approaches for mnemonics.

    Args:
        output_dir: Directory to save the figure
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data structure: Each strategy has two values: one for deepseek-v3 and one for deepseek-r1
    # Create a DataFrame for prompt percentages
    prompt_percentages_df = pd.DataFrame(
        {
            "Prompting Strategy": [
                "Vanilla",
                "Vanilla-Alt",
                "Structured 0-shot",
                "Structured 10-shot",
            ],
            "Deepseek-v3 (%)": [20, 26, 34, 68],
            "Deepseek-r1 (%)": [34, 40, 52, 84],
        }
    )

    # plot dataframe column side by side without stacking
    prompt_percentages_df.plot(kind="bar", x="Prompting Strategy", figsize=(5, 4))

    # Set labels and title
    plt.xticks(rotation=45)
    plt.title("Prompting Strategy on No. LINKS (50 samples)")
    plt.xlabel("Prompting Strategy")
    plt.ylabel("LINKS (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the figure
    file_path = output_dir / "prompt_comparison.pdf"
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Saved prompt comparison chart to {file_path}")
    plt.close()

    return file_path


def plot_model_comparison_results(output_dir="writeup/figures"):
    """Plot comparison of Gemma 3 base vs GRPO evaluations by LLM and human annotators.

    Args:
        output_dir: Directory to save the figure
    """
    # Create output directory if it doesn't exist
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    # Updated data for wins, ties, losses (in percentages)
    annotator1 = {"wins": 70, "ties": 15, "losses": 15}  # LLM
    annotator2 = {"wins": 65, "ties": 25, "losses": 10}  # Human

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Set up positions for bars
    bar_width = 0.6
    positions = np.arange(2)

    # Plot stacked bars
    colors = ["#5cb85c", "#f0ad4e", "#d9534f"]  # green, yellow, red

    # Annotator 1 (LLM)
    ax.barh(
        positions[0],
        annotator1["wins"],
        height=bar_width,
        color=colors[0],
        label="LINKS wins",
    )
    ax.barh(
        positions[0],
        annotator1["ties"],
        height=bar_width,
        left=annotator1["wins"],
        color=colors[1],
        label="Ties",
    )
    ax.barh(
        positions[0],
        annotator1["losses"],
        height=bar_width,
        left=annotator1["wins"] + annotator1["ties"],
        color=colors[2],
        label="LINKS loses",
    )

    # Annotator 2 (Human)
    ax.barh(positions[1], annotator2["wins"], height=bar_width, color=colors[0])
    ax.barh(
        positions[1],
        annotator2["ties"],
        height=bar_width,
        left=annotator2["wins"],
        color=colors[1],
    )
    ax.barh(
        positions[1],
        annotator2["losses"],
        height=bar_width,
        left=annotator2["wins"] + annotator2["ties"],
        color=colors[2],
    )

    # Add percentage labels on bars
    def add_labels(pos, data):
        for i, value in enumerate([data["wins"], data["ties"], data["losses"]]):
            if value > 3:  # Only label if segment is wide enough
                start = (
                    0
                    if i == 0
                    else (data["wins"] if i == 1 else (data["wins"] + data["ties"]))
                )
                ax.text(
                    start + value / 2,
                    pos,
                    f"{value}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

    add_labels(positions[0], annotator1)
    add_labels(positions[1], annotator2)

    # Set chart properties
    ax.set_yticks(positions)
    ax.set_yticklabels(["Annotator 1 (LLM)", "Annotator 2 (Human)"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Evaluations (%)")
    ax.set_title("Gemma3 Base vs RL", fontsize=14, pad=10)

    # Add inter-rater reliability annotation
    ax.text(
        -0.1,
        -0.15,
        "Cohen's κ = 0.62",
        ha="center",
        transform=ax.transAxes,
        fontsize=10,
        style="italic",
    )

    # Add legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    # Add grid for readability
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)

    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the legend

    # Save figure
    output_path = Path(output_dir) / "model_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved model comparison chart to {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    create_prompt_comparison_chart()
    plot_model_comparison_results()
