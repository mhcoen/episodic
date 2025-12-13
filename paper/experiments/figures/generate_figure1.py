"""
Generate figure demonstrating granularity mismatch in topic segmentation.

Key argument: The model correctly detects fine-grained subtopic shifts,
but gold annotations only mark coarse boundaries. This is NOT boundary jitter—
every predicted boundary corresponds to a real semantic transition.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# Dialogue with clear subtopic evolution
# Each predicted boundary marks a REAL semantic shift
DIALOGUE = [
    ("U", "I need a restaurant recommendation for tonight"),
    ("A", "Sure! What cuisine are you in the mood for?"),
    ("U", "Italian sounds good"),
    # ─── Shift: preference → dietary constraint ───
    ("A", "Great choice. Any dietary restrictions I should know about?"),
    ("U", "Yes, one person is gluten-free"),
    # ─── Shift: constraint → specific option ───
    ("A", "Trattoria Milano has excellent gluten-free pasta options"),
    ("U", "How's the atmosphere there?"),
    ("A", "It's upscale casual, good for dates or small groups"),
    # ─── Shift: evaluation → alternative exploration ───
    ("U", "Actually, is there something more casual?"),
    ("A", "Pasta House is more relaxed, also has GF options"),
    # ─── Shift: alternative → logistics/booking ───
    ("U", "That sounds perfect. Can you help me book it?"),
    ("A", "Of course. What time and how many people?"),
]

# Gold: only marks the ONE major conceptual shift (browsing → booking)
GOLD_BOUNDARIES = [10]  # Only when user decides to book

# Predicted: marks each meaningful subtopic transition
PRED_BOUNDARIES = [3, 5, 8, 10]
# 3: cuisine → dietary constraints
# 5: constraints → evaluating specific option
# 8: first option → exploring alternative
# 10: exploration → booking/logistics

# Subtopic labels (for annotation, not shown in figure)
SUBTOPICS = [
    (0, 2, "Initial request"),
    (3, 4, "Dietary constraints"),
    (5, 7, "Option evaluation"),
    (8, 9, "Alternative"),
    (10, 11, "Booking"),
]


def create_figure():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Layout
    y_start = 0.92
    y_step = 0.068
    left_margin = 0.08
    text_start = 0.14
    text_width = 0.75

    y = y_start

    for i, (role, text) in enumerate(DIALOGUE):
        # Role styling
        role_color = '#1565C0' if role == "U" else '#666666'
        role_label = "User:" if role == "U" else "Asst:"

        # Turn number
        ax.text(left_margin - 0.04, y, f"{i}", fontsize=9, color='#BBBBBB',
                va='center', ha='right', fontfamily='monospace')

        # Role
        ax.text(left_margin, y, role_label, fontsize=10, fontweight='bold',
                color=role_color, va='center', ha='left')

        # Text
        ax.text(text_start, y, text, fontsize=10.5, color='#333333',
                va='center', ha='left')

        # Draw boundary markers BEFORE this turn
        if i in GOLD_BOUNDARIES:
            # Gold boundary - thick solid line (darker green)
            ax.plot([left_margin - 0.02, text_start + 0.6],
                   [y + y_step/2 + 0.005, y + y_step/2 + 0.005],
                   color='#1B5E20', linewidth=4, solid_capstyle='round')
            ax.text(text_start + 0.62, y + y_step/2 + 0.005, 'GOLD',
                   fontsize=9, color='#1B5E20', fontweight='bold',
                   va='center', ha='left')

        if i in PRED_BOUNDARIES and i not in GOLD_BOUNDARIES:
            # Predicted boundary - dashed line (no inline label)
            ax.plot([left_margin - 0.02, text_start + 0.6],
                   [y + y_step/2 + 0.005, y + y_step/2 + 0.005],
                   color='#C62828', linewidth=2, linestyle='--',
                   dash_capstyle='round')

        if i in PRED_BOUNDARIES and i in GOLD_BOUNDARIES:
            # Both - show gold with model marker (darker green, thicker)
            ax.plot([left_margin - 0.02, text_start + 0.6],
                   [y + y_step/2 + 0.005, y + y_step/2 + 0.005],
                   color='#1B5E20', linewidth=4, solid_capstyle='round')
            ax.text(text_start + 0.62, y + y_step/2 + 0.005, 'GOLD + model',
                   fontsize=9, color='#1B5E20', fontweight='bold',
                   va='center', ha='left')

        y -= y_step

    # Right side: Subtopic annotations
    bracket_x = 0.84  # Moved left from 0.88

    # Add header annotation just above first bracket
    first_bracket_top = y_start - 0 * y_step + 0.01
    ax.text(bracket_x, first_bracket_top + 0.025, 'Illustrative conversational\nsubtopics',
           fontsize=9, color='#555555', va='bottom', ha='left', style='italic')

    subtopic_labels = [
        (0, 2, "Request &\nclarification"),
        (3, 4, "Constraint\nrefinement"),
        (5, 7, "Option\nevaluation"),
        (8, 9, "Alternative\nexploration"),
        (10, 11, "Booking"),
    ]

    for start, end, label in subtopic_labels:
        y_top = y_start - start * y_step + 0.01
        y_bot = y_start - end * y_step - 0.025
        y_mid = (y_top + y_bot) / 2

        # Bracket
        ax.plot([bracket_x, bracket_x + 0.015, bracket_x + 0.015, bracket_x],
               [y_top, y_top, y_bot, y_bot],
               color='#999999', linewidth=1)

        # Label
        ax.text(bracket_x + 0.025, y_mid, label, fontsize=9,
               color='#666666', va='center', ha='left', style='italic')

    # Metrics box at bottom
    metrics_y = 0.06
    ax.text(0.08, metrics_y, "Exact-match F1:", fontsize=11, fontweight='bold',
           va='top')
    ax.text(0.08, metrics_y - 0.045,
           "  Precision = 1/4 = 0.25  (3 semantically valid boundaries counted as FP)",
           fontsize=10, fontfamily='monospace', va='top')
    ax.text(0.08, metrics_y - 0.08,
           "  Recall    = 1/1 = 1.00",
           fontsize=10, fontfamily='monospace', va='top')
    ax.text(0.08, metrics_y - 0.115,
           "  F1        = 0.40",
           fontsize=10, fontfamily='monospace', va='top', fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#1B5E20', linewidth=4,
               label='Gold boundary (coarse)'),
        Line2D([0], [0], color='#C62828', linewidth=2, linestyle='--',
               label='Subtopic boundary (model)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
             bbox_to_anchor=(0.98, 0.01), fontsize=10, frameon=True)

    # Formatting
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    plt.tight_layout(pad=0.5)
    return fig


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate Figure 1: Granularity Mismatch")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for figures")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: paper/figures (two levels up from experiments/figures)
        output_dir = Path(__file__).parent.parent.parent / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = create_figure()

    fig.savefig(output_dir / "granularity_mismatch.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "granularity_mismatch.pdf",
                bbox_inches='tight', facecolor='white')

    print(f"Saved to {output_dir}/granularity_mismatch.png")
    print(f"Saved to {output_dir}/granularity_mismatch.pdf")


# =============================================================================
# ASCII MOCKUP
# =============================================================================
ASCII_MOCKUP = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  0  User: I need a restaurant recommendation for tonight                    │
│  1  Asst: Sure! What cuisine are you in the mood for?                       │
│  2  User: Italian sounds good                                               │
│     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ pred              │ ← Constraint
│  3  Asst: Great choice. Any dietary restrictions?                           │   refinement
│  4  User: Yes, one person is gluten-free                                    │
│     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ pred              │ ← Specific
│  5  Asst: Trattoria Milano has excellent GF pasta options                   │   option
│  6  User: How's the atmosphere there?                                       │
│  7  Asst: It's upscale casual, good for dates or small groups               │
│     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ pred              │ ← Alternative
│  8  User: Actually, is there something more casual?                         │   exploration
│  9  Asst: Pasta House is more relaxed, also has GF options                  │
│     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ GOLD + pred        │ ← Booking
│ 10  User: That sounds perfect. Can you help me book it?                     │
│ 11  Asst: Of course. What time and how many people?                         │
│                                                                             │
│  ━━━ Gold boundary (coarse)    ┄┄┄ Predicted boundary (fine)                │
│                                                                             │
│  Exact-match F1:  Precision = 1/4 = 0.25 (3 "false positives")              │
│                   Recall = 1/1 = 1.00                                       │
│                   F1 = 0.40                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

CAPTION = r"""
\textbf{Granularity mismatch in topic segmentation.}
A restaurant recommendation dialogue with gold annotations marking only
the major conceptual shift (browsing $\to$ booking), while the model detects
four semantically meaningful subtopic transitions: (1) preferences $\to$ dietary
constraints, (2) constraints $\to$ specific option evaluation, (3) first option
$\to$ alternative exploration, and (4) exploration $\to$ booking logistics.
\emph{Every predicted boundary corresponds to a genuine topic shift}---none are
positioning errors or noise. Yet exact-match F1 treats three correct predictions
as false positives, yielding F1\,=\,0.40 despite perfect boundary detection at
a finer granularity. Windowed F1 does not resolve this: the predictions are not
``off by one turn''---they mark \emph{additional} valid boundaries that the
coarse gold standard omits. The failure mode is boundary \emph{density}, not
boundary \emph{position}.
"""

EXPLANATION = """
HOW THIS FIGURE SUPPORTS THE PAPER'S ARGUMENT:

1. EVERY PREDICTED BOUNDARY IS CORRECT
   - Turn 3: Shifts from "what cuisine" to "dietary restrictions"
   - Turn 5: Shifts from constraints to evaluating a specific restaurant
   - Turn 8: Shifts from first option to considering alternatives
   - Turn 10: Shifts from browsing to booking logistics

   A human reader can verify each is a real subtopic change.

2. THE GOLD ANNOTATION IS ALSO CORRECT
   - At a coarse level, there's one major shift: exploring → committing
   - This is a valid segmentation for applications that only need major phases

3. THE METRIC FAILURE IS STRUCTURAL
   - Exact-match F1 = 0.40 despite the model being "right"
   - Three predictions are labeled FP, but they're semantically valid
   - This is NOT fixable by adding tolerance (W-F1)—the boundaries aren't misplaced

4. IMPLICATIONS FOR EVALUATION
   - Comparing against a single gold standard conflates granularity with correctness
   - Need metrics that assess segment coherence, not just boundary positions
   - Or: evaluate at multiple granularity levels explicitly
"""

print(ASCII_MOCKUP)
print("\n" + "="*80 + "\nCAPTION:\n" + "="*80)
print(CAPTION)
print("\n" + "="*80 + "\nEXPLANATION:\n" + "="*80)
print(EXPLANATION)
