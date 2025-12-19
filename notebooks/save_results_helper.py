"""
Helper utilities and examples to save results and plots from
notebooks/02_features_and_models.ipynb.

This script is SAFE to run standalone. When executed directly, it will:
- Ensure results/ and figures/ directories exist
- Print guidance on how to use the helpers inside a notebook

Copy/paste the example snippets below into the notebook cells where indicated.
"""

import os


def ensure_output_dirs():
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)


def save_all_results(comparison_df, trading_metrics_df, equity_fig=None):
    """
    Save all results and plots to disk.

    Args:
        comparison_df: DataFrame with model comparison metrics
        trading_metrics_df: DataFrame with trading performance metrics
        equity_fig: Matplotlib figure object for equity curves (optional)
    """
    ensure_output_dirs()

    comparison_df.to_csv('../results/model_metrics.csv')
    trading_metrics_df.to_csv('../results/trading_metrics.csv')

    if equity_fig is not None:
        equity_fig.savefig('../figures/equity_curves_test.png', dpi=300, bbox_inches='tight')

    print("\n" + "=" * 70)
    print("RESULTS SAVED SUCCESSFULLY")
    print("=" * 70)
    print("✓ Model metrics:     results/model_metrics.csv")
    print("✓ Trading metrics:   results/trading_metrics.csv")
    if equity_fig is not None:
        print("✓ Equity curves:     figures/equity_curves_test.png")
    print("=" * 70)


# -----------------------------
# Notebook code snippets
# -----------------------------

NOTEBOOK_SNIPPETS = r"""
# After building comparison_df
import os
os.makedirs('../results', exist_ok=True)
comparison_df.to_csv('../results/model_metrics.csv')

# After building trading_metrics_df
os.makedirs('../results', exist_ok=True)
trading_metrics_df.to_csv('../results/trading_metrics.csv')

# When plotting equity curves (before plt.show())
import os
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/equity_curves_test.png', dpi=300, bbox_inches='tight')

# In 01_eda.ipynb, after price and returns plots
import os
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/price_and_returns.png', dpi=300, bbox_inches='tight')
"""


def main():
    ensure_output_dirs()
    print("\nOutput directories ensured:")
    print("- results/")
    print("- figures/\n")

    print("This helper provides examples to save results from your notebooks.\n")
    print("Copy the snippets below into 02_features_and_models.ipynb:")
    print("\n" + "-" * 70)
    print(NOTEBOOK_SNIPPETS)
    print("-" * 70)
    print("\nAlternatively, from a notebook you can call save_all_results(comparison_df,\ntrading_metrics_df, plt.gcf()) to save all outputs at once.")


if __name__ == '__main__':
    main()
