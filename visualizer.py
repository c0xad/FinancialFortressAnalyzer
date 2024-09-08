import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_visualizations(ticker, ratios, valuation_metrics, avg_prediction, historical_ratios, comparison, risk_metrics):
    # Set up the plot style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 30))

    # Plot financial ratios
    ax1.bar(ratios.keys(), ratios.values())
    ax1.set_title(f"Current Financial Ratios for {ticker}")
    ax1.set_ylabel("Ratio Value")
    ax1.tick_params(axis='x', rotation=45)

    # Plot valuation metrics
    ax2.bar(valuation_metrics.keys(), valuation_metrics.values())
    ax2.set_title(f"Valuation Metrics for {ticker}")
    ax2.set_ylabel("Metric Value")
    ax2.tick_params(axis='x', rotation=45)

    # Plot historical ratios
    historical_ratios.plot(ax=ax3)
    ax3.set_title(f"Historical Financial Ratios for {ticker}")
    ax3.set_ylabel("Ratio Value")
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot comparative analysis
    comparison.T.plot(kind='bar', ax=ax4)
    ax4.set_title("Comparative Analysis")
    ax4.set_ylabel("Ratio Value")
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot risk metrics
    ax5.bar(risk_metrics.keys(), risk_metrics.values())
    ax5.set_title(f"Risk Metrics for {ticker}")
    ax5.set_ylabel("Metric Value")
    ax5.tick_params(axis='x', rotation=45)

    # Plot prediction likelihood
    ax6.pie([avg_prediction, 1-avg_prediction], labels=['Fortress', 'Non-Fortress'], autopct='%1.1f%%')
    ax6.set_title(f"Predicted Likelihood of Fortress Balance Sheet for {ticker}")

    plt.tight_layout()
    plt.savefig(f"{ticker}_advanced_analysis.png")
    plt.close()

    print(f"Advanced visualizations saved as {ticker}_advanced_analysis.png")