import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_interactive_visualizations(ticker, ratios, valuation_metrics, avg_prediction, historical_ratios, comparison, risk_metrics):
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "domain"}]],
        subplot_titles=("Current Financial Ratios", "Valuation Metrics", 
                        "Historical Financial Ratios", "Comparative Analysis",
                        "Risk Metrics", "Fortress Balance Sheet Prediction")
    )

    # Current Financial Ratios
    fig.add_trace(go.Bar(x=list(ratios.keys()), y=list(ratios.values()), name="Ratios"),
                  row=1, col=1)

    # Valuation Metrics
    fig.add_trace(go.Bar(x=list(valuation_metrics.keys()), y=list(valuation_metrics.values()), name="Metrics"),
                  row=1, col=2)

    # Historical Financial Ratios
    for column in historical_ratios.columns:
        fig.add_trace(go.Scatter(x=historical_ratios.index, y=historical_ratios[column], name=column),
                      row=2, col=1)

    # Comparative Analysis
    for company in comparison.index:
        fig.add_trace(go.Bar(x=comparison.columns, y=comparison.loc[company], name=company),
                      row=2, col=2)

    # Risk Metrics
    fig.add_trace(go.Bar(x=list(risk_metrics.keys()), y=list(risk_metrics.values()), name="Risk"),
                  row=3, col=1)

    # Fortress Balance Sheet Prediction
    fig.add_trace(go.Pie(labels=['Fortress', 'Non-Fortress'], values=[avg_prediction, 1-avg_prediction]),
                  row=3, col=2)

    fig.update_layout(height=1200, width=1000, title_text=f"Advanced Analysis for {ticker}")
    fig.write_html(f"{ticker}_interactive_analysis.html")
    print(f"Interactive visualizations saved as {ticker}_interactive_analysis.html")

def plot_feature_importance(feature_importance, feature_names):
    fig = go.Figure(go.Bar(x=feature_names, y=feature_importance))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance"
    )
    fig.write_html("feature_importance.html")
    print("Feature importance plot saved as feature_importance.html")