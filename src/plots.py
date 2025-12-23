import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import learning_curve

__all__ = [
    'plot_missing_values',
    'plot_distributions_by_label',
    'plot_numerical_distributions',
    'plot_categorical_distributions',
    'plot_correlation_matrix',
    'plot_boxplots',
    'plot_learning_curve',
    'plot_feature_importance',
    'plot_target_distribution',
    'plot_statistical_tests',
    'plot_imputation_comparison',
    'save_all_figures',
    'create_summary_report',
    'plot_experiment_comparison'
]

BLUE_PALETTE = ["#D0E1F2", "#B0D2E7", "#84BCDB", "#57A0CE", "#3383BE", "#1764AB"]
PURPLE_PALETTE = ["#E6D9F5", "#D4BEF0", "#B89BE6", "#9C78DC", "#7F55D1", "#6432C7"]

def plot_missing_values(data, title="Missing Values Analysis", threshold=20, save_path=None):
    """
    Visualize missing data percentages with a horizontal bar chart.
    """
    if isinstance(data, pd.DataFrame):
        missing_pct = (data.isnull().sum() / len(data) * 100)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    else:
        missing_pct = data.sort_values(ascending=True)
    
    if len(missing_pct) == 0:
        print("No missing values found in the data!")
        return
    
    df_plot = missing_pct.reset_index()
    df_plot.columns = ['Feature', 'Percentage']

    fig = px.bar(
        df_plot, x='Percentage', y='Feature', orientation='h',
        text_auto='.1f', color='Percentage', 
        color_continuous_scale='Blues',
        title=title
    )

    fig.update_layout(
        title_x=0.5, 
        template="plotly",
        xaxis_title="Percentage (%)", 
        yaxis_title="Feature" ,
        height=max(400, len(missing_pct) * 30),
        showlegend=False
    )
    
    if threshold:
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"{threshold}% threshold")
    
    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig

def plot_distributions_by_label(df, col, label_col="label", save_path=None):
    """
    Plot feature distributions grouped by label (categorical or numerical).
    """
    if df[col].nunique() <= 10:
        ct = pd.crosstab(df[col], df[label_col], normalize="index")
        fig = go.Figure()
        
        for i, lab in enumerate(sorted(df[label_col].unique())):
            fig.add_bar(
                name=f"Label {lab}", 
                x=ct.index.astype(str), 
                y=ct[lab],
                marker_color=BLUE_PALETTE[i % len(BLUE_PALETTE)]
            )
        
        fig.update_layout(
            barmode="stack", 
            title=f"{col} Distribution by {label_col}",
            xaxis_title=col,
            yaxis_title="Proportion",
            title_x=0.5
        )
    
    else:
        fig = px.box(
            df, x=label_col, y=col, 
            color=label_col,
            color_discrete_sequence=BLUE_PALETTE,
            title=f"{col} distribution by {label_col}"
        )
        
        fig.update_layout(
            title_x=0.5,
            template="plotly",
            xaxis_title=label_col,
            yaxis_title=col
        )

    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig


def plot_numerical_distributions(df, cols=None, bins=30, save_path=None):
    """
    Plot histograms for multiple numerical columns in a grid.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(cols)
    n_rows = (n_cols + 4) // 5  
    
    fig = make_subplots(
        rows=n_rows, cols=5,
        subplot_titles=cols,
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(cols):
        row = idx // 5 + 1
        col_pos = idx % 5 + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                nbinsx=bins,
                name=col,
                marker_color=BLUE_PALETTE[3],
                showlegend=False
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text="Numerical Features Distribution",
        template="plotly",
        title_x=0.5,
        height=300 * n_rows,
        width=1400, 
    )

    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig


def plot_categorical_distributions(df, cols=None, save_path=None):
    """
    Plot bar charts for multiple categorical columns in a grid.
    """
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    n_cols = len(cols)
    n_rows = (n_cols + 3) // 4  
    
    fig = make_subplots(
        rows=n_rows, cols=4,
        subplot_titles=cols,
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(cols):
        row = idx // 4 + 1
        col_pos = idx % 4 + 1
        
        value_counts = df[col].value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                name=col,
                marker_color=BLUE_PALETTE[3],
                showlegend=False
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text="Categorical Features Distribution",
        template="plotly",
        title_x=0.5,
        height=300 * n_rows,
        width=1200,  
    )

    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig

def plot_correlation_matrix(df, cols=None, method='pearson', show=True, save_path=None):
    """
    Create an annotated correlation heatmap.
    """
    if cols:
        corr = df[cols].corr(method=method)
    else:
        corr = df.select_dtypes(include=[np.number]).corr(method=method)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Blues',
        zmid=0,
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(title="Correlation")
    ))

    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            val = corr.iloc[i, j]
            fig.add_annotation(
                x=col, y=row,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    color="white" if abs(val) > 0.6 else "black",
                    size=9
                )
            )
    
    fig.update_layout(
        title={
            'text': f"Correlation Matrix ({method.capitalize()})",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(tickangle=45),
        template="plotly"
    )
    
    if save_path:
        fig.write_image(save_path, scale=3)
    
    return fig

def plot_boxplots(df, cols=None, save_path=None):
    """
    Create box plots for outlier detection.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(cols)
    n_rows = (n_cols + 2) // 3
    
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=cols,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        fig.add_trace(
            go.Box(
                y=df[col].dropna(),
                name=col,
                marker_color=BLUE_PALETTE[3],
                showlegend=False
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text="Outlier Detection - Box Plots",
        title_x=0.5,
        height=300 * n_rows,
        template="plotly"
    )

    if save_path:
        fig.write_image(save_path, scale=3)
    
    return fig

def plot_learning_curve(estimator, X, y, cv=5, save_path=None):
    """
    Plot learning curves for bias-variance diagnosis.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 0, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Train Std'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Test Std'
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='purple', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        mode='lines+markers',
        name='Cross-Validation Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Learning Curve - Bias/Variance Diagnosis",
        title_x=0.5,
        xaxis_title="Training Set Size",
        yaxis_title="Accuracy Score",
        legend=dict(
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.95
        ),
        height=600,
        template="plotly"
    )
    
    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig


def plot_feature_importance(feature_names, importances, top_n=20, save_path=None, show=True, title="Feature Importance"):
    """
    Display feature importance from model coefficients or feature importances.
    """
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'AbsImportance': np.abs(importances)
    })
    
    df_imp = df_imp.sort_values(by='AbsImportance', ascending=False).head(top_n)
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    colors = ['rgba(0, 0, 255, 0.6)' if x > 0 else 'rgba(128, 0, 128, 0.6)' 
              for x in df_imp['Importance']]

    fig = go.Figure(go.Bar(
        x=df_imp['Importance'],
        y=df_imp['Feature'],
        orientation='h',
        marker_color=colors,
        text=df_imp['Importance'].round(4),
        textposition='auto'
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Importance Value",
        yaxis_title="Feature",
        height=max(500, top_n * 25),
    )
    
    fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
    
    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig

def plot_target_distribution(y, title="Target Distribution", save_path=None):
    """
    Visualize target variable distribution.
    """
    if isinstance(y, pd.Series):
        value_counts = y.value_counts().sort_index()
    else:
        value_counts = pd.Series(y).value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker_color=BLUE_PALETTE[3],
            text=value_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Class",
        yaxis_title="Count",
        height=500,
        template="plotly"
    )

    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig

def plot_statistical_tests(test_results_df, test_type='chi-square', save_path=None):
    """
    Visualize results from statistical tests.
    """
    df = test_results_df.sort_values('p-value', ascending=False)
    
    colors = [BLUE_PALETTE[4] if sig == 'Yes' else BLUE_PALETTE[1] 
              for sig in df['Significant']]
    
    fig = go.Figure(go.Bar(
        x=-np.log10(df['p-value']),  
        y=df['Feature'],
        orientation='h',
        marker_color=colors,
        text=df['p-value'].apply(lambda x: f"{x:.2e}"),
        textposition='auto'
    ))
 
    fig.add_vline(
        x=-np.log10(0.05),
        line_width=2,
        line_dash="dash",
        line_color='red',
        annotation_text="α = 0.05"
    )
    
    fig.update_layout(
        title=f"{test_type.upper()} Test Results",
        title_x=0.5,
        xaxis_title="-log10(p-value)",
        yaxis_title="Feature",
        template="plotly",
        height=max(400, len(df) * 30)
    )

    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig

def plot_imputation_comparison(results_dict, metric='accuracy', save_path=None):
    """
    Compare different imputation strategies.
    """
    strategies = list(results_dict.keys())
    values = list(results_dict.values())
    
    fig = go.Figure(go.Bar(
        x=strategies,
        y=values,
        marker_color=BLUE_PALETTE[3],
        text=[f"{v:.4f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Imputation Strategy Comparison - {metric.capitalize()}",
        title_x=0.5,
        template="plotly",
        xaxis_title="Imputation Strategy",
        yaxis_title=metric.capitalize(),
        height=500
    )

    if save_path:
        fig.write_image(save_path, scale=3, width=1200, height=700)
    
    return fig

def plot_experiment_comparison(comparison, comparison_sorted, save_path=None):
    """
    Generates a performance comparison and the impact of Feature Engineering.
    """
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Performance: No FE vs With FE", "Feature Engineering Impact by Experiment"),
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Bar(
            x=comparison['Experiment'], 
            y=comparison['No FE'],
            name='No FE',
            marker_color='rgba(128, 0, 128, 0.2)',
            opacity=0.8
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=comparison['Experiment'], 
            y=comparison['With FE'],
            name='With FE',
            marker_color='rgba(128, 0, 128, 0.6)',
            opacity=0.8
        ),
        row=1, col=1
    )

    colors_improvement = ['rgba(128, 0, 128, 0.6)' if val > 0 else 'rgba(128, 0, 128, 0.2)' for val in comparison_sorted['Improvement']]
    
    fig.add_trace(
        go.Bar(
            x=comparison_sorted['Experiment'],
            y=comparison_sorted['Improvement'],
            marker_color=colors_improvement,
            opacity=0.8,
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        barmode='group',
        height=600,
        width=1400,
        showlegend=True,
        template="plotly"
    )

    fig.update_xaxes(title_text="Experiment", tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="CV Accuracy", row=1, col=1)
    
    fig.update_xaxes(title_text="Experiment (sorted by improvement)", tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Accuracy Improvement", row=1, col=2)

    fig.add_shape(type="line", x0=-0.5, x1=len(comparison_sorted)-0.5, y0=0, y1=0,
                  line=dict(color="black", width=1), row=1, col=2)

    if save_path:
        fig.write_image(save_path, scale=3)
    
    return fig

def save_all_figures(figures_dict, base_path="../images"):
    """
    Save multiple figures at once.
    """
    import os
    os.makedirs(base_path, exist_ok=True)
    
    for filename, fig in figures_dict.items():
        filepath = os.path.join(base_path, filename)
        fig.write_image(filepath, scale=3, width=1200, height=700)
        print(f"Saved: {filepath}")


def create_summary_report(df, output_path=None):
    """
    Create a comprehensive visual summary report.
    """
    
    print("Generating comprehensive summary report...")
    
    summary_stats = df.describe()
    
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features with missing values: {len(missing)}")
    print(f"Total missing percentage: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    
    return summary_stats, missing