# %%
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from funcs_mirt import load_dataframe_dict

DATASET_NAMES = ["advqa_combined", "fm2", "bamboogle", "trickme"]

DATASET_NAMES_SANITIZED = ["AdvQA", "FM2", "Bamboogle", "TrickMe"]

human_line_color = "rgba(46, 134, 193, 1)"  # Muted gray for humans
human_fill_color = "rgba(46, 134, 193, 0.5)"  # Muted gray for humans
human_marker_color = "rgba(26, 82, 118, 1)"  # Muted gray for humans

model_line_color = "rgba(203, 67, 53 , 1)"  # Slightly lighter gray for models
model_fill_color = "rgba(203, 67, 53 , 0.5)"  # Slightly lighter gray for models
model_marker_color = "rgba(123, 36, 28 , 1)"  # Slightly lighter gray for models


fig = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=DATASET_NAMES_SANITIZED,
    shared_yaxes=True,
    shared_xaxes=True,
)
for i, dataset_name in enumerate(DATASET_NAMES, 1):
    agents_df = load_dataframe_dict(dataset_name)["agents"]
    humans_df = agents_df[agents_df["subject_type"] == "human"]
    models_df = agents_df[agents_df["subject_type"] == "ai"]
    human_mean = humans_df["skill_0"].mean()
    model_mean = models_df["skill_0"].mean()

    # Calculate KDE for humans and models
    human_kde = stats.gaussian_kde(humans_df["skill_0"].dropna(), bw_method=0.5)
    model_kde = stats.gaussian_kde(models_df["skill_0"].dropna(), bw_method=0.5)

    # Generate points for smooth curves
    x_range = np.linspace(
        min(humans_df["skill_0"].min(), models_df["skill_0"].min()),
        max(humans_df["skill_0"].max(), models_df["skill_0"].max()),
        num=200,
    )
    x_range = np.linspace(-2, 2, num=200)
    # Add vertical lines for means
    fig.add_shape(
        type="line",
        x0=human_mean,
        y0=0,
        x1=human_mean,
        y1=max(human_kde(x_range)),
        line={"color": human_marker_color, "width": 2, "dash": "dot"},
        row=1,
        col=i,
    )
    fig.add_shape(
        type="line",
        x0=model_mean,
        y0=0,
        x1=model_mean,
        y1=max(model_kde(x_range)),
        line={"color": model_marker_color, "width": 2, "dash": "dot"},
        row=1,
        col=i,
    )

    # Add traces for humans and models with filled area
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=human_kde(x_range),
            name="Human",
            line={"color": human_line_color},
            fill="tozeroy",
            fillcolor=human_fill_color,
            showlegend=(i == 1),  # Show legend only for the first subplot
        ),
        row=1,
        col=i,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=model_kde(x_range),
            name="AI",
            line={"color": model_line_color},
            fill="tozeroy",
            fillcolor=model_fill_color,
            showlegend=(i == 1),  # Show legend only for the first subplot
        ),
        row=1,
        col=i,
    )

    # Update axes labels
    fig.update_xaxes(row=1, col=i, showticklabels=True)
    fig.update_yaxes(
        title_text="Density" if i == 1 else None, row=1, col=i, showticklabels=True
    )

# Update layout to match ggplot2 theme
fig.update_layout(
    template="ggplot2",
    height=300,
    width=1200,
    showlegend=True,
    margin=dict(l=10, r=10, t=20, b=45),  # Increased bottom margin for legend and title
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.0,
    ),  # Moved legend to bottom
    title=dict(
        text="Skill Distribution Across Datasets",
        y=0.02,  # Adjusted y position to accommodate legend
        # x=0.5,
        xanchor="center",
        yanchor="bottom",
    ),
)

fig.show()
fig.write_image("figs/skill_density.pdf")
# %%
