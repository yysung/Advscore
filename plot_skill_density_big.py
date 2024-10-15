# %%
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from funcs_mirt import (
    DIFF_COLS,
    REL_COLS,
    SKILL_COLS,
    irt_logit_func,
    load_dataframe_dict,
)

DATASET_NAMES = [ "fm2","bamboogle", "trickme","advqa_combined"]

DATASET_NAMES_SANITIZED = [ "FM2 (AdvScore: -0.08)", "BAMBOOGLE (AdvScore: -0.06)","TRICKME (AdvScore: 0.12)", "AdvQA (AdvScore: 0.13)" ]

human_line_color = "rgba(46, 134, 193, 1)"  # Muted gray for humans
human_fill_color = "rgba(46, 134, 193, 0.5)"  # Muted gray for humans
human_marker_color = "rgba(26, 82, 118, 1)"  # Muted gray for humans

model_line_color = "rgba(203, 67, 53 , 1)"  # Slightly lighter gray for models
model_fill_color = "rgba(203, 67, 53 , 0.5)"  # Slightly lighter gray for models
model_marker_color = "rgba(123, 36, 28 , 1)"  # Slightly lighter gray for models

iif_line_color = "rgba(76, 175, 80, 0.7)"  # Muted green for IIF
iif_fill_color = "rgba(76, 175, 80, 0.4)"  # Muted green for IIF
iif_marker_color = "rgba(46, 106, 53, 1)"  # Muted green for IIF

global_max_y_row1 = 0
global_max_y_row2 = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def irt_prob_func(skills, diff, rels):
    # skills: (n_agents, n_dim)
    logits = irt_logit_func(skills, diff, rels)
    probs = sigmoid(logits)
    return probs


def calculate_iif(skills, diff, rels):
    """Calculate Item Information Function (MIRT) for each item."""
    P = irt_prob_func(skills, diff, rels)
    Q = 1 - P
    pq = P * Q
    rel_sq = rels * rels
    value = rel_sq[:, 0] * pq
    return value.mean(axis=-1)



fig = make_subplots(
    rows=2,
    cols=4,
    subplot_titles=DATASET_NAMES_SANITIZED + [""] * 4,
    vertical_spacing=0.13,
    horizontal_spacing=0.05,
    shared_yaxes=True,
    shared_xaxes=True,
    # row_heights=[0.6, 0.4],
)
fig.update_annotations(font_size=25)

for i, dataset_name in enumerate(DATASET_NAMES, 1):
    # Generate points for smooth curves
    x_range = np.linspace(-3, 3, num=100)

    agents_df = load_dataframe_dict(dataset_name)["agents"]
    items_df = load_dataframe_dict(dataset_name)["questions"]
    diff = items_df[DIFF_COLS].values
    rels = items_df[REL_COLS].values
    skills = np.linalg.norm(agents_df[SKILL_COLS].values, axis=-1)
    humans_df = agents_df[agents_df["subject_type"] == "human"]
    models_df = agents_df[agents_df["subject_type"] == "ai"]
    human_mean = humans_df["skill_0"].mean()
    model_mean = models_df["skill_0"].mean()
    iif_values = calculate_iif(x_range[:, None], diff, rels)

    # Calculate KDE for humans and models
    human_kde = stats.gaussian_kde(humans_df["skill_0"].dropna(), bw_method=0.5)
    model_kde = stats.gaussian_kde(models_df["skill_0"].dropna(), bw_method=0.5)
    iif_kde = stats.gaussian_kde(iif_values, bw_method=0.5)

    # Calculate y values for all datasets
    human_y = human_kde(x_range)
    model_y = model_kde(x_range)
    iif_y = iif_kde(x_range)


    if i == 1:
        global_max_y = max(max(human_y), max(model_y))
    else:
        global_max_y = max(global_max_y, max(human_y), max(model_y))

    # Row 1: Skill density
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=human_y,
            name="Human",
            line={"color": human_line_color},
            fill="tozeroy",
            fillcolor=human_fill_color,
            showlegend=(i == 1),
        ),
        row=1,
        col=i,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=model_y,
            name="Model",
            line={"color": model_line_color},
            fill="tozeroy",
            fillcolor=model_fill_color,
            showlegend=(i == 1),
        ),
        row=1,
        col=i,
    )

    # Add vertical lines for means in row 1
    fig.add_shape(
        type="line",
        x0=human_mean,
        y0=0,
        x1=human_mean,
        y1=max(human_y),
        line={"color": human_marker_color, "width": 4, "dash": "dot"},
        row=1,
        col=i,
    )
    fig.add_shape(
        type="line",
        x0=model_mean,
        y0=0,
        x1=model_mean,
        y1=max(model_y),
        line={"color": model_marker_color, "width": 4, "dash": "dot"},
        row=1,
        col=i,
    )

    # Row 2: Item Information Function with highlighted margin
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=iif_y,
            #name="IIF(\u03b8)",
            name="IIF",
            line={"color": iif_line_color},
            fill="tozeroy",
            fillcolor=iif_fill_color,
            showlegend=(i == 1),
        ),
        row=2,
        col=i,
    )

    # Highlight the margin between human and model means
    margin_start = min(human_mean, model_mean)
    margin_end = max(human_mean, model_mean)
    margin_y = iif_y[(x_range >= margin_start) & (x_range <= margin_end)]
    margin_x = x_range[(x_range >= margin_start) & (x_range <= margin_end)]

    fig.add_trace(
        go.Scatter(
            x=margin_x,
            y=margin_y,
            name="Margin",
            fill="tozeroy",
            fillcolor="rgba(255, 255, 0, 0.3)",
            line=dict(color="rgba(255, 255, 0, 0.8)"),
            showlegend=(i == 1),
        ),
        row=2,
        col=i,
    )

    # Add vertical lines for margin boundaries in row 2
    fig.add_shape(
        type="line",
        x0=margin_start,
        y0=0,
        x1=margin_start,
        y1=4,
        line={"color": "black", "width": 4, "dash": "dot"},
        row=2,
        col=i,
    )
    fig.add_shape(
        type="line",
        x0=margin_end,
        y0=0,
        x1=margin_end,
        y1=4,
        line={"color": "black", "width": 4, "dash": "dot"},
        row=2,
        col=i,
    )

    # Update axes labels
    fig.update_xaxes(row=1, col=i, showticklabels=True)
    fig.update_xaxes(
        row=2,
        col=i,
        showticklabels=False,
        title_text="Skill (\u03b8)",
        title_font=dict(size=25),
        tickfont=dict(size=22),
    )
    fig.update_yaxes(
        title_text="Skill Density" if i == 1 else None,
        title_font=dict(size=25),
        row=1,
        col=i,
        showticklabels=True,
        range=[0, 1.8],
    )
    fig.update_yaxes(
        #title_text="Item Info. Density (\u03b8)" if i == 1 else None,
        title_text="Item Info. Density" if i == 1 else None,
        title_font=dict(size=25),
        row=2,
        col=i,
        showticklabels=True,
        tickfont=dict(size=20),
        
    )
    fig.update_xaxes(
        tickfont=dict(size=20),
        dtick=1.0,
        showticklabels=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=20),
    )
# Update layout
fig.update_layout(
    template="ggplot2",
    height=750,
    width=1600,
    showlegend=True,
    margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            orientation="h",  # Make the legend horizontal
            yanchor="top",    # Anchor the legend to the top
            y=-0.2,           # Position below the graph
            xanchor="center",
            x=0.5,
            font=dict(size=25),
        ),
    # title=dict(
    #     text="Skill Distribution and Item Information Function Across Datasets",
    #     font=dict(size=24),
    #     y=0.98,
    #     x=0.5,
    #     xanchor="center",
    #     yanchor="top",
    # ),
)

fig.show()
fig.write_image("figs/skill_density_and_iif.pdf")

# %%
