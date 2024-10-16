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

DATASET_NAMES = ["fm2", "bamboogle", "trickme", "advqa_combined"]

kappa_values = {
    "fm2": 0.22,
    "bamboogle": 0.93,
    "trickme": 0.56,
    "advqa_combined": 0.93,
}

DATASET_NAMES_SANITIZED = [
    "FM2 (AdvScore: -0.07)",
    "Bamboogle (AdvScore: -0.21)",
    "TrickMe (AdvScore: 0.13)",
    "AdvQA (<b>AdvScore: 0.31</b>)",
]

human_line_color = "rgba(46, 134, 193, 1)"  # Muted gray for humans
human_fill_color = "rgba(46, 134, 193, 0.4)"  # Muted gray for humans
human_marker_color = "rgba(26, 82, 118, 1)"  # Muted gray for humans

model_line_color = "rgba(203, 67, 53 , 1)"  # Slightly lighter gray for models
model_fill_color = "rgba(203, 67, 53 , 0.4)"  # Slightly lighter gray for models
model_marker_color = "rgba(123, 36, 28 , 1)"  # Slightly lighter gray for models

iif_line_color = "rgba(76, 175, 80, 0.7)"  # Muted green for IIF
iif_fill_color = "rgba(76, 175, 80, 0.4)"  # Muted green for IIF
iif_marker_color = "rgba(46, 106, 53, 1)"  # Muted green for IIF

global_max_y_row1 = 0
global_max_y_row2 = 0

sigma_unicode = "\u03c3"
theta_unicode = "\u03b8"
mu_unicode = "\u03bc"
kappa_unicode = "\u03ba"

p_2pl_unicode = f"{sigma_unicode}<sub>2pl</sub>({theta_unicode})"
margin_unicode = f"{mu_unicode}"


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


def create_p_func(dataset_name):
    questions_df = load_dataframe_dict(dataset_name)["questions"]
    diff = questions_df[DIFF_COLS].values
    rels = questions_df[REL_COLS].values

    def p_func(theta):
        values = irt_prob_func(theta[:, None], diff, rels)
        return values.mean(axis=-1)

    return p_func


def create_iif_func(dataset_name):
    questions_df = load_dataframe_dict(dataset_name)["questions"]
    diff = questions_df[DIFF_COLS].values
    rels = questions_df[REL_COLS].values

    def iif_func(theta):
        values = calculate_iif(theta[:, None], diff, rels)
        return values

    return iif_func


def get_experts_skills(dataset_name, agent_type: str):
    agents_df = load_dataframe_dict(dataset_name)["agents"]
    agents_df = agents_df[agents_df["subject_type"] == agent_type]
    if dataset_name == "trickme" and agent_type == "human":
        print("Taking all for trickme humans.")
        return agents_df
    skills = agents_df[SKILL_COLS]
    mean_skill = skills.mean()
    std_skill = skills.std()
    subject_idx = (skills > mean_skill).any(axis=1)
    experts_df = agents_df.loc[subject_idx]
    return experts_df


SKILL_ROW = 1
PROB_ROW = 2
IIF_ROW = 3

fig = make_subplots(
    rows=3,
    cols=4,
    subplot_titles=DATASET_NAMES_SANITIZED + [""] * 8,
    vertical_spacing=0.04,
    horizontal_spacing=0.03,
    shared_yaxes=True,
    shared_xaxes=True,
    row_heights=[0.3, 0.35, 0.35],
)
fig.update_annotations(font_size=25)

for i, dataset_name in enumerate(DATASET_NAMES, 1):
    # Generate points for smooth curves
    x_range = np.linspace(-3, 3, num=100)
    items_df = load_dataframe_dict(dataset_name)["questions"]
    diff = items_df[DIFF_COLS].values
    rels = items_df[REL_COLS].values
    humans_df = get_experts_skills(dataset_name, "human")
    models_df = get_experts_skills(dataset_name, "ai")
    human_mean = humans_df["skill_0"].mean()
    model_mean = models_df["skill_0"].mean()
    print(
        f"{dataset_name} Mean skills:",
        f"Human: {human_mean:.2f}",
        f"Model: {model_mean:.2f}",
    )
    iif_func = create_iif_func(dataset_name)
    p_func = create_p_func(dataset_name)

    info_density_max = 3

    # Calculate KDE for humans and models
    human_kde = stats.gaussian_kde(humans_df["skill_0"].dropna(), bw_method=0.5)
    model_kde = stats.gaussian_kde(models_df["skill_0"].dropna(), bw_method=0.5)

    # Calculate y values for all datasets
    skill_y_h = human_kde(x_range)
    skill_y_m = model_kde(x_range)
    iif_y = iif_func(x_range)
    prob_y = p_func(x_range)
    p_human_y = p_func(np.array([human_mean]))[0]
    p_model_y = p_func(np.array([model_mean]))[0]
    print(
        f"{dataset_name} P_margins: human:",
        p_human_y,
        "model:",
        p_model_y,
        "diff:",
        p_human_y - p_model_y,
    )

    if i == 1:
        global_max_y = max(max(skill_y_h), max(skill_y_m))
    else:
        global_max_y = max(global_max_y, max(skill_y_h), max(skill_y_m))

    # Row 1: Skill density
    def add_scatter_trace(y, name, line_color, fill_color):
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y,
                name=name,
                line={"color": line_color},
                fill="tozeroy",
                fillcolor=fill_color,
                showlegend=(i == 1),
            ),
            row=SKILL_ROW,
            col=i,
        )

    # Add vertical lines for means in row 1
    def add_vertical_line(x, y_max, color, width, row, col):
        fig.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=y_max,
            line={"color": color, "width": width, "dash": "dot"},
            row=row,
            col=col,
        )

    def add_horizontal_line(y, color, width, row, col):
        fig.add_shape(
            type="line",
            x0=x_range[0],
            y0=y,
            x1=x_range[-1],
            y1=y,
            line={"color": color, "width": width, "dash": "dot"},
            row=row,
            col=col,
        )

    add_scatter_trace(skill_y_h, "Human", human_line_color, human_fill_color)
    add_scatter_trace(skill_y_m, "Model", model_line_color, model_fill_color)

    add_vertical_line(human_mean, max(skill_y_h), human_line_color, 3, SKILL_ROW, i)
    add_vertical_line(model_mean, max(skill_y_m), model_line_color, 3, SKILL_ROW, i)

    # Row 2: Item Information Function with highlighted margin
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=iif_y,
            name="Item Information Function (IIF)",
            line={"color": iif_line_color},
            fill="tozeroy",
            fillcolor=iif_fill_color,
            showlegend=(i == 1),
        ),
        row=IIF_ROW,
        col=i,
    )

    # Highlight the margin between human and model means
    margin_start = min(human_mean, model_mean)
    margin_end = max(human_mean, model_mean)
    margin_x = np.linspace(margin_start, margin_end, num=10)
    margin_y = iif_func(margin_x)
    fig.add_trace(
        go.Scatter(
            x=margin_x,
            y=margin_y,
            name="Total Information",
            fill="tozeroy",
            fillcolor="rgba(0, 100, 0, 0.3)",
            line=dict(color="rgba(0, 100, 0, 0.8)"),
            marker=dict(size=3),
            showlegend=(i == 1),
        ),
        row=IIF_ROW,
        col=i,
    )

    # Add vertical lines for margin boundaries in row 2 and 3
    add_vertical_line(human_mean, info_density_max, human_line_color, 3, PROB_ROW, i)
    add_vertical_line(model_mean, info_density_max, model_line_color, 3, PROB_ROW, i)
    add_vertical_line(human_mean, info_density_max, human_line_color, 3, IIF_ROW, i)
    add_vertical_line(model_mean, info_density_max, model_line_color, 3, IIF_ROW, i)

    # Row 3: Probability values over theta
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=prob_y,
            name=f"Correctness Probability {p_2pl_unicode}",
            line={"color": "purple"},
            showlegend=(i == 1),
        ),
        row=PROB_ROW,
        col=i,
    )

    # Highlight the margin between human and model means in row 3
    margin_prob_y = prob_y[(x_range >= margin_start) & (x_range <= margin_end)]

    # Add vertical line for p-values difference
    p_value_start = p_func(np.array([[margin_start]]))[0]
    p_value_end = p_func(np.array([[margin_end]]))[0]
    # Determine the color gradient based on the sign of human margin - model margin
    color_start = (
        "rgba(46, 134, 193, 0.5)"
        if human_mean > model_mean
        else "rgba(203, 67, 53, 0.5)"
    )
    color_end = (
        "rgba(203, 67, 53, 0.5)"
        if human_mean > model_mean
        else "rgba(46, 134, 193, 0.5)"
    )

    fig.add_shape(
        type="rect",
        x0=x_range[0],
        x1=x_range[-1],
        y0=p_value_start,
        y1=p_value_end,
        fillcolor=color_start,
        line={"width": 0},
        row=PROB_ROW,
        col=i,
    )

    # Add annotation for P_margin
    annot_font_size = 20
    p_margin = p_human_y - p_model_y
    side = "left"

    # Add text that shows TIF =  tif_values[dataset_name]
    x_pos = x_range[0] if side == "left" else x_range[-1]
    fig.add_annotation(
        x=x_pos,
        y=2.5,
        text=f"{kappa_unicode} = {kappa_values[dataset_name]:.2f}",
        showarrow=False,
        font=dict(size=annot_font_size),
        xanchor=side,
        row=IIF_ROW,
        col=i,
    )
    x_pos = x_range[0] if side == "left" else x_range[-1]
    fig.add_annotation(
        x=x_pos,  # Move to the left side of the subplot
        y=(p_value_start + p_value_end) / 2,
        text=f"{margin_unicode} = {p_margin:.2f}",
        showarrow=False,
        font=dict(size=annot_font_size),
        # bgcolor="rgba(255, 255, 255, 0.7)",
        xanchor=side,  # Align text to the left
        row=PROB_ROW,
        col=i,
    )

    # Add vertical arrow to indicate the margin
    arrow_x = (
        x_pos - (0.02 * (x_range[-1] - x_range[0]))
        if side == "left"
        else x_pos + (0.02 * (x_range[-1] - x_range[0]))
    )
    arrow_color = human_line_color if human_mean > model_mean else model_line_color
    arrow_text = "↔"
    fig.add_annotation(
        x=arrow_x,
        y=(p_value_start + p_value_end) / 2,
        text=f"<b>{arrow_text}</b>",
        showarrow=False,
        font=dict(size=20, color=arrow_color),
        xanchor="center",
        yanchor="middle",
        textangle=90,
        row=PROB_ROW,
        col=i,
    )

    # Highlight the prob curve arc between the two vertical lines
    arc_color = "yellow"  # Gold yellow, slightly muted
    marker_colors = [human_marker_color, model_marker_color]
    if human_mean > model_mean:
        marker_colors = [model_marker_color, human_marker_color]
    fig.add_trace(
        go.Scatter(
            x=[margin_start, margin_end],
            y=[p_value_start, p_value_end],
            mode="lines+markers",
            line=dict(color=arc_color, width=4),
            marker=dict(color=marker_colors, size=7, symbol="circle"),
            showlegend=False,
        ),
        row=PROB_ROW,
        col=i,
    )

    add_horizontal_line(p_value_start, human_marker_color, 2, PROB_ROW, i)
    add_horizontal_line(p_value_end, model_marker_color, 2, PROB_ROW, i)

    # Update axes labels
    fig.update_xaxes(
        row=IIF_ROW,
        col=i,
        showticklabels=True,
        title_text="Skill (\u03b8)",
        title_font=dict(size=25),
        tickfont=dict(size=22, family="Roboto"),
    )
    fig.update_yaxes(
        title_text="Skill Density" if i == 1 else None,
        title_font=dict(size=25),
        row=SKILL_ROW,
        col=i,
        range=[0, global_max_y],
        tickfont=dict(family="Roboto"),
    )
    fig.update_yaxes(
        title_text=f"IIF({theta_unicode})" if i == 1 else None,
        title_font=dict(size=25),
        row=IIF_ROW,
        col=i,
        tickfont=dict(size=20, family="Roboto"),
        range=[0, 3],  # Set the yaxes max of iif to 3
    )
    fig.update_yaxes(
        title_text=f"{p_2pl_unicode}" if i == 1 else None,
        title_font=dict(size=25),
        row=PROB_ROW,
        col=i,
        tickfont=dict(size=20, family="Roboto"),
        range=[0, 1],
    )


fig.update_xaxes(
    tickfont=dict(size=20, family="Roboto"),
    dtick=1.0,
)

fig.update_yaxes(
    dtick=1.0,
    row=SKILL_ROW,
    range=[0, global_max_y],
)
fig.update_yaxes(
    showticklabels=True,
    tickfont=dict(size=20, family="Roboto"),
)
# Update layout
fig.update_layout(
    template="ggplot2",
    height=850,  # Increased height to accommodate the new row
    width=1800,
    showlegend=True,
    plot_bgcolor="rgba(240,240,240,1)",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.5)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        orientation="h",
        yanchor="top",
        y=-0.15,  # Adjusted position due to new row
        xanchor="center",
        x=0.5,
        font=dict(size=25, family="Roboto"),
    ),
)

fig.show()
fig.write_image("figs/skill_density_iif.pdf")

# %%
