# %%
import os.path as osp

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from rich import print as rprint
from scipy import integrate
from tabulate import tabulate

import funcs_caimira
import funcs_mirt

# Comment/uncomment to switch between methods
funcs = funcs_mirt
# funcs = funcs_caimira

# Load the functions and constants from the selected irt method
load_dataframe_dict = funcs.load_dataframe_dict
irt_logit_func = funcs.irt_logit_func
SKILL_COLS = funcs.SKILL_COLS
REL_COLS = funcs.REL_COLS
DIFF_COLS = funcs.DIFF_COLS

DATASET_NAMES = ["advqa_combined", "fm2", "bamboogle", "trickme"]

MODELS_BY_TIME = {
    "2020": ["DPR"],
    "2021": ["GPT-3-Instruct"],
    "2022": ["GPT-3.5-TURBO"],
    "2023": [
        "Mistral-0.1-instruct",
        "GPT-4",
        "Llama-2-7b-chat",
        "Llama-2-70b-chat",
    ],
    "2024": [
        "Llama-2-7b-chat",
        "Llama-2-70b-chat",
        "Llama-3-8b-instruct",
        "Llama-3-70b-instruct",
        "rag-command-r-plus",
    ],
}

YEARS = sorted(MODELS_BY_TIME.keys())

CUM_MODELS_BY_TIME = {}
for i, year in enumerate(sorted(YEARS)):
    if i == 0:
        CUM_MODELS_BY_TIME[year] = MODELS_BY_TIME[year]
    else:
        CUM_MODELS_BY_TIME[year] = (
            CUM_MODELS_BY_TIME[YEARS[i - 1]] + MODELS_BY_TIME[year]
        )

rprint(CUM_MODELS_BY_TIME)


def get_agent_type(subject):
    name = subject.lower()
    if (
        "gpt" in name
        or "rag" in name
        or "cohere" in name
        or "llama" in name
        or "mistral" in name
        or "dpr" in name
        or "chatgpt" in name
        or "llama3" in name
    ):
        return "ai"
    return "human"


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
    if rels.shape[1] == 1:
        rel_sq = rels * rels
        return rel_sq[:, 0] * pq
    else:
        rel_sq = np.einsum("bi,bj->bij", rels, rels)
        return rel_sq * pq[:, :, None, None]


def get_iif_metrics_1d(diff, rels, theta_range=(-5, 5)):
    """Calculate metrics based on Item Information Function"""
    theta = np.linspace(*theta_range, 5000)[:, None]
    iif = calculate_iif(theta, diff, rels)
    # iif is (n_theta, n_items)
    peak_info = np.max(iif, axis=0)
    peak_theta = theta[np.argmax(iif, axis=0)]
    total_info = []
    for i in range(iif.shape[1]):
        total_info.append(integrate.simpson(y=iif[:, i], x=theta[:, 0]))
    total_info = np.array(total_info)
    info_width = np.sum(iif > (peak_info[None, :] / 2), axis=0) * (theta[1] - theta[0])

    return total_info, peak_info, peak_theta, info_width


def get_iif_total_info_1d(diff, rels, theta_range=(-5, 5)):
    return get_iif_metrics_1d(diff, rels, theta_range)[0]


def get_iif_total_info_2d(skills, diff, rels):
    iif_matrices = calculate_iif(skills, diff, rels)
    print("iif_matrices:", iif_matrices[0, 0])
    print("iif_matrices.shape", iif_matrices.shape)
    # Compute determinant of each matrix
    dets = np.linalg.det(iif_matrices)
    print("dets.shape", dets.shape)
    print("dets:", dets)
    return dets.mean(axis=0)


def get_iif_total_info(skills, diff, rels):
    if rels.shape[1] == 1:
        return get_iif_total_info_1d(diff, rels, theta_range=(-5, 5))
    else:
        return get_iif_total_info_2d(skills, diff, rels)


def exp_normalize(x, k=1):
    return 1 - np.exp(-k * x)


def get_kappa_aggdisc(dataframe_dict):
    questions_df = dataframe_dict["questions"]
    print(questions_df.columns)
    # TODO: Check if this is correct
    agg_disc = questions_df[REL_COLS].values
    return exp_normalize(agg_disc)


def get_kappa_iif(dataframe_dict):
    questions_df = dataframe_dict["questions"]
    skills = dataframe_dict["agents"][SKILL_COLS].values
    diff = questions_df[DIFF_COLS].values
    rels = questions_df[REL_COLS].values
    total_info = get_iif_total_info(skills, diff, rels)
    return exp_normalize(total_info)


def create_avg_prob_subject(skills_df, questions_df):
    # skills: (n_agents, n_dim)
    # questions_df: (n_items, n_dim)
    skills = skills_df[SKILL_COLS].values
    diff = questions_df[DIFF_COLS].values  # (n_items, n_dim)
    rels = questions_df[REL_COLS].values  # (n_items, n_dim)
    probs = irt_prob_func(skills, diff, rels)
    return probs.mean(axis=-1)


def create_avg_prob(
    dataframe_dict,
    subject_type: str,
    ai_year: "2024",
    agg: str = "weighted_mean",
    cumulative: bool = False,
):
    questions_df = dataframe_dict["questions"]
    agents_df = dataframe_dict["agents"]
    models_by_time = CUM_MODELS_BY_TIME if cumulative else MODELS_BY_TIME

    skills_subset = agents_df[agents_df["subject_type"] == subject_type]

    if subject_type == "ai":
        ai_this_year = models_by_time[ai_year]
        skills_subset = skills_subset[skills_subset["subject_id"].isin(ai_this_year)]

    if agg == "max":
        skills_subset = skills_subset.loc[skills_subset["eff_skill"].idxmax()]
        max_skill = skills_subset[SKILL_COLS].values

    if agg == "top5":
        skills_subset = skills_subset.sort_values(by="eff_skill", ascending=False).head(
            5
        )
        max_skill = skills_subset[SKILL_COLS].values

    if agg == "best_mean":
        # Select the row where full_skill is just greater than the mean
        mean_skill = skills_subset["eff_skill"].mean()
        # print("Mean skill for", subject_type, ":", mean_skill)
        skills_subset = skills_subset[skills_subset["eff_skill"] > mean_skill]
        skills_subset = skills_subset.loc[
            (skills_subset["eff_skill"] - mean_skill).abs().idxmin()
        ]
        max_skill = skills_subset[SKILL_COLS].values

    if agg == "mean":
        skills = skills_subset[SKILL_COLS].values
        mean_skill = skills.mean(axis=0)
        # print("Mean skills:", mean_skill.shape)
        max_skill = mean_skill

    if agg == "weighted_mean":
        p_star = create_avg_prob_subject(skills_subset, questions_df)
        p_star = skills_subset["accuracy"].values + 1e-6
        skills = skills_subset[SKILL_COLS].values
        mean_skill = (skills * p_star[:, None]).sum(axis=0) / p_star.sum()
        max_skill = mean_skill

    if agg == "t25_mean":
        # Select the top 25%
        skills_subset = skills_subset.sort_values(by="eff_skill", ascending=False).head(
            int(len(skills_subset) * 0.25)
        )
        max_skill = skills_subset[SKILL_COLS].values

    diff = questions_df[DIFF_COLS].values  # (n_items, n_dim)
    rels = questions_df[REL_COLS].values  # (n_items, n_dim)
    probs = irt_prob_func(max_skill, diff, rels)
    return probs.mean(), probs.std(), probs


def compute_percieved_expert_diff(
    dataframe_dict,
    subject_type: str,
    is_crowd: bool = False,
):
    questions_df = dataframe_dict["questions"]
    agents_df = dataframe_dict["agents"]
    rels = questions_df[REL_COLS].values
    diff = questions_df[DIFF_COLS].values

    skills_subset = agents_df[agents_df["subject_type"] == subject_type]
    mean_skill = skills_subset[SKILL_COLS].values.mean(axis=0)
    std_skill = skills_subset[SKILL_COLS].values.std(axis=0)
    expert_skill_threshold = mean_skill + std_skill
    if is_crowd:
        expert_skills = skills_subset[
            (skills_subset[SKILL_COLS] > expert_skill_threshold).all(axis=1)
        ][SKILL_COLS].values
    else:
        expert_skills = skills_subset[SKILL_COLS].values

    expert_probs = irt_prob_func(expert_skills, diff, rels).T
    dev = expert_probs - expert_probs.mean(axis=1, keepdims=True)
    return np.abs(dev).mean(axis=1)


# %%


def mad_mean(x):
    mean = np.mean(x)
    absolute_deviations = np.abs(x - mean)
    mad = np.mean(absolute_deviations)
    return mad


def mad_median(x):
    median = np.median(x)
    absolute_deviations = np.abs(x - median)
    mad = np.median(absolute_deviations)
    return mad


def percieved_diff(dataframe_dict, subject_type: str):
    # returns a vector of size n_human_agents

    agents_df = dataframe_dict["agents"]
    questions_df = dataframe_dict["questions"]

    skills_h = agents_df[agents_df["subject_type"] == subject_type][SKILL_COLS].values
    diff = questions_df[DIFF_COLS].values
    rels = questions_df[REL_COLS].values

    # (n_agents, n_items)
    # TODO: Check if this is correct
    logits = irt_logit_func(skills_h, diff, rels)

    return logits.mean(axis=-1)


def human_diff(dataframe_dict):
    return percieved_diff(dataframe_dict, "human")


def ai_diff(dataframe_dict):
    return percieved_diff(dataframe_dict, "ai")


def compute_advscore_v1(margin, kappa, delta):
    return margin * (kappa + delta)


def compute_advscore_v2(margin, kappa, delta):
    bonus_terms = kappa + delta
    return margin * (1 + bonus_terms)


def compute_advscore_v3(margin, kappa, delta):
    return margin * (1 + kappa) / (1 + delta)


def get_advscore_df(name, year, agg="weighted_mean", cumulative=False, is_crowd=True):
    dataframe_dict = load_dataframe_dict(name)
    human_prob_mean, human_std, human_probs = create_avg_prob(
        dataframe_dict, "human", ai_year=year, agg=agg, cumulative=cumulative
    )
    ai_prob_mean, ai_std, ai_probs = create_avg_prob(
        dataframe_dict, "ai", ai_year=year, agg=agg, cumulative=cumulative
    )
    margin_probs = human_probs - ai_probs
    kappa_iif_values = get_kappa_iif(dataframe_dict)
    kappa_disc_values = get_kappa_aggdisc(dataframe_dict)
    delta_h_values = compute_percieved_expert_diff(
        dataframe_dict, "human", is_crowd=is_crowd
    )

    mean_margin_prob = margin_probs.mean()
    mean_kappa_iif = kappa_iif_values.mean()
    mean_kappa_disc = kappa_disc_values.mean()

    mean_delta = delta_h_values.mean()

    data = [
        ["Human average probability", f"{human_prob_mean:.4f} ± {human_std:.4f}"],
        ["AI average probability", f"{ai_prob_mean:.4f} ± {ai_std:.4f}"],
        ["Mean of margin probabilities", f"{mean_margin_prob:.4f}"],
        ["agg_disc (kappa)", f"{mean_kappa_disc:.4f}"],
        ["agg_iif (kappa)", f"{mean_kappa_iif:.4f}"],
        ["mad of human_difficulty (delta)", f"{mean_delta:.4f}"],
    ]

    # print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
    # print("__________________________________")
    print(mean_margin_prob.shape, mean_kappa_iif.shape, mean_delta.shape)
    df = pd.DataFrame(
        {
            "mean_margin": mean_margin_prob,
            "kappa": mean_kappa_iif,
            "delta": mean_delta,
        },
        index=[name],
    )
    adv_scores_v1 = compute_advscore_v1(margin_probs, kappa_iif_values, delta_h_values)
    adv_scores_v2 = compute_advscore_v2(margin_probs, kappa_iif_values, delta_h_values)
    adv_scores_v3 = compute_advscore_v3(margin_probs, kappa_iif_values, delta_h_values)
    df["advscore_v1"] = adv_scores_v1.mean()
    df["advscore_v2"] = adv_scores_v2.mean()
    df["advscore_v3"] = adv_scores_v3.mean()
    return df, margin_probs, kappa_iif_values


def get_all_advscore_per_year(
    year: str, agg: str = "weighted_mean", cumulative: bool = False
):
    df_list = []
    for name in DATASET_NAMES:
        print("Dataset:", name)
        is_crowd = name != "trickme"
        df, margin_probs, kappa_values = get_advscore_df(
            name, year, agg=agg, cumulative=cumulative, is_crowd=is_crowd
        )
        df_list.append(df)

    advscore_df = pd.concat(df_list)
    return advscore_df, margin_probs


# %%
for dataset_name in DATASET_NAMES:
    _, margin_probs, kappa_values = get_advscore_df(
        dataset_name, "2024", agg="weighted_mean", cumulative=True
    )
    print(margin_probs.shape)
    plt.hist(margin_probs, bins=20, ec="black")
    plt.hist(kappa_values, bins=20, ec="black")
    scores_df = pd.DataFrame({"margin": margin_probs, "kappa": kappa_values})
    scores_df.to_csv(f"data/{dataset_name}_scores.csv")

# %%
# df = get_advscore_df("trickme", "2024", agg="weighted_mean", cumulative=True)
table1_df, _ = get_all_advscore_per_year("2024", agg="weighted_mean", cumulative=True)
table1_df


# %%
def plot_advscore_over_time(YEARS, DATASET_NAMES, get_all_advscore_per_year):
    # Create DataFrame for cumulative advscore
    DF = pd.DataFrame()
    for year in YEARS:
        year_df, _ = get_all_advscore_per_year(
            year, agg="weighted_mean", cumulative=True
        )
        year_df = year_df["advscore_v2"]
        DF = pd.concat([DF, year_df], axis=1)
    DF.columns = YEARS

    # Create the plot
    fig = go.Figure()

    # Plot cumulative data
    for dataset_name in DF.index:
        fig.add_trace(
            go.Scatter(
                x=YEARS,
                y=DF.loc[dataset_name],
                mode="lines+markers",
                name=dataset_name,
                line=dict(width=2),
                marker=dict(size=8),
            )
        )

    # Update layout to match ggplot2 theme
    fig.update_layout(
        template="ggplot2",
        font=dict(family="Arial", size=12),
        legend=dict(
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(240,240,240,0.8)",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=80, b=60),
        height=600,
        width=800,
        xaxis_title="Year",
        yaxis_title="Cumulative Advscore",
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.1)",
        title_standoff=15,
    )

    # Add a main title
    fig.add_annotation(
        text="Cumulative Advscore Over Time for Different Datasets",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=18, family="Arial", color="black"),
    )

    # Show the plot
    fig.show()


# %%
# Call the function
plot_advscore_over_time(YEARS, DATASET_NAMES, get_all_advscore_per_year)

# %%

name = "trickme"
dataframe_dict = load_dataframe_dict(name)

skills_df = dataframe_dict["agents"]
questions_df = dataframe_dict["questions"]
skills_df["subject_type"] = skills_df["subject_id"].map(
    lambda x: "ai"
    if get_agent_type(x) == "ai"
    else "expert"
    if x.isalpha()
    else "crowd"
)
print(skills_df)
# scatter plot for skills_df
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=questions_df,
    x="dif_0",
    y="dif_1",
    # hue="subject_type",
    palette="deep",
    alpha=0.7,
    s=50,
)
sns.scatterplot(
    data=skills_df,
    x="skill_0",
    y="skill_1",
    hue="subject_type",
    palette="Set2",
    alpha=0.7,
    s=50,
)
# %%
# Get the questions dataframe
questions_df = dataframe_dict["questions"]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot histogram for rel_0
ax1.hist(questions_df["rel_0"], bins=20, edgecolor="black")
ax1.set_title("Histogram of rel_0")
ax1.set_xlabel("rel_0")
ax1.set_ylabel("Frequency")

# Plot histogram for rel_1
ax2.hist(questions_df["rel_1"], bins=20, edgecolor="black")
ax2.set_title("Histogram of rel_1")
ax2.set_xlabel("rel_1")
ax2.set_ylabel("Frequency")

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# %%
skills_df.sort_values(by="accuracy", ascending=False)
# %%


# Responses -> 2PL IRT params

# Fit LLTM model to learn a feature coef vectors that convert
#  feature vectors to 2PL IRT params (Supervision: Responses)

# Table 1:
# %%
