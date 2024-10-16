# %%
import importlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logreg

# %%
advtype_df = pd.read_csv("data/AdvQA_advtype_annots.csv")
feat_cols = advtype_df.columns[4:]
advtype_feat_df = advtype_df[feat_cols].astype(float)
new_feat_cols = []
for col in feat_cols:
    new_col = "Advtype_" + col.lower().replace(" / ", "/").replace(" ", "_")
    new_feat_cols.append(new_col)
advtype_feat_df.columns = new_feat_cols

advtype_feat_df

# %%
features_df = pd.read_csv("data/features_df.csv", index_col=0)
# Convert boolean columns to float
bool_columns = features_df.select_dtypes(include=["bool"]).columns
features_df[bool_columns] = features_df[bool_columns].astype(float)

# Display the updated DataFrame
cat_feat_cols = [c for c in features_df.columns if c.startswith("category_")]
adv_feat_cols = [c for c in features_df.columns if c.startswith("Advtype_")]
# features_df = features_df[cat_feat_cols]
advtype_feat_df = features_df[adv_feat_cols]
features_df = pd.concat([advtype_feat_df, features_df[cat_feat_cols]], axis=1)
# %%

labels_df = pd.read_csv("data/advqa_combined_scores.csv", index_col=0)
labels_df["adv_score"] = labels_df["margin"] * (1 + labels_df["kappa"])
labels_df["kappa_bin"] = (labels_df["kappa"] > 0.51).astype(float)
labels_df["margin_bin"] = (labels_df["margin"] > labels_df["margin"].median()).astype(
    float
)
labels_df["adv_bin"] = (labels_df["adv_score"] > labels_df["adv_score"].mean()).astype(
    float
)

feat_cols = features_df.columns
labels_df
print("Binary Labels distribution:")
print(labels_df["kappa_bin"].value_counts())
print(labels_df["margin_bin"].value_counts())
print(labels_df["adv_bin"].value_counts())

# %%

# Combine different features pair-wise
quad_feats = {}
for i, f1 in enumerate(features_df.columns):
    for j, f2 in enumerate(features_df.columns[i + 1 :], start=i + 1):
        values = features_df[f1] * features_df[f2]
        if values.equals(features_df[f1]) or values.equals(features_df[f2]):
            continue
        if values.sum() == 0:
            continue
        new_col = f"{f1} | {f2}"
        quad_feats[new_col] = values

quad_feats_df = pd.DataFrame(quad_feats)
quad_feats_df = pd.concat([features_df, quad_feats_df], axis=1)

print("# Quadratic Features:", len(quad_feats_df.columns))
print(quad_feats_df.columns)
# %%


importlib.reload(logreg)

fit, results_df_adv = logreg.log_regression_analysis(
    features_df, labels_df["adv_bin"], C=5.0
)

print("Model fit:", fit)
results_df_adv = results_df_adv[results_df_adv["coef"] != 0]
results_df_adv
# %%
# fit, results_df_advscore = logreg.linear_regression_with_significance(
#     features_df, labels_df["adv_score"], alpha=0.0001, fit_intercept=False
# )

# print("Model fit:", fit)
# results_df_advscore = results_df_advscore[results_df_advscore["coef"] != 0]
# results_df_advscore
# %%
# fit, results_df_margin = logreg.log_regression_analysis(
#     quad_feats_df, labels_df["margin_bin"], C=1.0
# )
# results_df_margin = results_df_margin[results_df_margin["sig"] != ""]

# print("Model fit:", fit)

# %%

# _, results_df_kappa = logreg.log_regression_analysis(features_df, labels_df["kappa"])
# _, results_df_margin = logreg.log_regression_analysis(features_df, labels_df["margin"])

# results_df_kappa[results_df_kappa["sig"] != ""]
# results_df_margin[results_df_margin["sig"] != ""]

# # Create dataframes for each output label and input feature types
# kappa_df = results_df_kappa["coef"]
# margin_df = results_df_margin["coef"]

# %%


# Split each dataframe by feature type and remove prefixes
def split_and_clean_df(df, prefix):
    subset_df = df.loc[df.index.str.startswith(f"{prefix}_")]
    subset_df.index = subset_df.index.str.removeprefix(f"{prefix}_")
    subset_df = subset_df.sort_values(by="coef", ascending=False)
    return subset_df


cat_df = split_and_clean_df(results_df_adv, "category")
advtype_df = split_and_clean_df(results_df_adv, "Advtype")
# %%


def create_bar_subplot(df, feat_type):
    if feat_type == "Cat":
        marker_color = "#85C1E9"
    else:
        marker_color = "#F1948A"
    labels = df.index.map(lambda x: f"<i>{x}</i>")
    return go.Bar(
        y=labels,
        x=df.values,
        orientation="h",
        marker_color=marker_color,
        showlegend=False,
    )


# Create subplots
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "<b><span style='font-size: 18px; font-family: Roboto;'>Categories</span></b>",
        "<b><span style='font-size: 18px; font-family: Roboto;'>Adversarial Tactics</span></b>",
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
    shared_xaxes=True,
)
# Add traces for each subplot
fig.add_trace(create_bar_subplot(cat_df["coef"], "Cat"), row=1, col=1)
fig.add_trace(create_bar_subplot(advtype_df["coef"], "Advtype"), row=2, col=1)

# Update layout to match ggplot2 theme and reflect horizontal layout
fig.update_layout(
    template="ggplot2",
    font={"size": 12, "family": "Roboto"},
    margin={"l": 10, "r": 10, "t": 30, "b": 10},
    height=500,
    width=700,
)

# Update axes for horizontal layout
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    dtick=0.25,
    tickfont={"size": 14, "family": "Roboto"},
)
fig.update_xaxes(
    title_font={"size": 25, "family": "Roboto"},
    title_text="Logistic Regression Coefficients",
    row=2,
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    tickfont={"size": 18, "family": "Roboto"},
)

# Show the plot
fig.show()
fig.write_image("figs/logreg_coefs.pdf")
# %%

features_df.sum(axis=0)
# %%

# check which two columns are are same
for i in range(len(features_df.columns)):
    for j in range(i + 1, len(features_df.columns)):
        if features_df[features_df.columns[i]].equals(
            features_df[features_df.columns[j]]
        ):
            print(features_df.columns[i], features_df.columns[j])
# %%
features_df


# %%
margin_df = labels_df[["margin"]]
# %%
print(margin_df.describe())
# %%
from tabulate import tabulate

table_data = []
for feat_name in features_df.columns:
    idx = features_df[features_df[feat_name] == 1].index
    metrics = margin_df.loc[idx].describe()["margin"]
    table_data.append([feat_name, metrics["mean"], metrics["std"]])

headers = ["Feature", "Mean", "Std Dev"]
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))


# %%
labels_df["adv_score"] = labels_df["margin"] * (1 + labels_df["kappa"])
import matplotlib.pyplot as plt

plt.hist(labels_df["adv_score"])

# %%
table_data = []
for feat_name in features_df.columns:
    idx = features_df[features_df[feat_name] == 1].index
    metrics = margin_df.loc[idx].describe()["margin"]
    table_data.append([feat_name, metrics["count"]])

headers = ["Feature", "Count"]
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

# %%
