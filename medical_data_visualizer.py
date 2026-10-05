import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("medical_examination.csv")

df["overweight"] = (df["weight"] / (df["height"] / 100) ** 2 > 25).astype(int)
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


def draw_cat_plot():
    """Create the categorical bar plot grouped by cardio outcome."""
    global df_cat

    df_cat = df.melt(
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
        var_name="variable",
        value_name="value",
    )

    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"], as_index=False)
        .size()
        .rename(columns={"size": "total"})
    )

    df_cat["value"] = df_cat["value"].map({0: "good", 1: "bad"})

    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
        height=6,
        aspect=1.5,
        order=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
        hue_order=["good", "bad"],
    )

    fig.set_axis_labels("", "count")
    fig.set_titles(col_template="cardio = {col_name}")
    return fig


def draw_heat_map():
    """Create the correlation heat map after removing invalid patient measurements."""
    global df_heat, corr, mask

    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ].copy()

    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    return fig
