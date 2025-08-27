import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def explore_feature(
    df: pd.DataFrame,
    target: str,
    feature: str,
    class_names: tuple[str, str] | None = None,
    kde: bool = True,
    bins: int = 30,
    palette: str = "coolwarm",
    show: bool = True,
):
    """
    Explore a single feature against a target variable.

    - If `feature` is numeric: prints group-by summary, draws boxplot + KDE/Histogram.
    - If `feature` is categorical: prints counts, draws countplot by target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data.
    target : str
        Target column name (binary or categorical).
    feature : str
        Feature column to explore.
    class_names : tuple[str, str] | None
        Custom display names for target classes. If None, inferred from data.
    kde : bool
        If True (numeric only), plots KDE; else histogram.
    bins : int
        Number of bins if histogram is used.
    palette : str
        Color palette for plots.
    show : bool
        Whether to display the plots.

    Returns
    -------
    pd.DataFrame
        Summary statistics (numeric: describe; categorical: counts).
    """
    if target not in df.columns or feature not in df.columns:
        raise KeyError(f"Missing columns: {[c for c in [target, feature] if c not in df.columns]}")

    uniq = pd.unique(df[target].dropna())
    if class_names is None:
        class_names = tuple(str(u) for u in uniq)

    is_numeric = pd.api.types.is_numeric_dtype(df[feature])

    if is_numeric:
        # --- Summary ---
        summary = df.groupby(target, dropna=False)[feature].describe()
        print(summary)

        # --- Boxplot ---
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=target, y=feature, data=df, palette=palette)
        plt.title(f"{feature} by {target}")
        if class_names and len(class_names) == 2:
            plt.xticks([0, 1], class_names)
        if show: plt.show()

        # --- KDE or Histogram ---
        plt.figure(figsize=(6, 4))
        if kde:
            sns.kdeplot(data=df, x=feature, hue=target, fill=True,
                        common_norm=False, palette=palette, alpha=0.5)
        else:
            for u, name in zip(uniq, class_names):
                sns.histplot(df[df[target] == u][feature], bins=bins,
                             stat="density", element="step", label=name, alpha=0.4)
            plt.legend()
        plt.title(f"Distribution of {feature} by {target}")
        if show: plt.show()

        return summary

    else:
        # --- Categorical ---
        counts = df.groupby([target, feature]).size().reset_index(name="count")
        print(counts)

        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x=feature, hue=target, palette=palette)
        plt.title(f"{feature} counts by {target}")
        plt.xticks(rotation=30, ha="right")
        if show: plt.show()

        return counts
