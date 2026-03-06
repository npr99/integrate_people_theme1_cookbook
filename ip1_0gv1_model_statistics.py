"""
Statistical comparison and effect size analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_cohens_d(data1, data2, n_bootstrap=1000, random_state=42):
    """Calculate bootstrap 95% CI for Cohen's d."""
    np.random.seed(random_state)
    n1, n2 = len(data1), len(data2)

    cohens_d_bootstrap = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=n1, replace=True)
        sample2 = np.random.choice(data2, size=n2, replace=True)

        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(sample1, ddof=1) + (n2 - 1) * np.var(sample2, ddof=1))
            / (n1 + n2 - 2)
        )

        if pooled_std > 0:
            d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
            cohens_d_bootstrap.append(d)

    if len(cohens_d_bootstrap) > 0:
        ci_low = np.percentile(cohens_d_bootstrap, 2.5)
        ci_high = np.percentile(cohens_d_bootstrap, 97.5)
        return ci_low, ci_high

    return np.nan, np.nan


def calculate_comparison_statistics(data1, data2, n_bootstrap=1000, random_state=42):
    """Calculate comprehensive comparison statistics between two datasets."""
    if len(data1) == 0 or len(data2) == 0:
        return {
            "n1": len(data1),
            "n2": len(data2),
            "mean1": np.nan,
            "mean2": np.nan,
            "std1": np.nan,
            "std2": np.nan,
            "f_statistic": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "effect_size": "insufficient_data",
            "welch_p": np.nan,
            "levene_p": np.nan,
        }

    n1, n2 = len(data1), len(data2)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    std1 = np.std(data1, ddof=1)
    std2 = np.std(data2, ddof=1)

    f_statistic, p_value = stats.f_oneway(data1, data2)

    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1))
        / (n1 + n2 - 2)
    )

    if pooled_std > 0:
        cohens_d = (mean1 - mean2) / pooled_std

        ci_low, ci_high = bootstrap_cohens_d(data1, data2, n_bootstrap, random_state)

        if abs(cohens_d) < 0.2:
            effect_size_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_size_interpretation = "medium"
        else:
            effect_size_interpretation = "large"
    else:
        cohens_d = np.nan
        ci_low, ci_high = np.nan, np.nan
        effect_size_interpretation = "undefined"

    _, levene_p = stats.levene(data1, data2)

    if levene_p < 0.05:
        _, welch_p = stats.ttest_ind(data1, data2, equal_var=False)
    else:
        welch_p = np.nan

    return {
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "std1": std1,
        "std2": std2,
        "f_statistic": f_statistic,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_size": effect_size_interpretation,
        "welch_p": welch_p,
        "levene_p": levene_p,
    }


def compare_resolution_anova(
    hua_ari_df, species_name, percentiles=None, matched_cases_only=False, verbose=True
):
    """Compare 1km and 4km resolution values across percentiles."""
    if percentiles is None:
        percentiles = ["p90"]

    if isinstance(species_name, str):
        species_list = [species_name]
    else:
        species_list = list(species_name)

    all_results = []

    if verbose:
        print("\n=== MULTI-SPECIES RESOLUTION COMPARISON ANALYSIS ===")
        print(f"Species: {species_list}")
        print(f"Analysis type: {'Matched cases' if matched_cases_only else 'All available data'}")
        print(f"Percentiles: {percentiles}")
        print("Note: Zero values excluded as they represent areas outside the pollution model domain")

    for current_species in species_list:
        if verbose:
            print("\n" + "=" * 80)
            print(f"PROCESSING SPECIES: {current_species.upper()}")
            print("=" * 80)

        results_list = []

        for percentile in percentiles:
            col_1km = f"uifl_1km_{current_species}_{percentile}"
            col_4km = f"uifl_4km_{current_species}_{percentile}"

            if col_1km not in hua_ari_df.columns or col_4km not in hua_ari_df.columns:
                if verbose:
                    print(f"\nSkipping {current_species} {percentile}: One or both columns not found:")
                    print(f"  {col_1km}: {'Found' if col_1km in hua_ari_df.columns else 'NOT FOUND'}")
                    print(f"  {col_4km}: {'Found' if col_4km in hua_ari_df.columns else 'NOT FOUND'}")
                continue

            if matched_cases_only:
                valid_both = (
                    (hua_ari_df[col_1km] > 0)
                    & (hua_ari_df[col_4km] > 0)
                    & hua_ari_df[col_1km].notna()
                    & hua_ari_df[col_4km].notna()
                )

                analysis_df = hua_ari_df[valid_both][[col_1km, col_4km]].copy()
                data_1km = analysis_df[col_1km].values
                data_4km = analysis_df[col_4km].values

                sample_description = f"{len(data_1km):,} matched housing units"
            else:
                data_1km = hua_ari_df[hua_ari_df[col_1km] > 0][col_1km].dropna().values
                data_4km = hua_ari_df[hua_ari_df[col_4km] > 0][col_4km].dropna().values

                sample_description = (
                    f"1km: {len(data_1km):,}, 4km: {len(data_4km):,} housing units"
                )

            if verbose:
                print("\n" + "=" * 60)
                print(
                    f"ANALYSIS - {current_species.upper()} {percentile.upper()} 1KM vs 4KM RESOLUTION"
                )
                print("=" * 60)
                print(f"Sample: {sample_description}")

            stats_results = calculate_comparison_statistics(data_1km, data_4km)

            if verbose and stats_results["n1"] > 0 and stats_results["n2"] > 0:
                print(
                    f"\n--- Descriptive Statistics for {current_species} {percentile} by Resolution ---"
                )
                print(f"1km Resolution (n={stats_results['n1']:,}):")
                print(f"  Mean: {stats_results['mean1']:.4f} ppb")
                print(f"  Std:  {stats_results['std1']:.4f} ppb")

                print(f"4km Resolution (n={stats_results['n2']:,}):")
                print(f"  Mean: {stats_results['mean2']:.4f} ppb")
                print(f"  Std:  {stats_results['std2']:.4f} ppb")

                print("\n--- Statistical Results ---")
                print(
                    f"ANOVA F-statistic: {stats_results['f_statistic']:.4f}, p-value: {stats_results['p_value']:.6f}"
                )
                print(
                    f"Cohen's d: {stats_results['cohens_d']:.4f} ({stats_results['effect_size']})"
                )
                print(
                    f"Cohen's d 95% CI: [{stats_results['ci_low']:.4f}, {stats_results['ci_high']:.4f}]"
                )
                print(
                    f"Levene's p-value: {stats_results['levene_p']:.6f} ({'Unequal' if stats_results['levene_p'] < 0.05 else 'Equal'} variances)"
                )
                if not np.isnan(stats_results["welch_p"]):
                    print(f"Welch's t-test p-value: {stats_results['welch_p']:.6f}")

                alpha = 0.05
                if stats_results["p_value"] < alpha:
                    print(f"Result: Significant difference (p < {alpha})")
                    direction = "1km > 4km" if stats_results["cohens_d"] > 0 else "4km > 1km"
                    print(f"Direction: {direction}")
                else:
                    print(f"Result: No significant difference (p >= {alpha})")
            elif verbose:
                print(
                    f"Insufficient data for {current_species} {percentile}: 1km n={stats_results['n1']}, 4km n={stats_results['n2']}"
                )

            results_list.append(
                {
                    "species": current_species,
                    "percentile": percentile,
                    "matched_cases": matched_cases_only,
                    "n_1km": stats_results["n1"],
                    "n_4km": stats_results["n2"],
                    "mean_1km": stats_results["mean1"],
                    "mean_4km": stats_results["mean2"],
                    "f_statistic": stats_results["f_statistic"],
                    "p_value": stats_results["p_value"],
                    "cohens_d": stats_results["cohens_d"],
                    "ci_low": stats_results["ci_low"],
                    "ci_high": stats_results["ci_high"],
                    "effect_size": stats_results["effect_size"],
                    "welch_p": stats_results["welch_p"],
                }
            )

        all_results.extend(results_list)

        if verbose and len(results_list) > 0:
            species_df = pd.DataFrame(results_list)
            print("\n" + "=" * 80)
            print(f"SUMMARY TABLE - {current_species.upper()} RESOLUTION COMPARISON")
            print("=" * 80)
            display_cols = ["percentile", "n_1km", "n_4km", "cohens_d", "ci_low", "ci_high", "p_value"]
            print(species_df[display_cols].round(4).to_string(index=False))

    results_df = pd.DataFrame(all_results)

    if verbose and len(results_df) > 0:
        print("\n" + "=" * 100)
        print("OVERALL SUMMARY - ALL SPECIES RESOLUTION COMPARISON")
        print("=" * 100)
        for species in results_df["species"].unique():
            species_data = results_df[results_df["species"] == species]
            print(f"\n{species}:")
            display_cols = ["percentile", "cohens_d", "ci_low", "ci_high", "p_value", "effect_size"]
            print(species_data[display_cols].round(4).to_string(index=False))

    return results_df


def compare_groupby_anova(
    hua_ari_df,
    species_name,
    percentiles=None,
    groupby_variable="ownershp",
    resolution="1km",
    matched_cases_only=False,
    verbose=True,
):
    """Compare groups within a single resolution using ANOVA across percentiles."""
    if percentiles is None:
        percentiles = ["p90"]

    if isinstance(species_name, str):
        species_list = [species_name]
    else:
        species_list = list(species_name)

    if resolution not in ["1km", "4km"]:
        raise ValueError("Resolution must be '1km' or '4km'")

    if groupby_variable not in hua_ari_df.columns:
        raise ValueError(f"Groupby variable '{groupby_variable}' not found in DataFrame columns")

    all_results = []

    if verbose:
        print("\n=== MULTI-SPECIES GROUPBY COMPARISON ANALYSIS ===")
        print(f"Species: {species_list}")
        print(f"Resolution: {resolution}")
        print(f"Groupby variable: {groupby_variable}")
        print(f"Analysis type: {'Matched cases' if matched_cases_only else 'All available data'}")
        print(f"Percentiles: {percentiles}")
        print("Note: Zero values excluded as they represent areas outside the pollution model domain")

    group_values = hua_ari_df[groupby_variable].dropna().unique()
    group_values = sorted([x for x in group_values if pd.notna(x)])

    if len(group_values) < 2:
        raise ValueError(f"Need at least 2 groups in '{groupby_variable}', found: {group_values}")

    if verbose:
        print(f"Group values found: {group_values}")

    for current_species in species_list:
        if verbose:
            print("\n" + "=" * 80)
            print(f"PROCESSING SPECIES: {current_species.upper()}")
            print("=" * 80)

        results_list = []

        for percentile in percentiles:
            col_name = f"uifl_{resolution}_{current_species}_{percentile}"

            if col_name not in hua_ari_df.columns:
                if verbose:
                    print(f"\nSkipping {current_species} {percentile}: Column not found: {col_name}")
                continue

            group_data = {}
            for group_val in group_values:
                group_mask = hua_ari_df[groupby_variable] == group_val
                group_df = hua_ari_df[group_mask].copy()
                data = group_df[group_df[col_name] > 0][col_name].dropna().values
                group_data[group_val] = data

            if matched_cases_only:
                valid_indices = set()
                for group_val in group_values:
                    group_mask = hua_ari_df[groupby_variable] == group_val
                    valid_mask = group_mask & (hua_ari_df[col_name] > 0) & hua_ari_df[col_name].notna()
                    valid_indices.update(hua_ari_df[valid_mask].index)

                matched_group_data = {}
                for group_val in group_values:
                    group_mask = hua_ari_df[groupby_variable] == group_val
                    matched_mask = group_mask & hua_ari_df.index.isin(valid_indices)
                    matched_df = hua_ari_df[matched_mask]
                    data = matched_df[matched_df[col_name] > 0][col_name].dropna().values
                    matched_group_data[group_val] = data

                group_data = matched_group_data

            for i, group1 in enumerate(group_values):
                for group2 in group_values[i + 1 :]:
                    data1 = group_data[group1]
                    data2 = group_data[group2]

                    sample_description = (
                        f"Group {group1}: {len(data1):,}, Group {group2}: {len(data2):,} housing units"
                    )

                    if verbose:
                        print("\n" + "=" * 60)
                        print(f"ANALYSIS - {current_species.upper()} {percentile.upper()} {resolution.upper()}")
                        print(f"Comparing {groupby_variable}: {group1} vs {group2}")
                        print("=" * 60)
                        print(f"Sample: {sample_description}")

                    stats_results = calculate_comparison_statistics(data1, data2)

                    if verbose and stats_results["n1"] > 0 and stats_results["n2"] > 0:
                        print(
                            f"\n--- Descriptive Statistics for {current_species} {percentile} by {groupby_variable} ---"
                        )
                        print(f"Group {group1} (n={stats_results['n1']:,}):")
                        print(f"  Mean: {stats_results['mean1']:.4f} ppb")
                        print(f"  Std:  {stats_results['std1']:.4f} ppb")

                        print(f"Group {group2} (n={stats_results['n2']:,}):")
                        print(f"  Mean: {stats_results['mean2']:.4f} ppb")
                        print(f"  Std:  {stats_results['std2']:.4f} ppb")

                        print("\n--- Statistical Results ---")
                        print(
                            f"ANOVA F-statistic: {stats_results['f_statistic']:.4f}, p-value: {stats_results['p_value']:.6f}"
                        )
                        print(
                            f"Cohen's d: {stats_results['cohens_d']:.4f} ({stats_results['effect_size']})"
                        )
                        print(
                            f"Cohen's d 95% CI: [{stats_results['ci_low']:.4f}, {stats_results['ci_high']:.4f}]"
                        )
                        print(
                            f"Levene's p-value: {stats_results['levene_p']:.6f} ({'Unequal' if stats_results['levene_p'] < 0.05 else 'Equal'} variances)"
                        )
                        if not np.isnan(stats_results["welch_p"]):
                            print(f"Welch's t-test p-value: {stats_results['welch_p']:.6f}")

                        alpha = 0.05
                        if stats_results["p_value"] < alpha:
                            print(f"Result: Significant difference (p < {alpha})")
                            direction = (
                                f"Group {group1} > Group {group2}"
                                if stats_results["cohens_d"] > 0
                                else f"Group {group2} > Group {group1}"
                            )
                            print(f"Direction: {direction}")
                        else:
                            print(f"Result: No significant difference (p >= {alpha})")
                    elif verbose:
                        print(
                            f"Insufficient data for {current_species} {percentile}: Group {group1} n={stats_results['n1']}, Group {group2} n={stats_results['n2']}"
                        )

                    results_list.append(
                        {
                            "species": current_species,
                            "percentile": percentile,
                            "resolution": resolution,
                            "groupby_variable": groupby_variable,
                            "matched_cases": matched_cases_only,
                            "group1": group1,
                            "group2": group2,
                            "n_group1": stats_results["n1"],
                            "n_group2": stats_results["n2"],
                            "mean_group1": stats_results["mean1"],
                            "mean_group2": stats_results["mean2"],
                            "f_statistic": stats_results["f_statistic"],
                            "p_value": stats_results["p_value"],
                            "cohens_d": stats_results["cohens_d"],
                            "ci_low": stats_results["ci_low"],
                            "ci_high": stats_results["ci_high"],
                            "effect_size": stats_results["effect_size"],
                            "welch_p": stats_results["welch_p"],
                        }
                    )

        all_results.extend(results_list)

        if verbose and len(results_list) > 0:
            species_df = pd.DataFrame(results_list)
            print("\n" + "=" * 80)
            print(f"SUMMARY TABLE - {current_species.upper()} GROUPBY COMPARISON")
            print("=" * 80)
            display_cols = [
                "percentile",
                "group1",
                "group2",
                "n_group1",
                "n_group2",
                "cohens_d",
                "ci_low",
                "ci_high",
                "p_value",
            ]
            print(species_df[display_cols].round(4).to_string(index=False))

    results_df = pd.DataFrame(all_results)

    if verbose and len(results_df) > 0:
        print("\n" + "=" * 100)
        print("OVERALL SUMMARY - ALL SPECIES GROUPBY COMPARISON")
        print("=" * 100)
        for species in results_df["species"].unique():
            species_data = results_df[results_df["species"] == species]
            print(f"\n{species}:")
            display_cols = [
                "percentile",
                "group1",
                "group2",
                "cohens_d",
                "ci_low",
                "ci_high",
                "p_value",
                "effect_size",
            ]
            print(species_data[display_cols].round(4).to_string(index=False))

    return results_df


def plot_effect_sizes_by_percentile(
    *dataframes, programname=None, title_description="", figsize=(12, 8), save_plot=True
):
    """Plot Cohen's d effect sizes vs percentiles with confidence intervals."""
    if len(dataframes) == 0:
        print("Error: No DataFrames provided")
        return None

    combined_df = pd.concat(dataframes, ignore_index=True)

    def extract_percentile_number(percentile_str):
        try:
            return int(percentile_str.replace("p", ""))
        except Exception:
            return np.nan

    combined_df["percentile_numeric"] = combined_df["percentile"].apply(extract_percentile_number)
    combined_df = combined_df.dropna(subset=["percentile_numeric"])
    combined_df = combined_df.sort_values(["species", "matched_cases", "percentile_numeric"])

    unique_percentiles = sorted(combined_df["percentile_numeric"].unique())
    percentile_positions = {p: i for i, p in enumerate(unique_percentiles)}
    combined_df["x_position"] = combined_df["percentile_numeric"].map(percentile_positions)

    unique_species = combined_df["species"].unique()
    n_species = len(unique_species)

    if n_species > 1:
        n_cols = min(3, n_species)
        n_rows = (n_species + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0] * n_cols / 2, figsize[1] * n_rows / 2)
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_species > 1 else [axes]
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]

    colors = {"True": "#2E86AB", "False": "#A23B72"}
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for idx, species in enumerate(unique_species):
        if idx >= len(axes):
            break

        ax = axes[idx]
        species_data = combined_df[combined_df["species"] == species].copy()

        for j, (matched_cases, group_data) in enumerate(species_data.groupby("matched_cases")):
            color = colors.get(str(matched_cases), colors["False"])
            linestyle = linestyles[j % len(linestyles)]
            marker = markers[j % len(markers)]

            group_data = group_data.sort_values("percentile_numeric")

            x = group_data["x_position"]
            y = group_data["cohens_d"]

            valid_mask = ~(np.isnan(y))
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            if len(x_valid) > 0:
                label = f"{'Matched Cases' if matched_cases else 'All Available Data'}"
                ax.plot(
                    x_valid,
                    y_valid,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=6,
                    linewidth=2,
                    label=label,
                )

                if "ci_low" in group_data.columns and "ci_high" in group_data.columns:
                    ci_low = group_data.loc[valid_mask, "ci_low"]
                    ci_high = group_data.loc[valid_mask, "ci_high"]

                    ci_valid_mask = ~(np.isnan(ci_low) | np.isnan(ci_high))
                    if ci_valid_mask.any():
                        x_ci = x_valid[ci_valid_mask]
                        ci_low_valid = ci_low[ci_valid_mask]
                        ci_high_valid = ci_high[ci_valid_mask]

                        ax.fill_between(x_ci, ci_low_valid, ci_high_valid, color=color, alpha=0.2, linewidth=0)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.7)
        ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(y=-0.2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(y=-0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel("Percentile")
        ax.set_ylabel("Cohen's d (Effect Size)")
        ax.set_title(f"{species} - {title_description}", fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        percentile_labels = [f"p{int(p)}" for p in unique_percentiles]
        ax.set_xticks(range(len(unique_percentiles)))
        ax.set_xticklabels(percentile_labels)

        ax.text(
            0.02,
            0.98,
            "Effect Size Guide:\n±0.2: Small\n±0.5: Medium\n±0.8: Large",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
        )

    if n_species > 1:
        for idx in range(n_species, len(axes)):
            axes[idx].set_visible(False)

    if n_species == 1:
        main_title = "Resolution Effect Sizes Across Percentiles"
    else:
        main_title = "Resolution Effect Sizes Across Percentiles - Multiple Species"

    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    if save_plot and programname:
        if not os.path.exists(programname):
            os.makedirs(programname)

        species_names = "_".join(sorted(unique_species)).replace(" ", "_").lower()
        filename = f"{programname}_{species_names}_effect_sizes_by_percentile.png"
        filepath = os.path.join(programname, filename)

        fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Effect size plot saved to: {filepath}")

    return fig


def create_species_percentiles_comparison_plot(
    hua_ari_df, species_name, programname, figsize=(15, 8), save_plot=True, matched_cases_only=False
):
    """Create a combined box plot comparing 1km and 4km resolutions for all percentiles."""
    species_1km_cols = [col for col in hua_ari_df.columns if f"uifl_1km_{species_name}_p" in col]
    species_4km_cols = [col for col in hua_ari_df.columns if f"uifl_4km_{species_name}_p" in col]

    percentiles_1km = []
    percentiles_4km = []

    for col in species_1km_cols:
        parts = col.split("_")
        if len(parts) >= 4 and parts[3].startswith("p"):
            percentile = int(parts[3][1:])
            percentiles_1km.append((percentile, col))

    for col in species_4km_cols:
        parts = col.split("_")
        if len(parts) >= 4 and parts[3].startswith("p"):
            percentile = int(parts[3][1:])
            percentiles_4km.append((percentile, col))

    percentiles_1km.sort()
    percentiles_4km.sort()

    common_percentiles = []
    for p1, col1 in percentiles_1km:
        for p2, col2 in percentiles_4km:
            if p1 == p2:
                common_percentiles.append((p1, col1, col2))

    if len(common_percentiles) > 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        all_data_1km = []
        all_data_4km = []
        percentile_labels = []

        for percentile, col_1km, col_4km in common_percentiles:
            if matched_cases_only:
                valid_both = (
                    (hua_ari_df[col_1km] > 0)
                    & (hua_ari_df[col_4km] > 0)
                    & hua_ari_df[col_1km].notna()
                    & hua_ari_df[col_4km].notna()
                )
                data_1km = hua_ari_df[valid_both][col_1km]
                data_4km = hua_ari_df[valid_both][col_4km]
            else:
                data_1km = hua_ari_df[hua_ari_df[col_1km] > 0][col_1km].dropna()
                data_4km = hua_ari_df[hua_ari_df[col_4km] > 0][col_4km].dropna()

            all_data_1km.append(data_1km)
            all_data_4km.append(data_4km)
            percentile_labels.append(f"p{percentile}")

        n_percentiles = len(common_percentiles)
        positions_1km = np.arange(1, 2 * n_percentiles, 2)
        positions_4km = np.arange(2, 2 * n_percentiles + 1, 2)

        bp1 = ax.boxplot(
            all_data_1km,
            positions=positions_1km,
            widths=0.6,
            patch_artist=True,
            flierprops=dict(marker="o", alpha=0.5, markersize=2),
        )

        bp2 = ax.boxplot(
            all_data_4km,
            positions=positions_4km,
            widths=0.6,
            patch_artist=True,
            flierprops=dict(marker="o", alpha=0.5, markersize=2),
        )

        for box_item in bp1["boxes"]:
            box_item.set_facecolor("lightblue")
            box_item.set_edgecolor("blue")

        for box_item in bp2["boxes"]:
            box_item.set_facecolor("lightcoral")
            box_item.set_edgecolor("red")

        for median in bp1["medians"] + bp2["medians"]:
            median.set_color("darkred")
            median.set_linewidth(2)

        tick_positions = (positions_1km + positions_4km) / 2
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(percentile_labels)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightblue", edgecolor="blue", label="1km Resolution"),
            Patch(facecolor="lightcoral", edgecolor="red", label="4km Resolution"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_title(f"{species_name} Concentrations by Percentile and Resolution", fontsize=16, fontweight="bold")
        ax.set_xlabel("Percentiles", fontsize=12)
        ax.set_ylabel("Concentration (ppb)", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

        if len(common_percentiles) > 0:
            first_percentile, col_1km_first, col_4km_first = common_percentiles[0]

            if matched_cases_only:
                valid_both = (
                    (hua_ari_df[col_1km_first] > 0)
                    & (hua_ari_df[col_4km_first] > 0)
                    & hua_ari_df[col_1km_first].notna()
                    & hua_ari_df[col_4km_first].notna()
                )
                matched_count = valid_both.sum()
                note_text = (
                    f"Note: Matched cases only - {matched_count:,} housing units with valid data in both resolutions"
                )
            else:
                data_1km_total = hua_ari_df[hua_ari_df[col_1km_first] > 0][col_1km_first].dropna()
                data_4km_total = hua_ari_df[hua_ari_df[col_4km_first] > 0][col_4km_first].dropna()
                note_text = (
                    f"Note: 1km resolution - {len(data_1km_total):,} housing units; 4km resolution - {len(data_4km_total):,} housing units"
                )

            ax.text(
                0.5,
                -0.12,
                note_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
            )

        plt.tight_layout()
        plt.show()

        if save_plot:
            if not os.path.exists(programname):
                os.makedirs(programname)

            suffix = "_matched" if matched_cases_only else ""
            filename = f"{programname}_{species_name.lower()}_percentiles_comparison{suffix}.png"
            filepath = os.path.join(programname, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")

        return fig

    return None
