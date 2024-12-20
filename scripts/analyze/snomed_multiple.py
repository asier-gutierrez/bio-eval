import argparse
import json
import collections
import itertools
import os
import logging

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multicomp
import statsmodels.stats.multitest

import bioeval.constants.general


def report_statistics(df):
    group_scores = []
    group_names = []
    logger.info("=-"*30)
    logger.info("Group statistics")
    for group_name, group_df in df.groupby('model_name'):
        scores = group_df['score']
        group_scores.append(scores)
        group_names.append(group_name)
        mean1, median1 = np.mean(scores), np.median(scores)

        logger.info(f"Group {group_name}: Mean = {mean1}, Median = {median1}")

    data = np.concatenate(group_scores)
    groups = []
    for gn, gs in zip(group_names, group_scores):
        groups.extend([gn] * len(gs))
    # Perform Tukey's HSD assumes normality and equal variances. Pair-wise test.
    logger.info("=-"*30)
    logger.info("Tukey's HSD test")
    tukey = statsmodels.stats.multicomp.pairwise_tukeyhsd(data, groups, alpha=0.05)
    logger.info(tukey)

    # Perform Kruskal-Wallis test. One vs Rest test.
    logger.info("=-"*30)
    logger.info("Kruskal-Wallis test")
    h_stat, p_value = scipy.stats.kruskal(*group_scores)
    logger.info(f"H-statistic: {h_stat:.2f}, P-value: {p_value}")
    # Interpretation
    if p_value < 0.05:
        logger.info("There is a significant difference among the group medians.")
    else:
        logger.info("No significant difference among the group medians.")


    # Perform pairwise Mann-Whitney U tests. Pairwise test.
    logger.info("=-"*30)
    logger.info("Mann-Whitney U test")
    combs = list(itertools.combinations(zip(group_names, group_scores), r=2))
    pairs = []
    for comb in combs:
        pairs.append((comb[0][1], comb[1][1]))
    p_values = []
    for g1, g2 in pairs:
        _, p = scipy.stats.mannwhitneyu(g1, g2, alternative='two-sided')
        p_values.append(p)

    # Adjust p-values for multiple comparisons (Bonferroni correction)
    adjusted_p = statsmodels.stats.multitest.multipletests(p_values, method='bonferroni')[1]

    for comb, (i, p) in zip(combs, enumerate(adjusted_p)):
        if p_value < 0.05:
            txt = "There is a significant difference."
        else:
            txt = "No significant difference."
        logger.info(f"Comparison ({comb[0][0]}-{comb[1][0]}): Adjusted p-value = {p:.4f}, {txt}")


def analyze_clm(input_files, model_names, output_path):
    results = []
    for input_file, model_name in zip(input_files, model_names):
        # Gather the results form input file.

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                scores = data['scores']
                for score_pack in scores:
                    score = np.prod(score_pack)
                    results.append({
                        'model_name': model_name,
                        'score': score
                    })

        # Create the DataFrame.
        df = pd.DataFrame(results)
        df = df[df['model_name'] == model_name]

        # Analysis
        all_scores = {
            'min': df['score'].min(),
            'max': df['score'].max(),
            'avg': df['score'].mean(),
            'std': df['score'].std()
        }
        logger.info(model_name, all_scores)

    df = pd.DataFrame(results)
    report_statistics(df)
    df['model_name'] = df['model_name'].apply(lambda x: x.replace('/', '\n'))
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")

    # Plot the orbital period with horizontal boxes
    sns.violinplot(
        df, x='score', y='model_name', hue='model_name', width=.6, palette='husl'
    )
    ax.set_xticks(np.logspace(-100, 0, 5))
    ax.set(xlim=(10e-101, 0))

    # Add in points to show each observation
    #sns.stripplot(df, x='score', y='model_name', size=4, color=".3")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set_xlabel("Score (log)")
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    f.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_path, f"multiple.pdf"), dpi=300)


def analyze_mlm(input_files, model_names, output_path):
    results = []
    for input_file, model_name in zip(input_files, model_names):
        # Gather the results form input file.

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                generations = data['generations']
                for generation in generations:
                    score_pack = generation['score']
                    score = np.sum(score_pack)
                    results.append({
                        'model_name': model_name,
                        'score': np.exp(score)
                    })

        # Create the DataFrame.
        df = pd.DataFrame(results)
        df = df[df['model_name'] == model_name]

        # Analysis
        all_scores = {
            'min': df['score'].min(),
            'max': df['score'].max(),
            'avg': df['score'].mean(),
            'std': df['score'].std()
        }
        logger.info(model_name, all_scores)

    df = pd.DataFrame(results)
    report_statistics(df)
    df['model_name'] = df['model_name'].apply(lambda x: x.replace('/', '\n'))
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")

    # Plot the orbital period with horizontal boxes
    """
    sns.boxplot(
        df, x='score', y='model_name', hue='model_name',
        whis=[0, 100], width=.6, palette="vlag"
    )
    """
    sns.violinplot(
        df, x='score', y='model_name', hue='model_name', width=.6, palette='husl'
    )
    ax.set_xticks(np.logspace(-75, 0, 4))
    ax.set(xlim=(10e-76, 0))
    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set_xlabel("Score (log)")
    sns.despine(trim=True, left=True)
    f.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_path, f"multiple.pdf"), dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", action="append", type=str, required=True)
    parser.add_argument("--model_names", action="append", type=str, required=True)
    parser.add_argument('--type',
                        type=bioeval.constants.general.ModelType,
                        choices=list(bioeval.constants.general.ModelType),
                        required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    assert len(args.input_files) == len(args.model_names)

    os.makedirs(args.output_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.output_path, 'output.log'),
        encoding='utf-8',
        filemode="w",
        level=logging.INFO)

    if args.type == bioeval.constants.general.ModelType.CLM:
        analyze_clm(args.input_files, args.model_names, args.output_path)
    elif args.type == bioeval.constants.general.ModelType.MLM:
        analyze_mlm(args.input_files, args.model_names, args.output_path)
    else:
        raise NotImplementedError(f"{args.type} is not implemented yet.")
