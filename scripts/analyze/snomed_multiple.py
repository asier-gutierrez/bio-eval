import argparse
import json
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multicomp

import bioeval.constants.general


def report_statistics(df):
    group_scores = []
    group_names = []
    for group_name, group_df in df.groupby('model_name'):
        scores = group_df['score']
        group_scores.append(scores)
        group_names.append(group_name)
        mean1, median1 = np.mean(scores), np.median(scores)

        print(f"Group {group_name}: Mean = {mean1}, Median = {median1}")

    data = np.concatenate(group_scores)
    groups = []
    for gn, gs in zip(group_names, group_scores):
        groups.extend([gn] * len(gs))
    # Perform Tukey's HSD
    tukey = statsmodels.stats.multicomp.pairwise_tukeyhsd(data, groups, alpha=0.05)
    print(tukey)

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
        print(model_name, all_scores)

    df = pd.DataFrame(results)
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

    # Add in points to show each observation
    #sns.stripplot(df, x='score', y='model_name', size=4, color=".3")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    f.tight_layout()
    plt.show()
    report_statistics(df)

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
        print(model_name, all_scores)

    df = pd.DataFrame(results)
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

    # Add in points to show each observation
    #sns.stripplot(df, x='score', y='model_name', size=4, color=".3")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    f.tight_layout()
    plt.show()
    report_statistics(df)

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

    if args.type == bioeval.constants.general.ModelType.CLM:
        analyze_clm(args.input_files, args.model_names, args.output_path)
    elif args.type == bioeval.constants.general.ModelType.MLM:
        analyze_mlm(args.input_files, args.model_names, args.output_path)
    else:
        raise NotImplementedError(f"{args.type} is not implemented yet.")
