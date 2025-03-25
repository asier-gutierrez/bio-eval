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

CHECK_MARK = 'CHECKMARK'
def report_statistics(df):
    batch_results = collections.defaultdict(dict)

    for file_name, file_group_df in df.groupby('file'):
        group_scores = []
        group_names = []
        group_means = []
        for model_name, file_model_group_df in file_group_df.groupby('model_name'):
            scores = file_model_group_df['score']
            group_scores.append(scores)
            group_names.append(model_name)
            group_means.append(np.mean(scores))
            batch_results[model_name][file_name] = ''
            batch_results[model_name]['model_name'] = model_name

        best_result = batch_results[group_names[np.argmax(group_means)]]
        best_result[file_name] = CHECK_MARK

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

        checks = 0
        for comb, (i, p) in zip(combs, enumerate(adjusted_p)):
            if best_result['model_name'] in [comb[0][0], comb[1][0]]:
                if p < 0.05:
                    checks += 1
        if checks == len(batch_results) - 1:
            best_result[file_name] += CHECK_MARK
    df = pd.DataFrame(batch_results.values())
    cols = list(df.columns)
    cols.remove('model_name')
    df = df[['model_name'] + sorted(cols)]
    df = df.rename(columns={"model_name": "Model"})
    return df


def analyze_clm(input_files, model_names, output_path):
    results = []
    for input_file, model_name in zip(input_files, model_names):
        # Gather the results form input file.

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                scores = data['scores']
                file = data['file']
                file = file.replace('.csv', '').replace('output_corpus_', '').replace('_', ' ').title()
                for score_pack in scores:
                    score = np.prod(score_pack)
                    results.append({
                        'model_name': model_name,
                        'file': file,
                        'score': score
                    })

    df = pd.DataFrame(results)
    df['model_name'] = df['model_name'].apply(lambda x: x.replace('/', '\n'))
    stats = report_statistics(df)
    stats.to_html(os.path.join(output_path, 'multiple_stats.html'), index=False)


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
                    file = data['file']
                    file = file.replace('.csv', '').replace('output_corpus_', '').replace('_', ' ').title()
                    results.append({
                        'model_name': model_name,
                        'score': np.exp(score),
                        'file': file
                    })

    df = pd.DataFrame(results)
    df['model_name'] = df['model_name'].apply(lambda x: x.replace('/', '\n'))
    stats = report_statistics(df)
    stats.to_html(os.path.join(output_path, 'multiple_stats.html'), index=False)


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
