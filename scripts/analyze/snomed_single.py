import argparse
import json
import collections
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import bioeval.constants.general


plt.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'


def analyze_clm(input_file, output_path):
    # Gather the results form input file.
    results = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            file = data['file']
            file = file.replace('.csv', '').replace('output_corpus_', '').replace('_', ' ').title()
            scores = data['scores']
            for score_pack in scores:
                score = np.prod(score_pack)
                results.append({
                    'file': file,
                    'score': score
                })

    # Create the DataFrame.
    df = pd.DataFrame(results)

    # Analysis
    all_scores = {
        'min': df['score'].min(),
        'max': df['score'].max(),
        'avg': df['score'].mean(),
        'std': df['score'].std()
    }
    print(all_scores)

    # To log10
    df['score'] = df['score'].apply(np.log)

    group = {k: v['score'] for k, v in df.groupby('file')}
    group = collections.OrderedDict(sorted(group.items(), reverse=True))

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.boxplot(group.values(), vert=False)
    ax.set_yticklabels(group.keys())
    ax.set_xticks(list(reversed([-n for n in range(0, 700, 100)])))
    ax.set_xlabel('Score (log)')
    fig.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_path, f"single.pdf"), dpi=300)

def analyze_mlm(input_file, output_path):
    # Gather the results form input file.
    results = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            file = data['file']
            file = file.replace('.csv', '').replace('output_corpus_', '').replace('_', ' ').title()
            generations = data['generations']
            for generation in generations:
                score_pack = generation['score']
                score = np.sum(score_pack)
                results.append({
                    'file': file,
                    'score': np.exp(score)
                })

    # Create the DataFrame.
    df = pd.DataFrame(results)

    # Analysis
    all_scores = {
        'min': df['score'].min(),
        'max': df['score'].max(),
        'avg': df['score'].mean(),
        'std': df['score'].std()
    }
    print(all_scores)

    # To log
    df['score'] = df['score'].apply(np.log)

    group = {k: v['score'] for k, v in df.groupby('file')}
    group = collections.OrderedDict(sorted(group.items(), reverse=True))

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.boxplot(group.values(), vert=False)
    ax.set_yticklabels(group.keys())
    ax.set_xlabel('Score (log)')
    fig.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_path, f"single.pdf"), dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument('--type',
                        type=bioeval.constants.general.ModelType,
                        choices=list(bioeval.constants.general.ModelType),
                        required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    if args.type == bioeval.constants.general.ModelType.CLM:
        analyze_clm(args.input_file, args.output_path)
    elif args.type == bioeval.constants.general.ModelType.MLM:
        analyze_mlm(args.input_file, args.output_path)
    else:
        raise NotImplementedError(f"{args.type} is not implemented yet.")
