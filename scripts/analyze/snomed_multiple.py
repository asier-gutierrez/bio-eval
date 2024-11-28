import argparse
import json
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bioeval.constants.general


def analyze_clm(input_files, model_names, output_path):
    model_score_dict = {}
    for input_file, model_name in zip(input_files, model_names):
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
        print(model_name, all_scores)

        # To log10
        df['score'] = df['score'].apply(np.log10)
        model_score_dict[model_name] = df['score']

    model_score_dict = collections.OrderedDict(
        sorted(model_score_dict.items(), reverse=True))

    fig, ax = plt.subplots()
    ax.boxplot(model_score_dict.values(), vert=False)
    ax.set_yticklabels(model_score_dict.keys())
    fig.tight_layout()
    plt.show()

def analyze_mlm(input_files, model_names, output_path):
    pass

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
