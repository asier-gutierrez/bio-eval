import os
import json
import math
import re

import pandas as pd
from tqdm import tqdm


def concept_lowercase(v):
    v = str(v)
    if re.match('^[A-Z]{2,}', v):
        return v
    else:
        return v[0].lower() + v[1:]


def concept_uppercase(v):
    v = str(v)
    if re.match('^[A-Z]{2,}', v):
        return v
    else:
        return v[0].upper() + v[1:]


if __name__ == '__main__':
    base_path = './output_corpus_es/'
    output_path = './output_corpus_json/output_corpus_es.jsonl'
    columns = ['a_concept_id', 'a_concept', 'relation_code', 'relation', 'b_concept_id', 'b_concept']
    df = None
    for file in os.listdir(base_path):
        print(file)
        df_in = pd.read_csv(os.path.join(base_path, file), sep=';', header=None, names=columns)
        df_in['file'] = file
        if df is None:
            df = df_in
        else:
            df = pd.concat([df, df_in], axis=0)
    output_data = []
    for name, group in tqdm(df.groupby(['a_concept_id', 'relation_code'])):
        d = {'a_concept_id': str(name[0]), 'a_concepts': [], 'relation': '', 'relation_code': name[1], 'b_concepts': [],
             'b_concept_ids': [], 'file': group.iloc[0]['file']}
        for idx, (_, row) in enumerate(group.iterrows()):
            if idx == 0:
                d['relation'] = row['relation']
            if type(row['a_concept']) == str or (type(row['a_concept']) != str and not math.isnan(row['a_concept'])):
                d['a_concepts'].append(concept_uppercase(row['a_concept']))
            if type(row['b_concept']) == str or (type(row['b_concept']) != str and not math.isnan(row['b_concept'])):
                d['b_concepts'].append(concept_lowercase(row['b_concept']))
                d['b_concept_ids'].append(row['b_concept_id'])
        d['a_concepts'] = list(set(d['a_concepts']))
        d['b_concepts'] = list(set(d['b_concepts']))
        d['b_concept_ids'] = list(set(d['b_concept_ids']))
        output_data.append(d)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for d in output_data:
            fout.write(json.dumps(d))
            fout.write('\n')