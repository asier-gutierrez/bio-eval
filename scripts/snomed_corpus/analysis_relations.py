import json
import collections


def calculate_sum_of_products(file_path):
    all_dict = collections.defaultdict(int)

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Assuming the two fields are named 'field1' and 'field2'
            field1 = data.get('a_concepts', [])
            field2 = data.get('b_concepts', [])
            
            # Calculate the product of the lengths of the two fields
            all_dict[data.get('file', None)] += len(field1) * len(field2)
    
    return all_dict

if __name__ == "__main__":
    file_path = "./output_corpus_json/output_corpus_en.jsonl"
    all_dict = calculate_sum_of_products(file_path)
    print("Relations obtained:")
    all_list = []
    for k, v in all_dict.items():
        all_list.append([k.replace('.csv', '').replace('output_corpus_', ''), v])
    for k, v in sorted(all_list, key=lambda x: x[0]):
        print(k, v)