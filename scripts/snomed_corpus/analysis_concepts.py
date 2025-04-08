import json

def calculate_sum_of_products(file_path):
    total_sum = 0
    total_lines = 0

    with open(file_path, 'r') as file:
        for line in file:
            total_lines += 1
            data = json.loads(line)
            # Assuming the two fields are named 'field1' and 'field2'
            field1 = data.get('a_concepts', [])
            field2 = data.get('b_concepts', [])
            
            # Calculate the product of the lengths of the two fields
            total_sum += len(field1) * len(field2)
    
    return total_sum, total_lines

if __name__ == "__main__":
    file_path = "./output_corpus_json/output_corpus_es.jsonl"
    prod, lines = calculate_sum_of_products(file_path)
    print("Total products calculated:", prod)
    print("Total lines processed:", lines)