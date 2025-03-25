import os
import json
import pandas as pd
import time
import csv
import re
from neo4j import GraphDatabase
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm



URI = "bolt://localhost:7687"
AUTH = ("neo4j", "root666")

language = 'es'
#language = 'en'

#debug = False
debug = False

#PARENTHESIS = re.compile(r'\([\s\S^\\(]*\)')
PARENTHESIS = re.compile(r"\([^()]*\)")

def get_concept_relations_syn(tx, sctid_value, relation, relation_text, concept_family, language):
    total_records = []

    if relation == 'ISA':
        query = "MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c2:ObjectConcept{active:'1'})-" \
                "[r2:ISA{active: '1'}]->(c1:ObjectConcept{sctid:$sctid, active:'1'}) - [r1: HAS_DESCRIPTION]->" \
                "(d1:Description{active:'1', languageCode:'en'}) " \
                "RETURN DISTINCT(c1['sctid'] + '|||' + trim(d2['term']) + '|||:ISA|||<relation>|||' + "\
                    "c2['sctid'] + '|||' + trim(d1['term']))"
    else:
        query = "MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c2:ObjectConcept{active:'1'})<-" \
                "[r2:ISA{active: '1'}]-(cr:RoleGroup)<-[r4:HAS_ROLE_GROUP]-(c1:ObjectConcept{sctid:$sctid, active:'1'})" \
                "-[r1:HAS_DESCRIPTION]->(d1:Description{active:'1', languageCode:'en'}) " \
                "RETURN DISTINCT(c1['sctid']+ '|||' + trim(d2['term']) + '|||:ISA|||<relation>|||' + " \
                    "c2['sctid'] + '|||' + trim(d1['term']))"

        query = query.replace(':ISA', ':' + relation)

    query = query.replace("languageCode:'en'", f"languageCode:'{language}'")
    query = query.replace('<relation>', relation_text)
    records = tx.run(query, sctid=sctid_value)


    for record in records:
        value = re.sub(PARENTHESIS, '', record[0])
        value = value.replace(' |||', '|||')
        total_records.append(value.strip())

    # Eliminar duplicados, a pesar del DISTINCT de la query en Neo4j pueden existir
    #  debido a la eliminación de las cadenas entre parentesis
    total_records = [*set(total_records)]
    return total_records


def get_snomed_neo(args):
    query_results = []
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        for arg in args:
            concept_id = arg[0]
            relations = arg[1]
            concept_family_map = arg[2]
            concept_family = arg[3]
            language = arg[4]

            for relation in relations:
                relation_text = concept_family_map[relation]['middle']
                query_result = driver.session().execute_read(get_concept_relations_syn,
                                                             concept_id, relation, relation_text, concept_family, language)
                if len(query_result) > 0:
                    query_results.append(query_result)

    return query_results


def process_concept(args):
    return get_snomed_neo(args)


if __name__ == '__main__':
    start = time.time()
    init_start = start
    print(f'reading prompting_map_{language}.json')
    with open(f'prompting_map_{language}.json', 'r', encoding='utf-8') as fin:
        concept_families_map = json.load(fin)

    print(time.time() - start)

    for concept_family, concept_family_map in concept_families_map.items():
        colnames = ['sctid', 'fsn']

        if debug:
            concept_family_file = 'data_concepts_test/' + concept_family.lower() + '.csv.gz'
        else:
            concept_family_file = 'data_concepts/' + concept_family.lower() + '.csv.gz'

        if os.path.exists(concept_family_file):
            print(f'reading concept_family_file: {concept_family_file}')
            start = time.time()
            concepts_df = pd.read_csv(concept_family_file, compression='gzip', sep=",", quotechar='"',
                                      skipinitialspace=True, encoding="utf-8", names=colnames, skiprows=1)
            print('read_csv: ' + str(time.time() - start))

            relations = list(concept_family_map.keys())
            results = []

            # PARALLEL
            if debug:
                processes = 1
            else:
                processes = cpu_count()
            pool = Pool(processes=processes)

            concept_list = concepts_df['sctid'].values.astype(str).tolist()
            args = [(concept, relations, concept_family_map, concept_family, language) for concept in concept_list]
            if len(args) // processes != 0:
                args = [args[i:i+len(args)//processes] for i in range(0, len(args), len(args) // processes)]
            else:
                args = []
            with tqdm(total=len(concept_list)) as pbar:
                for res in pool.imap_unordered(process_concept, args):
                    results.extend(res)
                    pbar.update()

            pool.close()
            pool.join()

            print(f'processed concepts: {len(results)}')

            concept_family = concept_family.lower()
            os.makedirs('output_corpus', exist_ok=True)
            csv_filename = f'output_corpus/output_corpus_{concept_family}.csv'

            lista_sin_repetidos = set()

            with open(csv_filename, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                for result in results:
                    for line in result:
                        tupla = tuple(line.split('|||')[1:3] + line.split('|||')[5:])
                        if tupla not in lista_sin_repetidos:
                            lista_sin_repetidos.add(tupla)
                            csv_writer.writerow(line.split('|||'))
                            #print(line.split('|||'))

    print('Complete process time: ' + str(time.time() - init_start))