# SNOMED-CT Corpus download and preparation.

## 1. Download SNOMED-CT
Download the SNOMED-CT version from https://www.nlm.nih.gov/healthit/snomedct/international.html


## 2. Install the Neo4J Community 4.4.15 database.
Change the configuration file
```
#dbms.directories.import=import  <- comment this one
dbms.memory.heap.initial_size=25500m
dbms.memory.heap.max_size=25500m
dbms.memory.pagecache.size=35500m
dbms.jvm.additional=-XX:+ExitOnOutOfMemoryError
dbms.security.allow_csv_import_from_file_urls=true
```

Optional but highly recommended: `dbms.security.auth_enabled` to `false`

## 3. Load SNOMED-CT in Neo4J
**For Spanish, we need to load English version first and then Spanish, so Section 3.1, and then Section 3.2.**

Create a virtualenv or conda env and `pip install py2neo monotonic packaging`

### 3.1 For English

```
cd scripts/snomed_build_en
python snomed_g_graphdb_build_tools.py db_build \
    --action create \
    --rf2 <path>/SnomedCT_InternationalRF2_PRODUCTION_20230228T120000Z/Full \
    --release_type full \
    --neopw <neo-password> \
    --output_dir ../../extraction/snomed_en/
```


### 3.2 For Spanish

First rename the Refset, by replacing "Language" by "SpanishExtension" like in `der2_cRefset_LanguageSpanishExtensionFull-es_INT_20221031.txt` by `der2_cRefset_SpanishExtensionSpanishExtensionFull-es_INT_20221031.txt`.

```
cd scripts/snomed_build_es
python snomed_g_graphdb_build_tools.py db_build \
    --action create \
    --rf2 <path>/SnomedCT_SpanishRelease-es_PRODUCTION_20240930T120000Z/Full \
    --release_type full \
    --neopw <neo-password> \
    --output_dir ../../extraction/snomed_es_new/ \
    --language_name SpanishExtension \
    --language_code es
```

## 4. Load output to the Database.
Scripts in Section 3 will already load on the database. It is a strong suggestion to remove that database and create a new one.

We are going to be loading the `build.cypher` files.

1. Edit English file. Replace "CREATE CONSTANT" by "CREATE CONSTANT IF NOT EXISTS"
2. Run the following command `cat build.cypher | cypher-shell`
3. Edit Spanish file. Replace "CREATE CONSTANT" by "CREATE CONSTANT IF NOT EXISTS" and comment all the "CREATE INDEX" lines.
4. Run the following command `cat build.cypher | cypher-shell`

## 5. Extract SNOMED-CT files.
Go to `scripts/snomed_queries`, run `run_queries.sh`
And then gzip everything in the folder `gzip *`.

# 6. Generate the corpus.
Go to `scripts/snomed_corpus` run `python generate_corpus.py` (modify the variables as needed)

# 7. Group synonyms
Go to `scripts/snomed_corpus` run `python group_corpus.py` (modify the variables as needed)