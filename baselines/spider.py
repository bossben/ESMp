from llm import gptcall,claudecall
from load_data import get_json_files
from tqdm import tqdm
import sqlite3
from nltk import ngrams
from nltk.metrics.distance import jaccard_distance

spider = get_json_files('spider/test_data/dev.json')
databases = get_json_files('spider/test_data/tables.json')


total = len(spider)

with tqdm(total=total, desc="Progress") as pbar:
    for id_no in range(total):
        database_id = spider[id_no]['db_id']
        conn = sqlite3.connect(f'spider/test_database/{database_id}/{database_id}.sqlite')
        c = conn.cursor()

        #character level
        def generate_ngrams_characters(text, n=3):
            return set(ngrams(text.lower(), n))
        #word level
        def generate_ngrams(text, n=3):
            return set(ngrams(text.lower().split(), n))

        def calculate_similarity(ngrams1, ngrams2):
            return 1 - jaccard_distance(ngrams1, ngrams2)
        def get_db(dbid):
            for i in databases:
                if i['db_id'] == dbid:
                    return i
        database = get_db(database_id)


        utterance = spider[id_no]['question']
        target_sql = spider[id_no]['query']
        def makeSchema(db): # Making the schema for the database
            schema = []
            table_no = 0
            for table_name in db['table_names_original']:
                schema += [table_name, ':']
                for col in [i for i in db['column_names_original'] if i[0] == table_no]:
                    schema.append(col[1])
                    schema.append(',')
                schema = schema[:-1]
                schema.append('|')
                table_no += 1
            return schema
        
        def parse_schema(schema,input_question):
            tables = " ".join(schema).split(" | ")
            formatted_tables = []
            formatted_tables_no_vals = []
            tablecount = 0
            for table in tables:
                tablecount += 1
                parts = table.split()
                table_name = parts[0]
                columns = parts[2::2]  # Skip ':', and ',' to only get column names
                cols_with_values = []
                colcount = 0
                for col in columns:
                    colcount += 1
                    try:
                        c.execute(f"SELECT {col} FROM {table_name}")
                        rows = c.fetchall()
                    except:
                        rows = []
                        best_match = "No data"
                    pbar.set_postfix_str(f"parsing schema, on table {tablecount} / {len(tables)}, column {colcount} / {len(columns)}, {len(rows)} rows", refresh=True)
                    highest_score_word = -1
                    highest_score_char = -1
                    best_match_word = ""
                    best_match_char = ""
                    for row in rows:
                        input_ngrams_word = generate_ngrams(input_question, 1)
                        input_ngrams_char = generate_ngrams_characters(input_question, 3)
                        database_value = str(row[0])
                        value_ngrams_word = generate_ngrams(database_value, 1)
                        value_ngrams_char = generate_ngrams_characters(database_value, 3)
                        similarity_score_word = calculate_similarity(input_ngrams_word, value_ngrams_word)
                        similarity_score_char = calculate_similarity(input_ngrams_char, value_ngrams_char)
                        if similarity_score_word > highest_score_word:
                            highest_score_word = similarity_score_word
                            best_match_word = database_value
                        if similarity_score_char > highest_score_char:
                            highest_score_char = similarity_score_char
                            best_match_char = database_value
                    if highest_score_word > 0:
                        best_match = best_match_word
                    else:
                        best_match = best_match_char
                    c.execute(f"SELECT l.name FROM pragma_table_info('{table_name}') as l WHERE l.pk > 0;")
                    pk =  [j.lower() for i in c.fetchall() for j in i]
                    c.execute('PRAGMA table_info({})'.format(table_name))
                    non_nulls = [i[1].lower() for i in c.fetchall() if i[3] == 1]
                    c.execute(f"PRAGMA index_list({table_name})")
                    indices = c.fetchall()
                    unique_columns = set()
                    for index in indices:
                        if index[2]:  # The unique flag is True
                            index_name = index[1]
                            c.execute(f"PRAGMA index_info({index_name})")
                            index_info = c.fetchall()
                            # Add all columns in this unique index to the set
                            if len(index_info) == 1:
                                unique_columns.update(info[2] for info in index_info)
                    unique_columns = {col.lower() for col in unique_columns}
                    non_null = False
                    unique = False
                    
                    if col.lower() in pk:
                        string = f"{col+'*'} [{best_match}]"
                    else:
                        string = f"{col} [{best_match}]"
                        if col.lower() in non_nulls:
                            non_null = True
                        if col.lower() in unique_columns:
                            unique = True
                    
                    if non_null:
                        string += " NON_NULL"
                    if unique:
                        string += " UNIQUE"
                    cols_with_values.append(string)
                c.execute(f"PRAGMA foreign_key_list({table_name})")
                fetchall = c.fetchall()
                for i in range(len(fetchall)):
                    if fetchall[i][4] == None:
                        c.execute(f"PRAGMA table_info({fetchall[i][2]})")
                        primary_keys = [col[1] for col in c.fetchall() if col[5] == 1]
                        assert len(primary_keys)==1, "Foreign key referencing more than one primary key or no primary keys"
                        lister = list(fetchall[i])
                        lister[4] = primary_keys[0]
                        fetchall[i] = tuple(lister)
                foreign_keys = {i[3].lower() : ".".join([i[2].lower(), i[4].lower()]) for i in fetchall}
                f = [f"Foreign key {key} references {foreign_keys[key]}" for key in foreign_keys] if len(foreign_keys) > 0 else ''
                formatted_table = f"{table_name} ({', '.join(cols_with_values)}) {' '.join(f)}"
                formatted_table_no_vals = f"{table_name} ({', '.join(columns)})"
                formatted_tables.append(formatted_table)
                formatted_tables_no_vals.append(formatted_table_no_vals)
            return formatted_tables,formatted_tables_no_vals

        # Get the schema for the database, convert it to adjusted format
        formatted_schema, _, = parse_schema(makeSchema(database),utterance)
        pbar.set_postfix_str(f"Generating", refresh=True)
        sysprompt = ""
        sysprompt += "### You are a sql generator, only output plain SQL code, starting with \"SELECT\" and nothing else. \n" # 1
        sysprompt += "### Answer the questions based on the following schema for the database (table (col1 [example value], col2 [example value],...)).\n"
        sysprompt += "### Only output what is necessary to answer the question, do not output any additional information.\n"
        sysprompt += "### If you are unable to answer the question, output your best guess.\n"
        sysprompt += '# ' + '\n# '.join(formatted_schema)
        
        prompt = ""
        prompt += utterance
        prompt += "\nSELECT "

        gpt = gptcall(
            sysprompt,
            prompt)
        claude = claudecall(
            sysprompt,
            prompt)
        with open('predict_claude_baseline.txt','a') as f:
            claude = claude.strip()
            claude = claude.replace('```sql',"")
            claude = claude.replace('```',"")
            claude = claude.strip()
            if claude.lower().startswith('select'):
                claude = claude.replace('\n',' ')+'\n'
            else:
                claude = "SELECT " + claude.replace('\n',' ')+'\n'
            f.write(claude)

        with open('predict_GPT4Turbo_baseline.txt','a') as f:
            gpt = gpt.strip()
            gpt = gpt.replace('```sql',"")
            gpt = gpt.replace('```',"")
            gpt = gpt.strip()
            if gpt.lower().startswith('select'):
                gpt = gpt.replace('\n',' ')+'\n'
            else:
                gpt = "SELECT " + gpt.replace('\n',' ')+'\n'
            f.write(gpt)

        pbar.update(1)
