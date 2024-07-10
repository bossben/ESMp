################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)     ############### /col_unit #CHANGED TO VAL_UNIT
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize
from copy import deepcopy
import networkx as nx

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')
JOIN_TYPES = ['join', 'left-join', 'right-join', 'inner-join', 'outer-join', 'cross-join']
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')



class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals['columns']:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema_dict = {'columns': [str(col[1].lower()) for col in cursor.fetchall()], 'primary_keys': []}
        cursor.execute(f"SELECT l.name FROM pragma_table_info('{table}') as l WHERE l.pk > 0;")
        schema_dict['primary_keys'] += [j.lower() for i in cursor.fetchall() for j in i]
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fetchall = cursor.fetchall()
        for i in range(len(fetchall)):
            if fetchall[i][4] == None:
                cursor.execute(f"PRAGMA table_info({fetchall[i][2]})")
                primary_keys = [col[1] for col in cursor.fetchall() if col[5] == 1]
                assert len(primary_keys) == 1, "Foreign key referencing more than one primary key or no primary keys"
                lister = list(fetchall[i])
                lister[4] = primary_keys[0]
                fetchall[i] = tuple(lister)
        foreign_keys = {i[3].lower(): ".".join([i[2].lower(), i[4].lower()]) for i in fetchall}
        schema_dict['foreign_keys'] = foreign_keys
        cursor.execute(f"PRAGMA table_info({table})")
        schema_dict['non_null'] = [i[1].lower() for i in cursor.fetchall() if i[3] == 1]
        # add primary keys if they aren't already there
        if len(schema_dict['primary_keys']) == 1:
            schema_dict['non_null'] += schema_dict['primary_keys']
        # remove duplicates
        schema_dict['non_null'] = list(set(schema_dict['non_null']))
        cursor.execute(f"PRAGMA index_list({table})")
        indices = cursor.fetchall()
        # Collect all columns that are part of a unique index
        unique_columns = set()
        for index in indices:
            if index[2]:  # The unique flag is True
                index_name = index[1]
                cursor.execute(f"PRAGMA index_info({index_name})")
                index_info = cursor.fetchall()
                # Add all columns in this unique index to the set
                if len(index_info) == 1:
                    unique_columns.update(info[2] for info in index_info)
        schema_dict['unique'] = [i.lower() for i in unique_columns]
        if len(schema_dict['primary_keys']) == 1:
            schema_dict['unique'] += schema_dict['primary_keys']
        schema_dict['unique'] = list(set(schema_dict['unique']))
        schema[table] = schema_dict

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string): # this tokenizes the string, accounting for quotes and !=, >=, <= issues with tokenization
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"'] 
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"
    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val
    toks = [word.lower() for word in word_tokenize(string)] # split into individual words/tokens
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

            
    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    eq_idxs = [idx for idx, tok in enumerate(toks) if tok.lower() == "join"]
    eq_idxs.reverse()
    prefix = ('left', 'right', 'outer', 'cross')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok.lower() in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "-join"] + toks[eq_idx+1:]
    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        # added by ram for dail
        toks[idx + 1] = toks[idx + 1].strip('"')
        toks[idx + 1] = toks[idx + 1].strip("'")
        toks[idx + 1] = toks[idx + 1].strip("`").lower()
        toks[idx - 1] = toks[idx - 1].strip('"')
        toks[idx - 1] = toks[idx - 1].strip("'")
        toks[idx - 1] = toks[idx - 1].strip("`").lower()
        # added by ram for dail
        alias[toks[idx+1]] = toks[idx-1]

    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks) # builds a dictionary of aliases (dict['alias'] = original)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables, active_rules):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    tok = tok.strip('"')
    tok = tok.strip('`')
    tok = tok.strip("'").lower()
    #
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col

        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]['columns']:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]
    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables, active_rules):
    """
        :returns next idx, (agg_op id, col_id)
    """
    
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables,active_rules)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        
        isDistinct = fixRule2col(schema,col_id,isDistinct,active_rules)
            
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables, active_rules)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    
    # if col_id is a primary key, isDistinct can be made False, since primary keys are already distinct
    isDistinct = fixRule2col(schema,col_id,isDistinct,active_rules)


    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')
    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables,active_rules)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables,active_rules)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema, active_rules):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = toks[idx]
    key = key.strip('"')
    key = key.strip('`')
    key = key.strip("'")
    toks[idx] = key.lower()
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    elif idx + 1 < len_ and toks[idx+1] not in JOIN_KEYWORDS \
        and toks[idx + 1] not in JOIN_TYPES and toks[idx+1] not in CLAUSE_KEYWORDS and toks[idx+1] \
        not in COND_OPS and toks[idx+1] not in ORDER_OPS and \
        toks[idx+1] != ")" and toks[idx+1] != ";" and toks[idx+1] != ',' \
        and toks[idx+1] not in SQL_OPS and toks[idx+1] != "on" and toks[idx+1] != "limit":
        # add to alias table
        if 14 in active_rules:
            print("Applying Rule 14: table as T equivalent to table T")
            tables_with_alias[toks[idx+1]] = key
            idx += 2
        else:
            idx += 1
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, db,default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)
    isBlock = False

    if toks[idx] == '(':
        isBlock = True
        idx += 1
    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, db, active_rules)
    elif "\"" in toks[idx]:  # token is a string value
        # first try to see if it can be converted to a float
        try:
            val = fixRule11(toks[idx],active_rules)
            idx += 1
        except: # it can't, so it's a string
            val = toks[idx]
            idx += 1
    elif toks[idx]=="null":
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:

            end_idx = idx
            while (end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS
                   and toks[end_idx] not in JOIN_TYPES):
                    end_idx += 1

            idx, val = parse_val_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables,active_rules)
            idx = end_idx
    
    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, db, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)
    conds = []
    while idx < len_:
        not_op = False
        if toks[idx] == 'not':
            not_op = not not_op
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables, active_rules) #unit_operation, col1, col2
        if toks[idx] == 'not':
            not_op = not not_op
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        # Special case: IN
        isnum = False
        try:
            float(toks[idx + 2])
            isnum = True
        except:
            pass
        if op_id == WHERE_OPS.index('in') and toks[idx + 2] != 'select' and ("\"" in toks[idx+2] or isnum):
            idx += 1
            assert toks[idx] == '('
            idx += 1
            in_vals = []
            
            while idx < len_ and toks[idx] != ')':
                idx, in_val = parse_value(toks, idx, tables_with_alias, schema, db,default_tables, active_rules)
                in_vals.append(in_val)
                if idx < len_ and toks[idx] == ',':
                    idx += 1
            assert toks[idx] == ')'
            idx += 1
            conds = fixRule15(in_vals, conds, not_op, val_unit,active_rules)
            
        else:
            idx += 1

            val1 = val2 = None
            if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, db, default_tables, active_rules)
                assert toks[idx] == 'and'
                idx += 1
                idx, val2 = parse_value(toks, idx, tables_with_alias, schema, db, default_tables, active_rules)
            else:  # normal case: single value
                if toks[idx] == 'not':
                    not_op = not not_op
                    idx += 1

                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, db, default_tables, active_rules)
                val2 = None
            if op_id == WHERE_OPS.index('=') and type(val1) == tuple: # order matters for evaluation
                val_unit,val1 = sorted([val_unit,val1])[0],sorted([val_unit,val1])[1]  # join columns sorting.

            not_op, op_id, val_unit, val1, val2 = fixRule20(not_op, op_id, val_unit, val1, val2,active_rules)
            rule15 = fixRule8(not_op, op_id, val_unit, val1, val2,schema, active_rules)
            if not rule15:
                conds.append((not_op, op_id, val_unit, val1, val2))
                

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS or toks[idx] in JOIN_TYPES):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or
    if conds:
        if conds[-1] in COND_OPS:
            conds = conds[:-1]
    if conds:
        if conds[0] in COND_OPS:
            conds = conds[1:]
    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []
    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")

        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables, active_rules)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == 'as':
            idx += 2
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
    
    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema,db, active_rules):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"
    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False # is it a subquery?
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select': # subquery
            idx, sql = parse_sql(toks, idx, db, active_rules)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            join_type = None
            if idx < len_ and toks[idx] in JOIN_TYPES:
                join_type = toks[idx]
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema, active_rules) # reads table (sometimes with alias)
            if join_type:
                table_units.append(join_type)
            table_units.append((TABLE_TYPE['table_unit'], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, db, default_tables, active_rules)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, db, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, db, default_tables, active_rules)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables, active_rules)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables, active_rules)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, db, default_tables, active_rules):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1

    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, db, default_tables, active_rules)
    return idx, conds


def parse_limit(toks, start_idx, active_rules):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2

        return idx, int(toks[idx-1])

    return idx, None
def fixRule9(sql,schema,active_rules):
    if 9 not in active_rules:
        return
    # if order by is desc, there is 1 val_unit, limit is 1, and select is selecting the same val_unit with aggregate 0
    sqlselect = sql['select']
    sqlfrom = sql['from']
    sqlorder = sql['orderBy']
    sqllimit = sql['limit']
    if sqlorder==[]:
        return
    is_distinct, val_units = sqlselect
    if len(val_units) != 1:
        return
    
    if val_units[0][0] != 0:
        return
    
    val_unit = val_units[0][1]
    if sqlorder[0] == 'desc' and sqllimit == 1:
        if len(sqlorder[1]) != 1:
            return
        if val_unit == sqlorder[1][0]:
            print("Applying Rule 9: Order by asc/desc limit 1 equivalent to min/max")
            sql['select'] = (is_distinct, [(1, val_unit)])
            sql['orderBy'] = []
            sql['limit'] = None
    if sqlorder[0] == 'asc' and sqllimit == 1:
        if len(sqlorder[1]) != 1:
            return
        if val_unit == sqlorder[1][0]:
            print("Applying Rule 9: Order by asc/desc limit 1 equivalent to min/max")
            sql['select'] = (is_distinct, [(2, val_unit)])
            sql['orderBy'] = []
            sql['limit'] = None

def fixRule7(sql,schema, active_rules):
    if 7 not in active_rules:
        return
    sqlselect = sql['select']
    sqlfrom = sql['from']
    join_conditions = sqlfrom['conds']
    is_distinct, val_units = sqlselect
    for val_unitind in range(len(val_units)):
        agg_id, val_unit = val_units[val_unitind]
        if agg_id == 3:
            unit_op, col_unit1, col_unit2 = val_unit
            assert col_unit2 == None, "Rule 2: Unit Operation Error"
            if unit_op == 0:
                agg_id, col_id, isDistinct = col_unit1
                if isDistinct:
                    break
                if agg_id == 0:
                    for key, value in schema.idMap.items():
                        if value == col_id:
                            col_name = key
                    if col_name == "*" or col_name == "__all__":
                        continue
                    table_name = col_name.split(".")[0]
                    col_name = col_name.split(".")[1]
                    if col_name in schema.schema[table_name]['non_null']:
                        print("Applying Rule 7: Count(non_null) equivalent to Count(*)")
                        val_units[val_unitind] = (3, (0, (0, '__all__', False), None))

    sqlgroupby = sql['groupBy']
    sqlhaving = sql['having']
    sqlorderby = sql['orderBy']
    if len(sqlgroupby) != 0:
        for col_unitind in range(len(sqlgroupby)):
            col_unit = sqlgroupby[col_unitind]
            agg_id, col_id, isDistinct = col_unit
            if agg_id == 3:
                for key, value in schema.idMap.items():
                    if value == col_id:
                        col_name = key
                if col_name == "*" or col_name == "__all__":
                    continue
                table_name = col_name.split(".")[0]
                col_name = col_name.split(".")[1]
                if col_name in schema.schema[table_name]['non_null']:
                    print("Applying Rule 7: Count(non_null) equivalent to Count(*)")
                    col_unit = agg_id, '__all__', isDistinct
                    sqlgroupby[col_unitind] = col_unit
    if len(sqlhaving) != 0:
        for condind in range(len(sqlhaving)):
            if type(sqlhaving[condind]) != tuple:
                continue
            cond = sqlhaving[condind]
            not_op, op_id, val_unit, val1, val2 = cond
            unit_op, col_unit1, col_unit2 = val_unit
            if unit_op != 0:
                continue
            assert col_unit2 == None, "Rule 2: Unit Operation Error"
            agg_id, col_id, isDistinct = col_unit1
            
            if agg_id == 3:
                for key, value in schema.idMap.items():
                    if value == col_id:
                        col_name = key
                if col_name == "*" or col_name == "__all__":
                    continue
                table_name = col_name.split(".")[0]
                col_name = col_name.split(".")[1]
                if col_name in schema.schema[table_name]['non_null']:
                    print("Applying Rule 7: Count(non_null) equivalent to Count(*)")
                    col_id = '__all__'
                    sqlhaving[condind] = (not_op, op_id, (unit_op, (agg_id, col_id, isDistinct), None), val1, val2)

    if len(sqlorderby) != 0:
        val_units = sqlorderby[1]
        for val_unitind in range(len(val_units)):
            val_unit = val_units[val_unitind]
            unit_op, col_unit1, col_unit2 = val_unit
            if unit_op != 0:
                continue
            assert col_unit2 == None, "Rule 2: Unit Operation Error"
            if unit_op == 0:
                agg_id, col_id, isDistinct = col_unit1
                if isDistinct:
                    break
                if agg_id == 3:
                    for key, value in schema.idMap.items():
                        if value == col_id:
                            col_name = key
                    if col_name == "*" or col_name == "__all__":
                        continue
                    table_name = col_name.split(".")[0]
                    col_name = col_name.split(".")[1]
                    if col_name in schema.schema[table_name]['non_null']:
                        print("Applying Rule 7: Count(non_null) equivalent to Count(*)")
                        val_units[val_unitind] = (unit_op, (agg_id, '__all__', isDistinct), None)



def fixRule3(sql, schema, active_rules):
    if 3 not in active_rules:
        return
    sqlselect = sql['select']
    sqlfrom = sql['from']
    sqlwhere = sql['where']
    sqlintersect = sql['intersect']
    if sql['union'] != None:
        return
    if sql['except'] != None:
        return
    if sqlintersect == None:
        return
    assert type(sqlintersect) == dict, "Rule 5: Intersect is not a SQL"
    sqlselect2 = sqlintersect['select']
    if sqlselect != sqlselect2:
        return

    # check that selected col is unique
    isUnique = False
    is_distinct, val_units = sqlselect
    for val_unitind in val_units:
        agg_id, val_unit = val_unitind
        if agg_id != 0:
            continue
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op != 0:
            continue
        assert col_unit2 == None, "Rule 5: Unit Operation Error"
        agg_id, col_id, isDistinct = col_unit1
        for key, value in schema.idMap.items():
            if value == col_id:
                col_name = key
        if col_name != '*':
            table_name = col_name.split(".")[0]
            col_name = col_name.split(".")[1]
            if col_name in schema.schema[table_name]['unique']:
                isUnique = True
                pass
        else:
            if len(sql['from']['table_units']) == 1 and sql['from']['table_units'] == sql['union']['from']['table_units']:
                if schema.schema[sql['from']['table_units'][0][1].strip('_')]['primary_keys'] or schema.schema[sql['from']['table_units'][0][1].strip('_')]['unique']:
                    isUnique = True

    if not isUnique:
        return
    
    sqlfrom2 = sqlintersect['from']
    
    if sqlfrom != sqlfrom2:
        return
    
    sqlwhere2 = sqlintersect['where']
    if len(sqlwhere) != 1 or len(sqlwhere2) != 1:
        return
    cond1 = sqlwhere[0]
    cond2 = sqlwhere2[0]
    if cond1==cond2:
        print("Applying Rule 3: Intersect two identical conditions equivalent to no intersect")
        sql['intersect'] = None
        return
    print("Applying Rule 3: Intersect two different conditions equivalent to and")
    sqlwhere.append('and')
    sqlwhere.append(cond2)
    sql['intersect'] = None
    sql['where'] = sqlwhere


def fixRule6(sql,schema,active_rules):
    if 6 not in active_rules:
        return
    
    sqlselect = sql['select']
    sqlfrom = sql['from']
    sqlexcept = sql['except']
    sqlwhere = sql['where']
    # if there is where not in SQL, then that sql can be moved to except, if the thing we are selecting is unique and non_null
    is_distinct, val_units = sqlselect

    if len(val_units) != 1:
        return
    
    val_unit = val_units[0]
    agg_op, val_unit = val_unit
    
    if agg_op != 0:
        return
    unit_op, col_unit1, col_unit2 = val_unit
    if unit_op != 0:
        return
    
    assert col_unit2 == None, "Rule 7: Unit Operation Error"
    agg_id, col_id, isDistinct = col_unit1
    for key, value in schema.idMap.items():
        if value == col_id:
            col = key
    table_name = col.split(".")[0]

    if col == "*" or col == "__all__":
        return
    col_name = col.split(".")[1]
    if col_name in schema.schema[table_name]['non_null'] and col_name in schema.schema[table_name]['unique']:
        pass
    else:
        return
    if sqlwhere == []:
        return
    
    if sqlexcept != None:
        return
    
    if len(sqlwhere) != 1:
        return

    cond = sqlwhere[0]
    not_op, op_id, val_unit, val1, val2 = cond
    op, col1, col2 = val_unit
    if op != 0:
        return
    agg_id, col_id, isDistinct = col1

    for key, value in schema.idMap.items():

        if value == col_id:
            wherecol = key
        
    table_name = wherecol.split(".")[0]
    if wherecol == "*" or wherecol == "__all__":
        return
    cname = wherecol.split(".")[1]

    if col != wherecol:
        return


    if not not_op:
        return
    
    if op_id != 8:
        return
    
    if val2 != None:
        return
    
    # check if val1 is sql
    
    try:
        if val1['select']:
            pass
    except:
        return
    print("Applying Rule 6: 'WHERE NOT IN SQL' equivalent to 'EXCEPT SQL'")
    sql['except'] = val1
    sql['where'] = []


def fixRule15(in_vals, conds, not_op, val_unit, active_rules):
    if 15 not in active_rules:
        conds.append((not_op, WHERE_OPS.index('in'), val_unit, [val for val in in_vals], None))
        return conds
    print("Applying Rule 15: 'IN (A,B,C)' Equivalent to '= A or = B or = C'")
    for val in in_vals:
        if not_op:
            conds.append((False, WHERE_OPS.index('!='), val_unit, val, None))
            conds.append('and')
        else:
            conds.append((False, WHERE_OPS.index('='), val_unit, val, None))
            conds.append('or')
    
    conds = conds[:-1]  # remove the last 'and/or'
    return conds


def fixRule2(sql,schema,active_rules):
    if 2 not in active_rules:
        return
    sqlselect = sql['select']
    sqlfrom = sql['from']
    # from has to have 1 table
    tables = sqlfrom['table_units']
    if len(tables) != 1:
        return
    is_distinct, val_units = sqlselect
    if is_distinct:
        #loop through val_units, if any of them are unique, set is_distinct to False
        for val_unitind in val_units:
            agg_id, val_unit = val_unitind
            if agg_id != 0:
                continue
            unit_op, col_unit1, col_unit2 = val_unit
            if unit_op != 0:
                continue
            assert col_unit2 == None, "Rule 10: Unit Operation Error"
            agg_id, col_id, isDistinct = col_unit1
            for key, value in schema.idMap.items():
                if value == col_id:
                    col_name = key
            table_name = col_name.split(".")[0]
            col_name = col_name.split(".")[1]
            if col_name in schema.schema[table_name]['unique']:
                print("Applying Rule 2: 'DISTINCT col' equivalent to 'col' if col is UNIQUE")
                sql['select'] = (False, sql['select'][1])
                break
def fixRule2col(schema,col_id,distinct, active_rules):
    if 2 not in active_rules:
        return distinct
    if distinct:
        for key, value in schema.idMap.items():
            if value == col_id:
                col_name = key
        if col_name == "*" or col_name == "__all__":
            print("Applying Rule 2: 'DISTINCT col' equivalent to 'col' if col is UNIQUE")
            return False
        table_name = col_name.split(".")[0]
        col_name = col_name.split(".")[1]
        if col_name in schema.schema[table_name]['unique']:
            print("Applying Rule 2: 'DISTINCT col' equivalent to 'col' if col is UNIQUE")
            return False
        return True
    return distinct

def fixRule4(sql, schema, active_rules):
    if 4 not in active_rules:
        return
    sqlselect = sql['select']
    sqlfrom = sql['from']
    sqlwhere = sql['where']
    sqlunion = sql['union']
    if sql['intersect'] != None:
        return
    if sql['except'] != None:
        return
    if sqlunion == None:
        return
    assert type(sqlunion) == dict, "Rule 11: Union is not a SQL"
    sqlselect2 = sqlunion['select']
    if sqlselect != sqlselect2:
        return
    # check that selected col is unique
    isUnique = False
    is_distinct, val_units = sqlselect
    for val_unitind in val_units:
        agg_id, val_unit = val_unitind
        if agg_id != 0:
            continue
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op != 0:
            continue
        assert col_unit2 == None, "Rule 11: Unit Operation Error"
        agg_id, col_id, isDistinct = col_unit1
        for key, value in schema.idMap.items():
            if value == col_id:
                col_name = key
        if col_name != '*':
            table_name = col_name.split(".")[0]
            col_name = col_name.split(".")[1]
            if col_name in schema.schema[table_name]['unique']:
                isUnique = True
                pass
        else:
            if len(sql['from']['table_units']) == 1 and sql['from']['table_units'] == sql['union']['from']['table_units']:
                if schema.schema[sql['from']['table_units'][0][1].strip('_')]['primary_keys'] or schema.schema[sql['from']['table_units'][0][1].strip('_')]['unique']:
                    isUnique = True
    if not isUnique:
        return
    sqlfrom2 = sqlunion['from']
    if sqlfrom != sqlfrom2:
        return
    sqlwhere2 = sqlunion['where']
    if len(sqlwhere) != 1 or len(sqlwhere2) != 1:
        return
    cond1 = sqlwhere[0]
    cond2 = sqlwhere2[0]
    if cond1==cond2:
        print("Applying Rule 4: Union two identical conditions equivalent to no union")
        sql['union'] = None
        return
    print("Applying Rule 4: Union two different conditions equivalent to or")
    sqlwhere.append('or')
    sqlwhere.append(cond2)
    sql['union'] = None
    sql['where'] = sqlwhere

def fixRule13CheckClauses(sql, schema, pk_col, pk_table, fk_col, fk_table, curr_join_idx):
    # check if where, groupby, orderby and having are using a column from the pk table.
    def check_col_units(col_unit1):
        if not col_unit1:
            return True

        agg_id, col_id, isDistinct = col_unit1
        if col_id == '__all__':
            if agg_id==3:
                return True
            else:
                return False
        table1 = col_unit1[1].split('.')[0].strip('_')

        # if col is from fk_table or col_name is same as pk_col
        if table1 == fk_table or (col_unit1[1].split('.')[1].strip('_') == pk_col and col_unit1[1].split('.')[0].strip('_') == pk_table) or (table1 != fk_table and table1 != pk_table):
            return True
        return False

    def check_where():
        all_conds_passed = False
        conds = sql['where']
        if conds:
            for cond in conds[::2]:
                _, _, val_unit, _, _ = cond
                _, col_unit1, col_unit2 = val_unit
                if check_col_units(col_unit1) and check_col_units(col_unit2):
                    all_conds_passed = True
                else:
                    return False
        else:
            return True
        return all_conds_passed

    def check_select():
        select = sql['select']
        all_conds_passed = False
        val_units_with_agg = select[1]
        for val_unit_with_agg in val_units_with_agg:
            agg_idx, val_unit = val_unit_with_agg
            if agg_idx == 3:
                _, col_unit1, col_unit2 = val_unit
                dummycol1 = (3,col_unit1[1],col_unit1[2])
                if col_unit2:
                    dummycol2 = (3,col_unit2[1],col_unit2[2])
                else:
                    dummycol2 = dummycol1
                if check_col_units(dummycol1) and check_col_units(dummycol2):
                    all_conds_passed = True
                else:
                    return False
            else:
                _, col_unit1, col_unit2 = val_unit
                if check_col_units(col_unit1) and check_col_units(col_unit2):
                    all_conds_passed = True
                else:
                    return False
        return all_conds_passed


    def check_orderBy():
        orderby = sql['orderBy']
        all_conds_passed = False
        if orderby:
            val_units = sql['orderBy'][1]
            for val_unit in val_units:
                _, col_unit1, col_unit2 = val_unit
                if check_col_units(col_unit1) and check_col_units(col_unit2):
                    all_conds_passed = True
                else:
                    return False
        else:
            return True

        return all_conds_passed

    def check_having():
        having = sql['having']
        all_conds_passed = []
        if having:
            for cond in having[::2]:
                _, _, val_unit, _, _ = cond
                _, col_unit1, col_unit2 = val_unit
                if check_col_units(col_unit1) and check_col_units(col_unit2):
                    all_conds_passed = True
                else:
                    return False
        else:
            return True
        return all_conds_passed

    def check_groupBy():
        group_by = sql['groupBy']
        all_conds_passed = []
        col_unit2 = None
        if group_by:
            for col_unit1 in group_by:
                if check_col_units(col_unit1) and check_col_units(col_unit2):
                    all_conds_passed = True
                else:
                    return False
        else:
            return True
        return all_conds_passed

    def check_conds():
        join_conds = sql['from']['conds']
        all_conds_passed = None
        if len(join_conds) > 1:
            if join_conds:
                for idx_, cond in enumerate(join_conds[::2]):
                    if curr_join_idx != idx_ * 2:
                        _, _, val_unit, _, _ = cond
                        _, col_unit1, col_unit2 = val_unit
                        if check_col_units(col_unit1) and check_col_units(col_unit2):
                            all_conds_passed = True
                        else:
                            return False

            else:
                return True
        else:
            return True
        return all_conds_passed

    where = check_where()
    select = check_select()
    having = check_having()
    orderby = check_orderBy()
    groupby = check_groupBy()
    join_ = check_conds()
    if where and select and having and orderby and groupby and join_:
        return True
    else:
        return False

def fixRule13UpdateColUnits(col_unit1, pk_col, pk_table, fk_col, fk_table, col_unit2 = None):
    agg_id, col_id, isDistinct = col_unit1
    if col_id == '__all__':
        pass
    else:
        table_name, col_name = col_unit1[1].split('.')
        if col_name.strip('_') == pk_col and table_name.strip('_') == pk_table:
            col_unit1 = (col_unit1[0],) + (f'__{fk_table}.{fk_col}__',) + (col_unit1[2],)
        else:
            col_unit1 = col_unit1

    if col_unit2:
        agg_id, col_id, isDistinct = col_unit2
        if col_id == '__all__':
            pass
        else:
            table_name, col_name = col_unit2[1].split('.')
            if col_name.strip('_') == pk_col and table_name.strip('_') == pk_table:
                col_unit2 = (col_unit2[0],) + (f'__{fk_table}.{fk_col}__',) + (col_unit2[2],)
            else:
                col_unit2 = col_unit2
    return col_unit1, col_unit2


def fixRule13CheckUpdateSelect(sql, pk_col=None, pk_table=None, fk_col=None, fk_table=None):
    select_unit = sql['select']
    out = []
    isDistinct, select_val_units_with_agg = select_unit
    select_unit = sql['select']
    isDistinct, select_val_units_with_agg = select_unit
    for val_unit_with_agg in select_val_units_with_agg:
        agg_idx, val_unit = val_unit_with_agg
        unit_op, col_unit1, col_unit2 = val_unit
        col_unit1, col_unit2 = fixRule13UpdateColUnits(col_unit1, pk_col, pk_table, fk_col, fk_table, col_unit2)
        out.append((agg_idx, (unit_op, col_unit1, col_unit2)))
    return (isDistinct, out)

def fixRule13CheckUpdateWhereHaving(sql, pk_col=None, pk_table=None, fk_col=None, fk_table=None, where=True):
    if where:
        conds = sql['where']
    else:
        conds = sql['having']
    if conds:
        out = []
        for cond in conds:
            if cond == 'and' or cond == 'or':
                out.append(cond)
                continue
            not_op, where_ops_idx, val_unit, val1, val2 = cond
            unit_op, col_unit1, col_unit2 = val_unit
            col_unit1, col_unit2 = fixRule13UpdateColUnits(col_unit1, pk_col, pk_table, fk_col, fk_table, col_unit2)
            val_unit = (unit_op, col_unit1, col_unit2)
            out.append((not_op, where_ops_idx, val_unit, val1, val2))
        return out

    else:
        return []



def fixRule13CheckUpdateGroupby(sql, pk_col=None, pk_table=None, fk_col=None, fk_table=None):
    group_by = sql['groupBy']
    if group_by:
        out = []
        for col_unit1 in group_by:
            col_unit1, _ = fixRule13UpdateColUnits(col_unit1, pk_col, pk_table, fk_col, fk_table)
            out.append(col_unit1)
        return out

    else:
        return []

def fixRule13CheckUpdateOrderby(sql, pk_col=None, pk_table=None, fk_col=None, fk_table=None):
    out = []
    orderby = sql['orderBy']
    if orderby:
        out = []
        if orderby:
            order, val_units = sql['orderBy']
            for val_unit in val_units:
                unit_op, col_unit1, col_unit2 = val_unit
                col_unit1, col_unit2 = fixRule13UpdateColUnits(col_unit1, pk_col, pk_table, fk_col, fk_table,
                                                                col_unit2)
                val_unit = (unit_op, col_unit1, col_unit2)
                out.append(val_unit)
        return order,out

    else:
        return []


def fixRule13Updateall(sql, schema, pk_col, pk_table, fk_col, fk_table, idx_1):
    table_units = sql['from']['table_units']
    conds = sql['from']['conds']
    # 1: remove pk_table_unit
    for idx, tab_unit in enumerate(table_units):
        if type(tab_unit) == str:
            continue
        if tab_unit[1] == schema.idMap[pk_table]:

            print('Applying Rule 13: Excess Joins equivalent to no join')

            # fix conditions
            for idy, cond in enumerate(conds):
                if type(cond) == str:
                    continue
                if not cond:
                    continue
                not_op, where_ops, val_unit1, val1, val2 = cond
                if where_ops == 2:
                    value1 = val_unit1[1][1]
                    value2 = val1[1][1]

                    if (value1 == f'__{pk_table}.{pk_col}__' and value2 == f'__{fk_table}.{fk_col}__') or (
                            value1 == f'__{fk_table}.{fk_col}__' and value2 == f'__{pk_table}.{pk_col}__'):

                        conds[idy] = None
                        try:
                            conds[idy + 1] = None
                        except:
                            conds[-1] = None

                    else:

                        if val_unit1[1][1] == f'__{tab_unit[1].strip("_")}.{pk_col}__':

                            conds[idy] = (not_op, where_ops, (
                            val_unit1[0], (val_unit1[1][0], f'__{fk_table}.{fk_col}__', val_unit1[1][0]), val_unit1[2]),
                                          val1, val2)

                        if val1[1][1] == f'__{tab_unit[1].strip("_")}.{pk_col}__':
                            conds[idy] = (not_op, where_ops, val_unit1,
                                          (val1[0], (val1[1][0], f'__{fk_table}.{fk_col}__', val1[1][2]), val1[2]),
                                          val2)
            table_units.pop(idx)

            if idx < len(table_units):
                table_units.pop(idx)
            else:
                table_units.pop(-1)
    # remove any conds that are None
    conds = [x for x in conds if x is not None]
    sql['from']['conds'] = conds

    sqlold = sql.copy()
    sql['select'] = fixRule13CheckUpdateSelect(sql, pk_col, pk_table, fk_col, fk_table)
    sql['where'] = fixRule13CheckUpdateWhereHaving(sql, pk_col, pk_table, fk_col, fk_table)
    sql['having'] = fixRule13CheckUpdateWhereHaving(sql, pk_col, pk_table, fk_col, fk_table, False)
    sql['groupBy'] = fixRule13CheckUpdateGroupby(sql, pk_col, pk_table, fk_col, fk_table)
    sql['orderBy'] = fixRule13CheckUpdateOrderby(sql, pk_col, pk_table, fk_col, fk_table)
    if sqlold['select'] != sql['select']:
        print("Rule 13: Select changed")
    if sqlold['where'] != sql['where']:
        print("Rule 13: Where changed")
    if sqlold['having'] != sql['having']:
        print("Rule 13: Having changed")
    if sqlold['groupBy'] != sql['groupBy']:
        print("Rule 13: GroupBy changed")
    if sqlold['orderBy'] != sql['orderBy']:
        print("Rule 13: OrderBy changed")




def fixRule13(sql, schema, active_rules):
    if 13 not in active_rules:
        return
    if sql['from']['conds']:
        conds = sql['from']['conds']
        for idx, cond in enumerate(conds[::2]):
            not_op, where_ops, val_unit1, val1, val2 = cond
            if where_ops != 2:
                return
            unit_op_join1, col_unit_join1, col_unit_join12 = val_unit1
            unit_op_join2, col_unit_join2, col_unit_join22 = val1
            if col_unit_join22 and col_unit_join12:
                return
            
            join1_col = col_unit_join1[1]
            join2_col = col_unit_join2[1]
            table_name_join1, col_name_join1 = join1_col.split('.')
            table_name_join2, col_name_join2 = join2_col.split('.')
            table_name_join1, col_name_join1 = table_name_join1.strip('_'), col_name_join1.strip('_')
            table_name_join2, col_name_join2 = table_name_join2.strip('_'), col_name_join2.strip('_')
            pk_col, pk_table, isjoin1pk = None, None, None
            fk_col, fk_table = None, None
            if col_name_join1 in schema.schema[table_name_join1]['primary_keys'] and len(schema.schema[table_name_join1]['primary_keys']) == 1:
                # col1 is a non-composite primary key, is col2 a foreign key?
                if col_name_join2 in schema.schema[table_name_join2]['foreign_keys']:
                    # col2 is foreign key, does it reference col1?
                    if join1_col.strip('_') == schema.schema[table_name_join2]['foreign_keys'][col_name_join2]:
                        pk_table, pk_col = table_name_join1, col_name_join1
                        fk_table, fk_col = table_name_join2, col_name_join2
            if col_name_join2 in schema.schema[table_name_join2]['primary_keys'] and len(schema.schema[table_name_join2]['primary_keys']) == 1:
                # col2 is non-composite primary key, is col1 a foreign key?
                if col_name_join1 in schema.schema[table_name_join1]['foreign_keys']:
                    # col1 is foreign key, does it reference col2?
                    if join2_col.strip('_') == schema.schema[table_name_join1]['foreign_keys'][col_name_join1]:
                        pk_table, pk_col = table_name_join2, col_name_join2
                        fk_table, fk_col = table_name_join1, col_name_join1

            if not fk_col and not fk_table:
                return
            
            if pk_table and len(schema.schema[pk_table]['primary_keys']) > 1:
                return

            change = fixRule13CheckClauses(sql, schema, pk_col, pk_table, fk_col, fk_table, idx*2)

            if change:
                if fk_col and fk_col in schema.schema[fk_table]['non_null']:
                    fixRule13Updateall(sql, schema, pk_col, pk_table, fk_col, fk_table, idx*2)

def fixRule10(sql,schema,active_rules):
    if 10 not in active_rules:
        return
    sqlselect = sql['select']
    sqlfrom = sql['from']
    cols = []
    is_distinct, val_units = sqlselect
    for val_unitind in val_units:
        agg_id, val_unit = val_unitind
        if agg_id != 0:
            return sqlselect
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op != 0:
            return sqlselect
        assert col_unit2 == None, "Rule 13: Unit Operation Error"

        agg_id, col_id, isDistinct = col_unit1
        if agg_id != 0:
            return sqlselect
        cols.append(col_id)
    list_of_all_cols = []
    tables = sqlfrom['table_units']
    for table in tables:
        if table[0] == 'table_unit':
            table_id = table[1]
            for key, value in schema.idMap.items():
                if value == table_id:
                    table_name = key
            assert table_name in schema.schema, "Table not in schema"
            for col in schema.schema[table_name]['columns']:
                list_of_all_cols.append(schema.idMap[table_name + "." + col])
    # compare lists (sort and compare)
    if sorted(cols) == sorted(list_of_all_cols):
        print("Applying Rule 10: 'SELECT a,b,c,...' equivalent to 'SELECT *' if a,b,c,... are all columns in table")
        # change select to select all
        sqlselect = (is_distinct, [(0, (0, (0, '__all__', False), None))])
        sql['select'] = sqlselect
        return sqlselect

    return sqlselect

def fixRule17(sql, schema, active_rules):
    if 17 not in active_rules:
        return
    where_conds = sql['where']
    boo = False
    if where_conds:
        idx = None
        for idx, cond in enumerate(where_conds[::2]):
            not_op, where_ops, val_unit, val1, val2 = cond
            if not not_op and where_ops == 8:
                if type(val1)==dict:
                    boo = True
                    break
        if boo:
            not_op, where_ops, val_unit, val1, val2 = where_conds[2 * idx]
            if sorted(val1['from']['table_units'][0::2]) == sorted(sql['from']['table_units'][0::2]):   # CHECK_UPDATE
                sub_query_select_unit = val1['select'][1]
                if not val1['where']:
                    return
                if val_unit == sub_query_select_unit[0][1]:
                    print("Applying Rule 17: 'SELECT col FROM A WHERE Col IN (SELECT col FROM A WHERE COND)' equivalent to 'SELECT col FROM A WHERE COND'")
                    sql['where'] = sql['where'][:2*idx]
                    sql['where'] += val1['where'] 
                    try:
                        sql['where'] += sql['where'][2*idx+1:]
                    except:
                        pass
 

def fixRule8(not_op, op_id, val_unit, val1, val2,schema,active_rules):
    # where col is not null returns false if column is non_null
    # this function returns True if rule is applied, False if not
    if 8 not in active_rules:
        return False
    if not_op:
        if op_id == 10:
            if val1 != 'null':
                return False
            if val2 != None:
                return False
            unit_op, col1, col2 = val_unit
            if unit_op != 0:
                return False
            assert col2 == None, "Rule 15: Unit Operation Error"
            agg_id, col_id, isDistinct = col1
            if agg_id != 0:
                return False
            for key, value in schema.idMap.items():
                if value == col_id:
                    col_name = key
            table_name = col_name.split(".")[0]
            col_name = col_name.split(".")[1]
            if col_name in schema.schema[table_name]['non_null']:
                print("Applying Rule 8: 'A IS NOT NULL' equivalent to 'A' if A is NON_NULL")
                return True

    return False

def fixRule11(val, active_rules):
    if 11 not in active_rules:
        raise ValueError("Rule 17 is not active")
    val = float(val.replace("\"", ""))
    print("Applying Rule 11: String number value equivalent to float")
    return val
            

def fixRule20(not_op, op_id, val_unit, val1, val2,active_rules):
    if 20 not in active_rules:
        return not_op, op_id, val_unit, val1, val2
    # ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
    if not_op:
        not_op = False
        if op_id == 2:
            print("Applying Rule 20: Flip between NOT operator, and opposite operator")
            op_id = 7
        elif op_id == 3:
            print("Applying Rule 20: Flip between NOT operator, and opposite operator")
            op_id = 6
        elif op_id == 4:
            print("Applying Rule 20: Flip between NOT operator, and opposite operator")
            op_id = 5
        elif op_id == 5:
            print("Applying Rule 20: Flip between NOT operator, and opposite operator")
            op_id = 4
        elif op_id == 6:
            print("Applying Rule 20: Flip between NOT operator, and opposite operator")
            op_id = 3
        elif op_id == 7:
            print("Applying Rule 20: Flip between NOT operator, and opposite operator")
            op_id = 2
        else:
            not_op = True
    
    return not_op, op_id, val_unit, val1, val2

def fixRule12(sql, schema, active_rules):
    if 12 not in active_rules:
        return
    where_conds = sql['where']
    if where_conds:
        idx = None
        for idx, cond in enumerate(where_conds[::2]):
            not_op, where_ops, val_unit, val1, val2 = cond
            if not not_op and where_ops == 8 or where_ops == 2:
                break
        not_op, where_ops, val_unit, val1, val2 = where_conds[2*idx]
        unit_op, col_unit1, col_unit2 = val_unit
        if (where_ops == 2 or where_ops == 8) and not not_op and type(val1) == dict and len(val1['select'][1]) == 1:
            sub_query_select_val_unit = val1['select'][1][0][1]
            unit_op, sub_col_unit1, sub_col_unit2 = sub_query_select_val_unit
            if sub_col_unit2 is None:
                # is the retrieved column is a pk of the outer table ?
                subquery_table, subquery_column = sub_col_unit1[1].split('.')
                subquery_table, subquery_column = subquery_table.strip('_'), subquery_column.strip('_')
                if subquery_column in schema.schema[subquery_table]['primary_keys']:
                    if schema.schema[col_unit1[1].split('.')[0].strip('_')]['foreign_keys'].get(col_unit1[1].split('.')[1].strip('_'), -99) == sub_col_unit1[1].strip('_'):
                        print("Applying Rule 12: 'WHERE a IN (SELECT pk FROM B WHERE COND)' equivalent to 'A JOIN B ON A.a = B.pk WHERE COND'")
                        col_units_sorted = sorted([sub_col_unit1[1], col_unit1[1]])
                        sql['from']['conds'] += [(False, 2, (0, (0, col_units_sorted[0], False), None), (0, (0, col_units_sorted[1], False), None), None)]
                        sql['from']['conds'] += ['and'] + val1['from']['conds'] if val1['from']['conds'] else []
                        sql['from']['table_units'] += ['join'] + val1['from']['table_units']
                        sql['where'][2*idx:2*idx+1] = val1['where']

def fixRule1(sql, schema, active_rules):
    if 1 not in active_rules:
        return
    # select _ from t1 where c1 = (select min(c1) from t1) -> select _ from t1 order by c1 asc/desc limit 1
    where_conds = sql['where']
    
    if len(where_conds) != 1:
        return
    cond = where_conds[0]
    not_op, where_ops, val_unit, val1, val2 = cond
    if sql['orderBy']:
        return
    if where_ops != 2:
        return
    if not_op:
        return

    if type(val1) != dict:
        return
    if len(val1['select'][1]) != 1:
        return
    sub_query_select_val_unit = val1['select'][1][0][1]
    unit_op, sub_col_unit1, sub_col_unit2 = sub_query_select_val_unit
    if sub_col_unit2:
        return
    if val_unit[0] != 0:
        return
    if val_unit[1] != sub_col_unit1:
        return
    col_id = sub_col_unit1[1]
    col_name = None
    for key, value in schema.idMap.items():
        if value == col_id:
            col_name = key
    if not col_name:
        return
    if col_name == "*" or col_name == "__all__":
        return
    table_name = col_name.split(".")[0]
    col_name = col_name.split(".")[1]
    if col_name in schema.schema[table_name]['unique']:
        print("Applying Rule 1: 'SELECT _ FROM t1 WHERE C1 = (SELECT min/max(c1) FROM t1)' equivalent to 'SELECT _ FROM t1 ORDER BY c1 ASC/DESC LIMIT 1'")
        sql['where'] = []
        sql['orderBy'] = ('asc', [(0, (0, col_id, False), None)])
        sql['limit'] = 1




def fixRule16(sql, schema, active_rules):
    if 16 not in active_rules:
        return
    # if join exists with = and one of the col_units in the entire query are the ones in joins then replace.
    join_conditions = sql['from']['conds']
    join_cols = []

    if join_conditions:
        for join_condition in join_conditions[::2]:
            not_op, where_idx, val_unit, val1, val2 = join_condition
            unit_op, join_col_unit1, join_col_unit2 = val_unit
            unit_op_2, val_col_unit1, val_col_unit2 = val1
            if val2 is not None:
                continue
            if where_idx != 2:
                continue
            if val_unit[0] != 0:
                continue
            #decompose join_col_unit1
            agg_id, col1, isDistinct = join_col_unit1
            if agg_id != 0:
                continue
            if isDistinct:
                continue
            agg_id, col2, isDistinct = val_col_unit1
            if agg_id != 0:
                continue
            if isDistinct:
                continue

            join_cols.append({col1, col2})
        #aggregate all the join_cols
        G = nx.Graph()
        for s in join_cols:
            node_list = list(s)
            if len(node_list)==2:
                G.add_edge(node_list[0], node_list[1])
            elif len(node_list) == 1:
                G.add_node(node_list[0])
        connected_components = list(nx.connected_components(G))
        join_cols = [component for component in connected_components]

        join_cols = [sorted(list(i)) for i in join_cols]
        join_cols = sorted(join_cols)
        for j in join_cols:
            for col in j:
                for key,value in schema.idMap.items():
                    if value == col:
                        col_name = key
                table_name = col_name.split('.')[0]
                col_name = col_name.split('.')[1]
                if col_name in schema.schema[table_name]['non_null']:
                    # all values in j should be non_null
                    for col in j:
                        for key, value in schema.idMap.items():
                            if value == col:
                                col_name = key
                        table_name = col_name.split('.')[0]
                        col_name = col_name.split('.')[1]
                        if col_name not in schema.schema[table_name]['non_null']:
                            schema.schema[table_name]['non_null'].append(col_name)
                if col_name in schema.schema[table_name]['unique']:
                    # all values in j should be non_null
                    for col in j:
                        for key, value in schema.idMap.items():
                            if value == col:
                                col_name = key
                        table_name = col_name.split('.')[0]
                        col_name = col_name.split('.')[1]
                        if col_name not in schema.schema[table_name]['unique']:
                            schema.schema[table_name]['unique'].append(col_name)


        
        def change_select_col_units():
            val_units_with_agg = sql['select'][1]
            for idx, val_unit_with_agg in enumerate(val_units_with_agg):
                agg_idx, val_unit = val_unit_with_agg
                unit_op, col_unit1, col_unit2 = val_unit
                for j in join_cols:
                    if col_unit1[1] in j:
                        print("Applying Rule 16: a equivalent to b when join on a=b")

                        updated_val_unit = (agg_idx, (unit_op, (col_unit1[0], j[0], col_unit1[2]), col_unit2))
                        val_units_with_agg[idx] = updated_val_unit
                    if col_unit2:
                        if col_unit2[1] in j:
                            print("Applying Rule 16: a equivalent to b when join on a=b")
                            updated_val_unit = (agg_idx, (unit_op, col_unit1, (col_unit2[0], j[0], col_unit2[2])))
                            val_units_with_agg[idx] = updated_val_unit

        def change_where_having_col_units(where=True):
            if where:
                conds = sql['where']
            else:
                conds = sql['having']
            if conds:
                for idx, cond in enumerate(conds[::2]):
                    not_op, where_ops_idx, val_unit, val1, val2 = cond
                    unit_op, col_unit1, col_unit2 = val_unit
                    for j in join_cols:
                        if col_unit1[1] in j:
                            print("Applying Rule 16: a equivalent to b when join on a=b")
                            val_unit_updated = (unit_op, (col_unit1[0], j[0], col_unit1[2]),
                                                col_unit2)
                            updated_condition_unit = (not_op, where_ops_idx, val_unit_updated, val1, val2)
                            conds[2* idx] = updated_condition_unit

                        if col_unit2:
                            if col_unit2 in j:
                                print("Applying Rule 16: a equivalent to b when join on a=b")
                                val_unit_updated = (unit_op, col_unit1,
                                                    (col_unit2[0], j[0], col_unit2[2]))
                                updated_condition_unit = (not_op, where_ops_idx, val_unit_updated, val1, val2)
                                conds[2*idx] = updated_condition_unit
        def change_groupby():
            col_units = sql['groupBy']
            if col_units:
                for idx, col_unit in enumerate(col_units):
                    for j in join_cols:
                        if col_unit[1] in j:
                            print("Applying Rule 16: a equivalent to b when join on a=b")
                            updated_col_unit = (col_unit[0], j[0], col_unit[2])
                            col_units[idx] = updated_col_unit

        def change_orderby():
            orderby = sql['orderBy']
            if orderby:
                val_units = orderby[1]
                for idx, val_unit_with_agg in enumerate(val_units):
                    unit_op, col_unit1, col_unit2 = val_unit_with_agg
                    for j in join_cols:
                        if col_unit1[1] in j:
                            print("Applying Rule 16: a equivalent to b when join on a=b")
                            updated_val_unit = (unit_op, (col_unit1[0], j[0], col_unit1[2]), col_unit2)
                            val_units[idx] = updated_val_unit
                        if col_unit2:
                            if col_unit2[1] in j:
                                print("Applying Rule 16: a equivalent to b when join on a=b")
                                updated_val_unit = (unit_op, col_unit1, (col_unit2[0], j[0], col_unit2[2]))
                                val_units[idx] = updated_val_unit
        change_select_col_units()
        change_where_having_col_units()
        change_where_having_col_units(False)
        change_groupby()
        change_orderby()


def fixRule19(sql, schema, active_rules):
    if 19 not in active_rules:
        return
    where_unit = sql['where']
    for idx, cond in enumerate(where_unit[::2]):
        if cond[1] == 1:
            cond1 = list(cond)
            cond1[1] = 5
            cond1[-2] = cond[-2]
            cond1[-1] = None
            cond2 = list(cond)
            cond2[1] = 6
            cond2[-2] = cond[-1]
            cond2[-1] = None
            print("Applying Rule 19: 'X BETWEEN A AND B' equivalent to 'X <= B AND X >= A'")
            where_unit[2*idx:2*idx+1] = (tuple(cond1), 'and', tuple(cond2))


def fixRule5(sql, schema, active_rules):
    if 5 not in active_rules:
        return
    groupby = sql['groupBy']
    having = sql['having']
    if groupby and not having and len(sql['from']['table_units']) == 1:
        for col_unit in groupby:
            table_name, col_name = col_unit[1].split('.')
            table_name, col_name = table_name.strip('_'), col_name.strip('_')
            if (len(schema.schema[table_name]['primary_keys']) == 1 and col_name in schema.schema[table_name]['primary_keys']
                or col_name in schema.schema[table_name]['unique']):
                #check if select does not have any aggregates.
                select_unit = sql['select']
                isDistinct, val_units = select_unit
                for val_unit in val_units:
                    agg_idx, (unit_op, col_unit1, col_unit2) = val_unit
                    if agg_idx != 0:
                        return
                print("Applying Rule 5: 'SELECT _ FROM table GROUP BY col' equivalent to 'SELECT _ FROM table' if col is unique")
                sql['groupBy'] = []
    groupby = sql['groupBy']
    if sql['groupBy']:
        if len(groupby) > 1:
            for col_unit in groupby:
                table_name, col_name = col_unit[1].split('.')
                table_name, col_name = table_name.strip('_'), col_name.strip('_')
                if (col_name in schema.schema[table_name]['primary_keys'] and len(schema.schema[table_name]['primary_keys']) ==1) or col_name in schema.schema[table_name]['unique']:
                    print("Applying Rule 5: 'SELECT _ FROM table GROUP BY col,col2,...' equivalent to 'SELECT _ FROM table GROUP BY col' if col is unique")
                    sql['groupBy'] = [col_unit]
                    return

    


def fixRule18(sql, schema, active_rules):
    if 18 not in active_rules:
        return
    if sql['intersect']:
        sql_copy = deepcopy(sql)
        sql_copy['intersect'] = None
        if sql_copy == sql['intersect']:
            print("Applying Rule 18: Intersect with same query is equivalent to same query")
            sql['intersect'] = None

    if sql['union']:
        sql_copy = deepcopy(sql)
        sql_copy['union'] = None
        if sql_copy == sql['union']:
            print("Applying Rule 18: Union with same query is equivalent to same query")
            sql['union'] = None
def parse_sql(toks, start_idx, db,active_rules):
    schema = Schema(get_schema(db))
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    

    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    
    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema,db,active_rules)

    sql['from'] = {'table_units': table_units, 'conds': conds}


    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables,active_rules)
    idx = from_end_idx # skip from clause, go to right after it
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, db, default_tables,active_rules)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables,active_rules)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, db, default_tables,active_rules)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables,active_rules)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx,active_rules)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, db,active_rules)
        sql[sql_op] = IUE_sql

    fixRule13(sql,schema, active_rules)
    fixRule16(sql,schema, active_rules)
    fixRule3(sql,schema, active_rules)
    fixRule4(sql,schema, active_rules)
    fixRule1(sql,schema, active_rules)
    fixRule9(sql,schema, active_rules)
    fixRule10(sql,schema, active_rules)
    fixRule17(sql,schema, active_rules)
    fixRule7(sql,schema, active_rules)
    fixRule6(sql,schema, active_rules)
    fixRule2(sql,schema, active_rules)
    fixRule12(sql,schema, active_rules)
    fixRule19(sql, schema, active_rules)
    fixRule5(sql, schema, active_rules)
    fixRule18(sql, schema, active_rules)

    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(db, query,active_rules):
    query = query.replace('<>', '!=')
    print(query)
    print(db)
    toks = tokenize(query)
    _, sql = parse_sql(toks, 0, db,active_rules)
    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx
