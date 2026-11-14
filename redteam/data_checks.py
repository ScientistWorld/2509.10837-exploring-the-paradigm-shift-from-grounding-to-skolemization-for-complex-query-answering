"""Small audit diagnostics on the restored, canonical generated dataset."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from redteam.common import adjacency, exact_answers, read_rows, write_json

data = Path('data/synthetic')
rows = read_rows(data/'test.jsonl')
adj = adjacency()
meta = json.loads((data/'meta.json').read_text())
groups = {}
for index, row in enumerate(rows):
    query = {key: row['query'].get(key) for key in ('type', 'anchors', 'relations')}
    key = json.dumps({'query': query, 'answers': row['answers']}, sort_keys=True, separators=(',', ':'))
    groups.setdefault(query['type'], []).append((hashlib.sha256(key.encode()).hexdigest(), index))
test_indices = sorted(index for group in groups.values()
                      for _, index in sorted(group)[max(1, len(group)*3//5):])
test_rows = [rows[i] for i in test_indices]
def query_key(row):
    return json.dumps({key: row['query'][key] for key in ('type', 'anchors', 'relations')}, sort_keys=True)
training_keys = {query_key(row) for row in read_rows(data/'train.jsonl')}

def stats(subset):
    matches, negated, negation_changes, multi_answer, answer_count = 0, 0, 0, 0, 0
    for row in subset:
        query = row['query']
        answers, _ = exact_answers(query, adj, meta['num_entities'])
        matches += answers == set(row['answers'])
        multi_answer += len(answers)>1
        answer_count += len(answers)
        if query['type'] == '2in':
            negated += 1
            pos = adj[query['relations'][0]][query['anchors'][0]] & adj[query['relations'][1]][query['anchors'][1]]
            negation_changes += answers != pos
    return {'queries': len(subset), 'types': dict(Counter(row['query']['type'] for row in subset)),
        'labels_exactly_answerable_from_supplied_graph': matches,
        'negated_queries': negated, 'negation_changes_answer_set': negation_changes,
        'multi_answer_queries': multi_answer, 'labeled_answers': answer_count,
        'queries_also_present_in_training': sum(query_key(row) in training_keys for row in subset)}
write_json('redteam/data_checks.json', {'full_file': stats(rows), 'hidden_test': stats(test_rows)})
