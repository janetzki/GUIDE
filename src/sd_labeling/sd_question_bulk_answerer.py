# bulk answer semantic domain questions using the following approach:

import math
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

df = pd.read_excel("data/2_sd_labeling/matched_questions.xlsx", usecols=["direct_question", "answer", "qid", "tokens"],
                   nrows=156501)

# assert that no qid starts with '9'
assert not any(df['qid'].str.startswith('9'))

# filter out qids that start with '1'
df = df[~df['qid'].str.startswith('1')]

# group questions by (qid, token)
qid_and_token_to_rows = defaultdict(list)
for idx, row in df.iterrows():
    qid_and_token_to_rows[(row['qid'], row['tokens'])].append((idx, row))

# filter (qid, token) pairs that have at least 2 answers (0, 1)
qid_and_token_to_rows = {k: rows for k, rows in qid_and_token_to_rows.items()
                         if len([row['answer'] for (idx, row) in rows
                                 if row['answer'] in (0, 1)]) >= 2}

# if all answers are the same, use this answer as the answer to all questions of this (qid, token) pair
additional_answers_count = 0
for _, rows in tqdm(qid_and_token_to_rows.items(),
                    desc='Bulk answering SD questions...',
                    total=len(qid_and_token_to_rows)):
    answers = [row['answer'] for (idx, row) in rows if row['answer'] in (0, 1)]
    if len(set(answers)) > 1:
        continue
    bulk_answer = answers[0]
    for (idx, row) in rows:
        if math.isnan(row['answer']):
            additional_answers_count += 1
            target_row = df.loc[idx]
            assert target_row['direct_question'] == row['direct_question']
            target_row['answer'] = bulk_answer # todo: fix bug that df does not add new answer
            print(f'{bulk_answer}: {row["direct_question"]}')
        else:
            assert (row['answer'] == bulk_answer)
print(f'Added {additional_answers_count} additional answers.')

# save to csv
df.to_csv("data/2_sd_labeling/matched_questions_with_bulk_answers.csv", index=False)
print('Done.')
