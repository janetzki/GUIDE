import re

from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

df_main = pd.read_excel("data/2_sd_labeling/matched_questions.xlsx", usecols=["direct_question", "qid"], nrows=156501)

# add a column 'old_question' that is 'direct_question' until the first number
df_main['old_question'] = df_main['direct_question'].replace(r'\(\w{3} \d+:\d+\) ', '', regex=True)

# use old_question as index
df_main = df_main.set_index('old_question')

# update each question in each test set by looking it up in df_main 'old_question' column
for test_file_num in tqdm([2, 4, 6], total=3):
    df_test = pd.read_csv(f"data/2_sd_labeling/test sets/test_set_{test_file_num}.csv")
    for index in df_test.index:
        old_question = df_test.at[index, 'direct_question']
        qid = df_test.at[index, 'qid']

        # skip questions that have already been updated (i.e., that contain a bible reference)
        if len(re.findall(r' \(\w{3} \d+:\d+\) ', old_question)) > 0:
            continue

        if old_question in df_main.index:
            # lookup old_question
            results = df_main.loc[old_question]
            if type(results) == pd.DataFrame:
                # lookup qid
                results = results[results['qid'] == qid]['direct_question']
                assert len(results) == 1
                new_question = results[0]
            else:
                new_question = results['direct_question']
            df_test.at[index, 'direct_question'] = new_question
        else:
            print(f"Error: question not found: '{old_question}'")

            # get all rows with same qid
            df_main_same_qid = df_main[df_main['qid'] == qid]

            # find the most similar index entry
            df_main_same_qid['similarity'] = df_main_same_qid.index.to_series().apply(
                lambda x: SequenceMatcher(None, old_question, x).ratio())
            new_question = df_main_same_qid.sort_values('similarity', ascending=False).iloc[0]['direct_question']
            df_test.at[index, 'direct_question'] = '??? ' + old_question + ' ==> ' + new_question

    df_test.to_csv(f"data/2_sd_labeling/test sets/test_set_{test_file_num}_updated.csv", index=False)
