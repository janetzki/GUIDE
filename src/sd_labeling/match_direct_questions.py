import pandas as pd
from tqdm import tqdm

# load matched_questions_gospels_raw.csv
raw_df = pd.read_csv('data/2_sd_labeling/matched_questions_gospels_raw.csv')

# keep rows with qids not starting with 9
raw_df = raw_df[~raw_df['qid'].str.startswith('9')]

# load matched_questions.xlsx
mq_df = pd.read_excel('data/2_sd_labeling/matched_questions.xlsx', nrows=156501)

# add column question_without_reference to raw_df and mq_df
raw_df['question_without_reference'] = raw_df['direct_question'].replace(r'\(\w{3} \d+:\d+\) ', '', regex=True)
mq_df['question_without_reference'] = mq_df['direct_question'].replace(r'\(\w{3} \d+:\d+\) ', '', regex=True)

# use question_without_reference as index
raw_df = raw_df.set_index('question_without_reference')
mq_df = mq_df.set_index('question_without_reference')

# for each question_without_reference in mq_df, find all corresponding rows in raw_df
for index, row in tqdm(mq_df.iterrows(), desc='Filling up answers...', total=mq_df.shape[0]):
    raw_df.loc[(raw_df.index == index) & (raw_df['qid'] == row['qid']), ['answer', 'gpt3_answer']] =\
        pd.DataFrame([row[['answer', 'gpt3_answer']]])

    matches = raw_df.loc[(raw_df.index == index) & (raw_df['qid'] == row['qid']), ['answer', 'gpt3_answer']]
    if type(matches) == pd.Series:
        matches = pd.DataFrame([matches])
    num_matches = len(matches)
    if row['num_verses'] != num_matches:
        print(row['num_verses'], num_matches, row)

# save raw_df to matched_questions_gospels_raw.csv
raw_df.to_csv('data/2_sd_labeling/matched_questions_gospels_raw_out.csv', index=False)

print('Done.')
