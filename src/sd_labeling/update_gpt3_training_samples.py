import pandas as pd
from tqdm import tqdm


def lowercase_question(prompt):
    # assert that there are at exactly 4 quotation marks
    assert prompt.count('"') == 4

    # lowercase everything between the 1st and 4th quotation mark in the prompt
    qm_1 = prompt.find('"')
    qm_2 = prompt.find('"', qm_1 + 1)
    qm_3 = prompt.find('"', qm_2 + 1)
    qm_4 = prompt.find('"', qm_3 + 1)
    # assert prompt[qm_4 + 2] == '('
    return prompt[:qm_1] + prompt[qm_1:qm_4].lower() + prompt[qm_4:]


if __name__ == '__main__':
    # load data/2_sd_labeling/fine-tuning/data.jsonl
    df_train = pd.read_json("data/2_sd_labeling/fine-tuning/data.jsonl", lines=True)

    # # load data/2_sd_labeling/matched_questions.xlsx into a dataframe
    # df_main = pd.read_excel("data/2_sd_labeling/matched_questions.xlsx", usecols=["direct_question", "old_question"])

    # # use direct_question as index
    # df_main.set_index('direct_question', inplace=True)
    #
    # # for each row in df_train, replace "prompt" with the corresponding "qid" from df_main
    # for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
    #     direct_question = row['prompt'].replace(' ->', '?')
    #     if direct_question in df_main.index:
    #         df_train.at[index, 'qid'] = df_main.at[direct_question, 'qid']
    #     else:
    #         df_train.at[index, 'qid'] = '?'

    # # use old_question as index
    # df_main.set_index('old_question', inplace=True)
    #
    # # for each row in df_train, replace "prompt" with the corresponding "direct_question" from df_main
    # for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
    #     old_question = row['prompt'].replace(' ->', '?')
    #     if old_question in df_main.index:
    #         new_question = df_main.at[old_question, 'direct_question'][:-1] + ' ->'
    #         df_train.at[index, 'prompt'] = new_question
    #     else:
    #         print(f"Error: question not found: '{old_question}'")
    #         df_train.at[index, 'prompt'] = '??? ' + old_question

    for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
        df_train.at[index, 'prompt'] = lowercase_question(row['prompt'])

    # save df_train
    df_train.to_json("data/2_sd_labeling/fine-tuning/data_updated.jsonl", orient='records', lines=True)
    print('Done.')
