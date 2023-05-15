from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
import pandas as pd
from dictionary_creator.link_prediction_dictionary_creator import LinkPredictionDictionaryCreator
from semantic_domain_identifier import SemanticDomainIdentifier
import dill
from nltk.metrics.distance import edit_distance
import gensim.downloader as api


def correlate_domains(lexical_domains, sdi):
    # load matched_questions_gospels_raw.csv
    sd_df = pd.read_csv('data/2_sd_labeling/matched_questions_gospels_raw_answered.csv',
                        usecols=['qid', 'verse_id', 'answer'])

    # filter questions with answer == 1
    sd_df = sd_df[sd_df['answer'] == 1]

    # aggregate sd_df by verse_id
    sd_df = sd_df.groupby(['verse_id']).agg(list)

    # use the verse_id as index
    sd_df = sd_df.reset_index().set_index('verse_id')

    # replace MRK 16:99 with MRK 16:20
    for index, ld in lexical_domains.iterrows():
        if ld['book'] == 'MRK' and ld['chapter'] == 16 and ld['verse'] == 99:
            lexical_domains.loc[ld.name, 'verse'] = 20

    # get list of all verse_ids
    verse_ids = sd_df.index.unique()

    # add column verse_id to lexical_domains
    lexical_domains['verse_id'] = lexical_domains.apply(
        lambda row: sdi.convert_bible_reference_to_verse_id(row['book'], row['chapter'], row['verse']), axis=1)

    # filter lexical_domains by verse_ids
    lexical_domains = lexical_domains[lexical_domains['verse_id'].isin(verse_ids)]

    correlated_lds_by_cid = defaultdict(Counter)  # count the number of correlated LDs for each CIDs
    correlated_cids_by_ld = defaultdict(Counter)  # count the number of correlated CIDs for each LD

    # for each lexical domain, lookup all cids that are correlated to it
    for index, ld in tqdm(lexical_domains.iterrows(), total=lexical_domains.shape[0], desc='Correlating domains...'):
        verse_id = ld['verse_id']
        lexical_domain_name = ld['domains']
        qids = sd_df.loc[verse_id]['qid']
        cids = [qid.split(' ')[0] for qid in qids]
        for cid in cids:
            correlated_lds_by_cid[cid][lexical_domain_name] += 1
            correlated_lds_by_cid[cid]['# total CID verses #'] += 1
            correlated_cids_by_ld[lexical_domain_name][cid] += 1
            correlated_cids_by_ld[lexical_domain_name]['# total LD verses #'] += 1

    correlation_scores_by_ld_by_cid = defaultdict(dict)  # product
    for cid, lds in correlated_lds_by_cid.items():
        for ld, count in lds.items():
            if ld == '# total CID verses #':
                continue
            ld_prob = correlated_lds_by_cid[cid][ld] / correlated_lds_by_cid[cid]['# total CID verses #']
            cid_prob = correlated_cids_by_ld[ld][cid] / correlated_cids_by_ld[ld]['# total LD verses #']
            correlation_scores_by_ld_by_cid[cid][ld] = (ld_prob * cid_prob, ld_prob, cid_prob)

    # sort correlated_lds_by_cid by correlation score
    for cid, lds in correlation_scores_by_ld_by_cid.items():
        correlation_scores_by_ld_by_cid[cid] = sorted(lds.items(), key=lambda x: x[1][0], reverse=True)

    # save the results
    with open('data/2_sd_labeling/correlated_lds_by_cid.pkl', 'wb') as f:
        dill.dump(correlated_lds_by_cid, f)
    with open('data/2_sd_labeling/correlated_cids_by_ld.pkl', 'wb') as f:
        dill.dump(correlated_cids_by_ld, f)
    with open('data/2_sd_labeling/correlation_scores_by_ld_by_cid.pkl', 'wb') as f:
        dill.dump(correlation_scores_by_ld_by_cid, f)


def compute_word_vector_distance(string1, string2, word_vectors):
    # remove punctuation
    string1 = string1.replace(',', '')
    string2 = string2.replace(',', '')

    # tokenize strings
    string1 = string1.lower().split()
    string2 = string2.lower().split()

    # remove words that are not in word_vectors
    string1 = [word for word in string1 if word in word_vectors]
    string2 = [word for word in string2 if word in word_vectors]

    # average word vectors
    vector1 = np.mean([word_vectors[word] for word in string1], axis=0)
    vector2 = np.mean([word_vectors[word] for word in string2], axis=0)

    # compute distance between 2 ndarrays
    return np.linalg.norm(vector1 - vector2)


def correlate_domain_names(dc):
    # load results
    # with open('data/2_sd_labeling/correlated_lds_by_cid.pkl', 'rb') as f:
    #     correlated_lds_by_cid = dill.load(f)
    # with open('data/2_sd_labeling/correlated_cids_by_ld.pkl', 'rb') as f:
    #     correlated_cids_by_ld = dill.load(f)
    with open('data/2_sd_labeling/correlation_scores_by_ld_by_cid.pkl', 'rb') as f:
        correlation_scores_by_ld_by_cid = dill.load(f)

    # load names for CIDs from dc
    sd_name_by_cid = {}
    sd_df = dc.sds_by_lang['eng']
    for cid in correlation_scores_by_ld_by_cid.keys():
        sd_name_by_cid[cid] = list(sd_df[sd_df['cid'] == cid]['category'])[0]

    # add SD names to correlation_scores_by_ld_by_cid keys
    correlation_scores_by_ld_by_sd = defaultdict(dict)
    for cid, lds in correlation_scores_by_ld_by_cid.items():
        for ld, score in lds:
            correlation_scores_by_ld_by_sd[cid + ' ' + sd_name_by_cid[cid]][ld] = score

    # compute edit distance between sd names and ld names
    edit_distances = defaultdict(dict)
    for sd_name, lds in correlation_scores_by_ld_by_sd.items():
        for ld, score in lds.items():
            pure_sd_name = ' '.join(sd_name.split(' ')[1:])
            edit_distances[sd_name][ld] = edit_distance(pure_sd_name, ld) / max(len(pure_sd_name), len(ld))

    # sort edit_distances by edit distance
    for sd_name, lds in edit_distances.items():
        edit_distances[sd_name] = sorted(lds.items(), key=lambda x: x[1], reverse=False)

    # filter out lds with edit distance > 0.2
    for sd_name, lds in edit_distances.items():
        edit_distances[sd_name] = [(ld, score) for ld, score in lds if 0.6 <= score and score < 0.7]

    # filter out empty SDs
    edit_distances = {sd_name: lds for sd_name, lds in edit_distances.items() if len(lds) > 0}

    # sort by sd_name
    edit_distances = sorted(edit_distances.items(), key=lambda x: x[0], reverse=False)

    # write edit_distances to txt file
    with open('data/2_sd_labeling/edit_distances_2.txt', 'w') as f:
        for sd_name, lds in edit_distances:
            f.write(sd_name + '\n')
            for ld, score in lds:
                f.write('\t' + ld + '\t' + str(score) + '\n')
            f.write('\n')

    # compute distances between SDs and LDs with word vectors
    # load word vectors
    word_vectors = api.load('word2vec-google-news-300')

    # compute distances
    distances = defaultdict(dict)
    for sd_name, lds in correlation_scores_by_ld_by_sd.items():
        for ld, score in lds.items():
            pure_sd_name = ' '.join(sd_name.split(' ')[1:])
            distances[sd_name][ld] = compute_word_vector_distance(pure_sd_name, ld, word_vectors)

    # filter out lds with distance > 2
    for sd_name, lds in distances.items():
        distances[sd_name] = [(ld, score) for ld, score in lds.items() if score <= 2]

    # sort distances by distance
    for sd_name, lds in distances.items():
        distances[sd_name] = sorted(lds, key=lambda x: x[1], reverse=False)

    # filter out empty SDs
    distances = {sd_name: lds for sd_name, lds in distances.items() if len(lds) > 0}

    # sort by sd_name
    distances = sorted(distances.items(), key=lambda x: x[0], reverse=False)

    # write distances to txt file
    with open('data/2_sd_labeling/word_vector_distances.txt', 'w') as f:
        for sd_name, lds in distances:
            f.write(sd_name + '\n')
            for ld, score in lds:
                f.write('\t' + ld + '\t' + str(score) + '\n')
            f.write('\n')


def load_sd_ld_mapping(code_by_ld):
    # load sd_ld_mapping.xlsx
    sd_ld_mapping = pd.read_excel('data/2_sd_labeling/sd_ld_mapping.xlsx', engine='openpyxl')

    # add code column
    sd_ld_mapping['lexical domain code'] = sd_ld_mapping['lexical domain name'].apply(lambda x: code_by_ld.loc[x])

    # add all missing LDs from code_by_ld to dataframe sd_ld_mapping
    for ld in code_by_ld.index:
        if ld not in sd_ld_mapping['lexical domain name'].values:
            sd_ld_mapping = pd.concat(
                [sd_ld_mapping, pd.DataFrame([[ld, '', code_by_ld.loc[ld][0]]],
                                             columns=['lexical domain name', 'semantic domain name',
                                                      'lexical domain code'])])

    # make unique
    sd_ld_mapping = sd_ld_mapping.drop_duplicates()

    # save sd_ld_mapping to csv file
    sd_ld_mapping.to_csv('data/2_sd_labeling/sd_ld_mapping.csv', index=False)


def split_semantic_domain_column_into_name_and_code(sd_ld_mapping):
    # e.g., "1.1.2 Air" -> "1.1.2" and "Air"
    # only for values that are not nan
    sd_ld_mapping['cid'] = sd_ld_mapping['semantic domain name'].apply(
        lambda x: x.split(' ')[0] if type(x) == str else x)
    sd_ld_mapping['semantic domain name'] = sd_ld_mapping['semantic domain name'].apply(
        lambda x: ' '.join(x.split(' ')[1:]) if type(x) == str else x)


def fill_missing_semantic_domain_names(dc, sd_ld_mapping):
    # for each missing SD name, look if there is an SD with the same name
    missing_sds = sd_ld_mapping[sd_ld_mapping['semantic domain name'].isna()]
    for index, row in missing_sds.iterrows():
        ld_name = row['lexical domain name']
        sd_df = dc.sds_by_lang['eng']
        if ld_name in sd_df['category'].values:
            # fill in the missing sd_name
            sd_ld_mapping.loc[index, 'semantic domain name'] = ld_name
            sd_ld_mapping.loc[index, 'semantic domain code'] = sd_df[sd_df['category'] == ld_name]['cid'].values[0]

    # filter all missing_sds that have a comma in the LD name
    missing_sds = missing_sds[missing_sds['lexical domain name'].str.contains(',')]

    # if the word before or after the comma matches an SD name, fill in the missing sd_name
    for index, row in missing_sds.iterrows():
        sd_df = dc.sds_by_lang['eng']

        print(row['lexical domain name'])
        ld_names = row['lexical domain name'].split(',')
        ld_names = [ld_name.strip() for ld_name in ld_names]

        for ld_name in ld_names:
            if ld_name in sd_df['category'].values:
                # fill in the missing sd_name
                sd_ld_mapping.loc[index, 'semantic domain name'] = ld_name
                sd_ld_mapping.loc[index, 'semantic domain code'] = sd_df[sd_df['category'] == ld_name]['cid'].values[0]
                print('  Match: ' + ld_name)


def add_missing_sd_codes(dc, sd_ld_mapping):
    sd_df = dc.sds_by_lang['eng']
    # sd_df[sd_df['category'] == sd_ld_mapping['semantic domain name']]['cid']
    for index, row in sd_ld_mapping.iterrows():
        if type(row['semantic domain name']) == str:
            sd_ld_mapping.loc[index, 'semantic domain code'] = sd_df[sd_df['category'] == row['semantic domain name']][
                'cid'].values[0]


def correlate_sds_and_lds_by_words(sd_dataframe, words_by_domain, code_by_ld):
    sd_ld_mapping = pd.read_excel('data/2_sd_labeling/sd_ld_mapping.xlsx', engine='openpyxl')

    for sd_name, sd in sd_dataframe.iterrows():
        sd_words = sd['answer']
        sd_words = ', '.join(sd_words).split(',')
        sd_words = set([word.strip() for word in sd_words])
        if '' in sd_words:
            sd_words.remove('')
        # iterate over words_by_domain (pandas series)
        for ld_name, ld_words in words_by_domain.items():
            sd_ld_mapping.loc[sd_ld_mapping['lexical domain name'] == ld_name, 'lexical domain words'] = ', '.join(
                ld_words)
            intersection = sd_words.intersection(ld_words)
            if len(intersection) >= 1:
                print(len(intersection), intersection, sd_name, '; ', ld_name)
            if len(intersection) >= 3:
                # if this lexical domain has no sd yet
                if pd.isna(
                        sd_ld_mapping[sd_ld_mapping['lexical domain name'] == ld_name]['semantic domain name'].values[
                            0]):
                    sd_ld_mapping.loc[sd_ld_mapping['lexical domain name'] == ld_name, 'semantic domain name'] = sd_name
                    sd_ld_mapping.loc[sd_ld_mapping['lexical domain name'] == ld_name, 'overlap'] = ', '.join(
                        intersection)
                else:
                    # add line
                    sd_ld_mapping = pd.concat(
                        [sd_ld_mapping,
                         pd.DataFrame([[ld_name, sd_name, code_by_ld.loc[ld_name][0], ', '.join(intersection)]],
                                      columns=['lexical domain name',
                                               'semantic domain name',
                                               'lexical domain code',
                                               'overlap'])])


    sd_ld_mapping.to_csv('data/2_sd_labeling/sd_ld_mapping.csv', index=False)


if __name__ == '__main__':
    LinkPredictionDictionaryCreator.BIBLES_BY_BID.update({
        'bid-eng-DBY-1000': '../../../dictionary_creator/test/data/eng-engDBY-1000-verses.txt',
        'bid-fra-fob-1000': '../../../dictionary_creator/test/data/fra-fra_fob-1000-verses.txt',
    })

    dc = LinkPredictionDictionaryCreator(['bid-eng-DBY-1000', 'bid-fra-fob-1000'], score_threshold=0.2)
    dc._preprocess_data()

    sdi = SemanticDomainIdentifier(dc)

    # correlate_domain_names(dc)

    # Load the lexical domain data
    # format: for each word in each verse: lexical domains
    # ot_domains = pd.read_parquet('data/OT_domains.parquet', engine='fastparquet')
    domains = pd.read_parquet('data/NT_domains.parquet', engine='fastparquet')

    # append the two dataframes
    # domains = pd.concat([ot_domains, nt_domains])

    # create a mapping from domains to domain_codes
    # remove all domain_codes and domains but the first in each line
    code_by_domain = domains[['domain_codes', 'domains']]
    code_by_domain['domain_codes'] = code_by_domain['domain_codes'].apply(lambda x: x[0])
    code_by_domain['domains'] = code_by_domain['domains'].apply(lambda x: x[0])
    code_by_domain = code_by_domain.drop_duplicates(subset=['domain_codes', 'domains'], keep='first')
    # code_by_domain = pd.concat([code_by_domain, pd.DataFrame([['Rebuke', '033046']], columns=['domains', 'domain_codes'])])
    code_by_domain = code_by_domain.set_index('domains')
    code_by_domain = code_by_domain.rename(columns={'domain_codes': 'domain_code'})
    code_by_domain = code_by_domain.sort_values(by=['domain_code'], ascending=True)
    # load_sd_ld_mapping(code_by_domain)

    # filter all lines that contain the word "moon"
    moon_lines = domains[domains['english'] == 'moon']
    print(moon_lines)

    # remove column 'domain_codes'
    domains = domains.drop(columns=['domain_codes'])
    # split domain list
    domains = domains.explode('domains').reset_index(drop=True)
    # aggregate the words by domains
    words_by_domain = domains.groupby(['domains']).agg(set)
    # filter column english
    words_by_domain = words_by_domain['english']
    print(words_by_domain)

    # Count distinct domains
    print(domains['domains'].nunique())  # 1020 (OT + NT), 669 (NT)

    # correlate_domains(domains, sdi)

    # match LDs with SDs by finding word overlaps
    sd_dataframe = dc.sds_by_lang['eng']
    sd_dataframe['answer'] = sd_dataframe['answer'].apply(lambda x: [i.strip().lower() for i in x.split('/')])
    sd_dataframe = sd_dataframe.explode('answer').reset_index(drop=True)
    sd_dataframe = sd_dataframe.groupby(['category']).agg(set)
    correlate_sds_and_lds_by_words(sd_dataframe, words_by_domain, code_by_domain)
