import os

from src.dictionary_creator.link_prediction_dictionary_creator import LinkPredictionDictionaryCreator

if __name__ == '__main__':  # pragma: no cover
    LinkPredictionDictionaryCreator.BIBLES_BY_BID.update({
        'bid-eng-DBY-1000': '../../../dictionary_creator/test/data/eng-engDBY-1000-verses.txt',
        'bid-fra-fob-1000': '../../../dictionary_creator/test/data/fra-fra_fob-1000-verses.txt',
    })

    # set the working directory to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.chdir(project_root)

    # dc = LinkPredictionDictionaryCreator(['bid-eng-DBY-1000', 'bid-fra-fob-1000'], score_threshold=0.2)
    # dc = LinkPredictionDictionaryCreator(['bid-eng-DBY', 'bid-fra-fob', 'bid-tpi', 'bid-meu'], score_threshold=0.2)
    dc = LinkPredictionDictionaryCreator([
        # LATIN BIBLES WITH SEMANTIC DOMAINS
        'bid-eng-web',
        'bid-fra-sbl',
        'bid-ind',
        'bid-por-bsl',
        'bid-swh-ulb',
        # 'bid-swh-onen',
        'bid-spa-blm',

        # NON-LATIN BIBLES WITH SEMANTIC DOMAINS
        'bid-hin',
        'bid-mal',
        'bid-npi',
        'bid-ben',
        'bid-mkn',
        'bid-cmn-s',

        # # 'bid-arb-vd',
        # # 'bid-arb-nav',
        'bid-pes',
        'bid-urd',
        'bid-tel',
        # # 'bid-mya', # Burmese
        # # unused langs: tha

        # BIBLES WITHOUT SEMANTIC DOMAINS
        'bid-deu-1951',
        'bid-azb',
        'bid-ibo',
        'bid-gej',
        'bid-yor',
        'bid-tpi',
        'bid-meu',
        'bid-hmo',
    ],
        score_threshold=0.2,
        gt_langs=['eng', 'fra', 'spa', 'por', 'ind', 'swh', 'hin', 'mal', 'npi', 'ben', 'mkn', 'cmn', 'pes', 'urd'])
    dc.create_dictionary(load=False, save=True, plot_wtxt='neither', min_count=4, print_reciprocal_ranks=True,
                         plot_subgraph=False)

    dc._plot_subgraph(lang='eng', text='book', min_count=4)
