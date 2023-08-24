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
        'bid-eng-web',
        'bid-fra-fob',
        'bid-ind',
        'bid-por',
        'bid-swa',
        'bid-spa',

        # non-latin
        # 'bid-mya',
        # 'bid-cmn',
        'bid-hin',
        'bid-mal',
        'bid-nep',
        # 'bid-urd',
        # 'bid-pes',

        # no semantic domains
        # 'bid-gej',
        'bid-deu',
        # 'bid-yor',
        # 'bid-tpi',
        # 'bid-meu',
    ],
        score_threshold=0.2)
    dc.create_dictionary(load=False, save=True, plot_wtxt='neither', min_count=4, print_reciprocal_ranks=True,
                         plot_subgraph=False)

    for verse_id in range(25042, 25047):  # Lk 2:1-5
        dc._plot_subgraph(lang='eng', text='graven', min_count=1, verse_id=verse_id)
    dc._plot_subgraph(lang='eng', text='book', min_count=4)
