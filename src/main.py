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
    # dc = LinkPredictionDictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
    # dc = LinkPredictionDictionaryCreator(['bid-eng-DBY', 'bid-tpi'], score_threshold=0.2)
    dc = LinkPredictionDictionaryCreator(['bid-eng-DBY', 'bid-meu'], score_threshold=0.2)
    dc.create_dictionary(load=True, save=False, plot_wtxt='water', min_count=4)
    dc._save_state()
    # dc.print_lemma_groups()
    dc._plot_subgraph(lang='eng', text='graven', min_count=4)
