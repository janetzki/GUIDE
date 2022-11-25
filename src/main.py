import os

from src.dictionary_creator.link_prediction_dictionary_creator import LinkPredictionDictionaryCreator

if __name__ == '__main__':  # pragma: no cover
    # set the working directory to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.chdir(project_root)

    # dc = LinkPredictionDictionaryCreator(['bid-eng-DBY-1000', 'bid-fra-fob-1000'], score_threshold=0.2)
    dc = LinkPredictionDictionaryCreator(['bid-eng-DBY', 'bid-fra-fob'], score_threshold=0.2)
    dc.create_dictionary(load=True, save=True, plot_wtxt='river')
