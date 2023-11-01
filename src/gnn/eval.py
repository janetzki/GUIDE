import argparse

from src.gnn.gnn import evaluate

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mag-file', type=str, required=True)
    parser.add_argument('--input-model-file', type=str, required=True)
    parser.add_argument('--input-data-split-file', type=str, required=True)
    parser.add_argument('--output-results-file', type=str, required=True)
    args = parser.parse_args()

    # evaluate the GNN
    mag_path = args.input_mag_file
    model_file = args.input_model_file
    data_split = args.input_data_split_file
    results_output_file = args.output_results_file
    evaluate(mag_path, model_file, data_split, results_output_file)
