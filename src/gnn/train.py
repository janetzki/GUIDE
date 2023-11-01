import argparse

from src.gnn.gnn import train

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mag-file', type=str, required=True)
    parser.add_argument('--output-model-file', type=str, required=True)
    parser.add_argument('--output-data-split-file', type=str, required=True)
    args = parser.parse_args()

    # train the GNN
    mag_path = args.input_mag_file
    output_model_file = args.output_model_file
    output_data_split_file = args.output_data_split_file
    train(mag_path, output_model_file, output_data_split_file)
