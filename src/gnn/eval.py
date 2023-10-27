import argparse

from src.gnn.gnn import eval

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--data-split', type=str, required=True)
    args = parser.parse_args()

    # evaluate the GNN
    model_file = args.model_file
    data_split = args.data_split
    mag_path = args.input_data
    eval(model_file, mag_path, data_split)
