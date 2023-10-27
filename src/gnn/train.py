import argparse

from src.gnn.gnn import train

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    args = parser.parse_args()

    # train the GNN
    mag_path = args.input_data
    train(mag_path)
