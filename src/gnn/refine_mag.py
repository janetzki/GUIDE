import argparse

from src.gnn.gnn import refine_mag

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mag-directory', type=str, required=True)
    parser.add_argument('--output-mag-file', type=str, required=True)
    args = parser.parse_args()

    # refine the MAG
    input_mag_path = args.input_mag_directory
    output_mag_file = args.output_mag_file
    refine_mag(input_mag_path, output_mag_file)
