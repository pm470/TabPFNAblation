"""Generate synthetic prior data to an HDF5 file.
Currently, this is a placeholder since the project uses a static downloaded file (300k_150x5_2.h5).
Future work: integrate NanopriorDataset to dump HDF5 locally.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="300k_150x5_2.h5")
    # Add generation params here when implementing
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Data generation is currently handled by the static downloaded file: {args.output_file}")
    print("In the future, this script will iterate over NanopriorDataset and dump to HDF5.")

if __name__ == "__main__":
    main()
