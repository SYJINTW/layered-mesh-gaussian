from argparse import ArgumentParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--progressive_policy', type=str, default="uniform")
    parser.add_argument('--total_iterations', type=int, default=30000)
    parser.add_argument('--total_splats', type=int, default=30000)
    args = parser.parse_args()
    