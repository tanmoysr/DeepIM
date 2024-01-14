import argparse

parser = argparse.ArgumentParser(description="GenIM")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5']
parser.add_argument("-d", "--dataset", required=False, default="netscience", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", required=False, default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20] # In dataset 10, 50, 100, 200
parser.add_argument("-sp", "--seed_rate", required=False, default=1, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", required=False, default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
run_mode = ['train', 'test']
parser.add_argument("-rm", "--run_mode", required=False, default="test", type=str,
                    help="one of: {}".format(", ".join(sorted(run_mode))))
parser.add_argument("-e", "--numEpoch", required=False, default=600, type=int,
                    help="Number of Epochs") # 600
parser.add_argument("-r", "--inferRange", required=False, default=300, type=int,
                    help="Number of inference range") # 300
# args = parser.parse_args()
args = parser.parse_args(args=[]) # use it for jupyter notebook