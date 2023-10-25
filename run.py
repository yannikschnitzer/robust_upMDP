from UI.get_args import run as get_args
from main.interval import run as int_run
from main.individual import run_all
from main.solvers import run_all as robust_run
from main.sampler import get_samples
import datetime
import sys

def main():
    args = get_args()
    samples = get_samples(args)
    start = datetime.datetime.now().isoformat().split('.')[0]
    if args["file_write"]:
        sys.stdout = open(args["file_write"],'wt')
    int_run(args, samples)
    run_all(args, samples)
    robust_run(args, samples)

if __name__=="__main__":
    main()

