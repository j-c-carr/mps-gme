import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from mps_optimizer import MPSOptimizer
from experiments import recreate_wei_fig_1, recreate_wei_fig_2, plot_qudit_states

plt.style.use("highres")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Fix seed to build the same tensors each time random is called
np.random.seed(0) 


def init_logger(name: str,
                f: Optional[str] = None,
                level: Optional[int] = logging.INFO) -> logging.Logger:
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")

    if f is not None:
        file_handler = logging.FileHandler(f)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


logger = init_logger(__name__)


def make_parser() -> argparse.Namespace:
    """Makes command line argument parser. Returns ArgumentParser"""

    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("datetime", help="datetime of exectution")
    parser.add_argument("root_dir", help="name project root directory")
    return parser.parse_args()


if __name__ == "__main__":

    args = make_parser()
    FIG_DIR = args.root_dir + "/figures"

    # recreate_wei_fig_1(f"{FIG_DIR}/{args.datetime}_wei_fig_2_exp_100steps",
    #                    steps=100)
    plot_qudit_states()
