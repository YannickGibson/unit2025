from dataset import *
import os
import datetime
import re
from pathlib import Path
import argparse
import npfl138
import torch
npfl138.require_version("2425.5")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


DATA_PATH = Path("/mount/data/preprocessed_dataset")
TRAIN_PATH = DATA_PATH / "train"
VAL_PATH = DATA_PATH / "val"


class DummyNet(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self._model = torch.nn.Sequential(

        )

def main(args: argparse.Namespace):
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))



    info_list = list(ADDITIONAL_INFO_DICT.keys())
    ds_train = SeqGreenEarthNetDataset(
        folder=TRAIN_PATH,
        input_channels=["red", "green", "blue"],
        target_channels=["ndvi", "class"],
        additional_info_list=info_list,
        time=True,
        use_mask=True,
        return_filename=True,
    )
    ds_val = SeqGreenEarthNetDataset(
        folder=VAL_PATH,
        input_channels=["red", "green", "blue"],
        target_channels=["ndvi", "class"],
        additional_info_list=info_list,
        time=True,
        use_mask=True,
        return_filename=True,
    )


    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    os.makedirs(args.logdir, exist_ok=True)

    test_img = dl_train[0]["inputs"]
    print(test_img)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)