import logging
from utils import train, evaluate
from dataset_pm25 import get_dataloader
from main_model_update import CD2_PM25
import argparse
import torch
import datetime
import json
import yaml
import os

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    filename='train_'+current_time+'.log',
    filemode='w',
)

logging.info("training with no detach")

parser = argparse.ArgumentParser(description="cd2")
parser.add_argument("--config", type=str, default="basepm.yaml")
parser.add_argument('--device', default='cuda:7', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")

args = parser.parse_args()
logging.info(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
# config["model"]["test_missing_ratio"] = args.testmissingratio
# config["model"]["target_stragety"] = args.targetstrategy

logging.info(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = (
    "./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)
logging.info('model folder:%s', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args.device, validindex=args.validationindex
)
model = CD2_PM25(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load(
        "./save/" + args.modelfolder + "/model_399.pth"))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
