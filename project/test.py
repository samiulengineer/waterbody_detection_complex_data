import os
import argparse
import time
from metrics import *
from tensorflow import keras
from tensorflow.keras.models import load_model

from loss import *
from config import get_config
from dataset import get_test_dataloader
from utils import create_paths, patch_show_predictions, frame_to_video, show_predictions



# Parsing variable
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name")
parser.add_argument("--plot_single")
parser.add_argument("--index", type=int)
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size")
parser.add_argument("--gpu")
parser.add_argument("--evaluation")
parser.add_argument("--video_path")
args = parser.parse_args()

if args.plot_single == 'True': # by default config will not work
    args.plot_single = True
else:
    args.plot_single = False
    
if args.evaluation == 'True': # by default config will not work
    args.evaluation = True
else:
    args.evaluation = False

t0 = time.time()
# Set up test configaration
# ----------------------------------------------------------------------------------------------
config = get_config(vars(args))

if config["evaluation"]:
    create_paths(config, eval = True)
else:
    create_paths(config, test = True)



# setup gpu
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load Model
# ----------------------------------------------------------------------------------------------
print("Loading model {} from {}".format(
    config['load_model_name'], config['load_model_dir']))
# with strategy.scope(): # if multiple GPU is required
model = load_model(os.path.join(
    config['load_model_dir'], config['load_model_name']), compile=False)


# Dataset
# ----------------------------------------------------------------------------------------------
test_dataset = get_test_dataloader(config)

# Prediction Plot
# ----------------------------------------------------------------------------------------------
print("Saving test/evaluation predictions...")
print("call patch_show_predictions")
if config['patchify']:
    patch_show_predictions(test_dataset, model, config)
else:
    show_predictions(test_dataset, model, config)


# Evaluation Score
# ----------------------------------------------------------------------------------------------
if not config['evaluation']:
    metrics = list(get_metrics(config).values())
    adam = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=adam, loss=focal_loss(), metrics=metrics)
    model.evaluate(test_dataset)

# Frame to video
# ----------------------------------------------------------------------------------------------
if config["video_path"] != 'None':
    fname = config['dataset_dir'] + 'prediction.avi'
    frame_to_video(config, fname, fps=30)


print("training time sec: {}".format((time.time()-t0)))