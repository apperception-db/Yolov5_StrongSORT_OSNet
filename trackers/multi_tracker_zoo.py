import yaml
from pathlib import Path
from .strong_sort.strong_sort import StrongSORT


FILE = Path(__file__).resolve()


def create_tracker(tracker_type, appearance_descriptor_weights, device, half):
    if tracker_type == 'strongsort':
        # initialize StrongSORT
        with open(FILE.parent / 'strong_sort/configs/strong_sort.yaml', 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        strongsort = StrongSORT(
            appearance_descriptor_weights,
            device,
            half,
            max_dist=cfg['STRONGSORT']['MAX_DIST'],
            max_iou_distance=cfg['STRONGSORT']['MAX_IOU_DISTANCE'],
            max_age=cfg['STRONGSORT']['MAX_AGE'],
            n_init=cfg['STRONGSORT']['N_INIT'],
            nn_budget=cfg['STRONGSORT']['NN_BUDGET'],
            mc_lambda=cfg['STRONGSORT']['MC_LAMBDA'],
            ema_alpha=cfg['STRONGSORT']['EMA_ALPHA'],

        )
        return strongsort
    else:
        print('No such tracker')
        exit()