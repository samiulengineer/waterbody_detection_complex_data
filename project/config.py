import yaml
from datetime import datetime



config = {
    # Image Input/Output
    # ----------------------------------------------------------------------------------------------
    'in_channels': 3,
    'num_classes': 2,
    'height': 512, #for PHR-CB experiment patch size = height = width
    'width': 512,  # image height and width must be same because we are patchifing square images from 11361 X 10820 image.
    'actual_height' : 10820,
    'actual_width' : 11361,
    
    # Training
    # ----------------------------------------------------------------------------------------------
    # mnet = fapnet, unet, ex_mnet*, dncnn, u2net, vnet, unet++, sm_unet, sm_linknet, sm_fpn, sm_pspnet*, kuc_vnet, kuc_unet3pp, kuc_r2unet,# kuc_unetpp*, 'kuc_restunet', kuc_tensnet*, 'kuc_swinnet', kuc_u2net, kuc_attunet, ad_unet, transformer
    'model_name': 'unet',
    'batch_size': 10,
    'epochs': 1000,
    'learning_rate': 0.0003,  #3e-4
    'val_plot_epoch': 20,
    'augment': True,
    'transfer_lr': False,
    'gpu': '3',
    
    # Experiment Setup
    # ----------------------------------------------------------------------------------------------
    # regular/cls_balance/patchify/patchify_WOC
    # cfr = regular, cfr_cb = cls_balance, phr = patchify, phr_cb = patchify_WOC
    'experiment': 'phr_cb',
    
    # Patchify (phr & phr_cb experiment)
    # ----------------------------------------------------------------------------------------------
    'patchify': True,
    'patch_class_balance': False, # whether to use class balance while doing patchify
    'patch_size': 512, # height = width, anyone is suitable
    'stride': 64,
    
    # Dataset
    # --------------------------------mask--------------------------------------------------------------
    'weights': True, # False if cfr, True if cfr_cb
    'balance_weights': [0.76, 0.24],
    'root_dir': '/home/mdsamiul/github_project/waterbody_segmentation_complex_data',
    'dataset_dir': '/home/mdsamiul/github_project/waterbody_segmentation_complex_data/data/',
    'visualization_dir': '/home/mdsamiul/github_project/waterbody_segmentation_complex_data/visualization/',
    'train_size': 0.8, 
    'train_dir': 'train.csv',
    'valid_dir': 'valid.csv',
    'test_dir': 'test.csv',
    'eval_dir': 'eval.csv',
    # Logger/Callbacks
    # ----------------------------------------------------------------------------------------------
    'csv': True,
    'val_pred_plot': True,
    'lr': True,
    'tensorboard': True,
    'early_stop': False,
    'checkpoint': True,
    'patience': 300, # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

    # Evaluation
    # ----------------------------------------------------------------------------------------------
    'load_model_name': 'unet_ex_cfr_cb_ep_1000_11-Feb-23.hdf5',
    'load_model_dir': None, #  If None, then by befault 'root_dir/model/model_name/load_model_name'
    'evaluation': False, # default evaluation value will not work
    'video_path': None, # If None, then by default 'root_dir/data/video_frame'

    # Prediction Plot
    # ----------------------------------------------------------------------------------------------
    'plot_single': True, # if True, then only index x_test image will plot # default plot_single  value will not work
    'index': -1 #170 # by default -1 means random image else specific index image provide by user
}


def get_config(args={}):
    """
    Summary:
        parsing the config.yaml file and re organize some variables
    Arguments:
        path (str): config.yaml file directory
        args (dict): dictionary of passing arguments
    Return:
        a dictonary
    """

    # Replace default values with passing values
    for key in args.keys():
        if args[key] != None:
            config[key] = args[key]

    if config['patchify']:
        config['height'] = config['patch_size']
        config['width'] = config['patch_size']

    # Merge paths
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'