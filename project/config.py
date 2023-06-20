from datetime import datetime

config = {
    # Image Input/Output
    # ----------------------------------------------------------------------------------------------
    'in_channels': 1,
    'num_classes': 2,
    'height': 512, #for PHR-CB experiment patch size = height = width
    'width': 512,  # image height and width must be same because we are patchifing square images from 11361 X 10820 image.
    'actual_height' : 10820,
    'actual_width' : 11361,
    'tiles_size' : 1024, # creating square tiles
    
    # Training
    # ----------------------------------------------------------------------------------------------
    # mnet = fapnet, unet, mimonet, ex_mnet*, dncnn, u2net, vnet, unet++, sm_unet, sm_linknet, sm_fpn, sm_pspnet*, kuc_vnet, kuc_unet3pp, kuc_r2unet,# kuc_unetpp*, 'kuc_restunet', kuc_tensnet*, 'kuc_swinnet', kuc_u2net, kuc_attunet, ad_unet, transformer
    'model_name': 'mimonet',
    'batch_size': 10,
    'epochs': 1000,
    'learning_rate': 0.0003,  #3e-4
    'val_plot_epoch': 1,
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
    'patch_class_balance': True, # whether to use class balance while doing patchify
    'patch_size': 256, # height = width, anysize is suitable
    'stride': 64,
    
    # Dataset
    # --------------------------------mask--------------------------------------------------------------
    'weights': False, # False if cfr, True if cfr_cb
    'balance_weights': [0.76, 0.24],
    'root_dir': '/mnt/hdd2/mdsamiul/waterbody_detection_complex_data',
    'dataset_dir': '/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/data/',
    'visualization_dir': '/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/visualization/',
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
    'load_model_name': 'aunet_ex_phr_cb_epochs_1000_11-May-23.hdf5',
    'load_model_dir': '/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/logs/model/unet/', #  If None, then by befault 'root_dir/model/model_name/load_model_name'
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

    experiment_name = ''

    if config['patchify']:
        config['height'] = config['patch_size']
        config['width'] = config['patch_size']

        if config["patch_class_balance"]:
            experiment_name = 'phr_cb_'
        else:
            experiment_name = 'phr_'
    else:
        config['height'] = config['tiles_size']
        config['width'] = config['tiles_size']


    config['p_train_dir'] = 'json/train_patch_' + experiment_name + str(config['patch_size']) + '.json'
    config['p_valid_dir'] = 'json/valid_patch_' + experiment_name  + str(config['patch_size']) + '.json'
    config['p_test_dir'] = 'json/test_patch_' + experiment_name  + str(config['patch_size']) + '.json'

    # Merge paths
    config['train_dir'] = config['dataset_dir'] + config['train_dir']
    config['valid_dir'] = config['dataset_dir'] + config['valid_dir']
    config['test_dir'] = config['dataset_dir'] + config['test_dir']
    
    config['p_train_dir'] = config['dataset_dir'] + config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir'] + config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir'] + config['p_test_dir']
    
    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_ex_{}_epochs_{}_{}".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir']+'/logs/'+'/tens_logger/'+config['model_name']+'/'

    config['csv_log_name'] = "{}_ex_{}_epochs_{}_{}.csv".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir']+'/logs/'+'/csv_logger/'+config['model_name']+'/'

    config['checkpoint_name'] = "{}_ex_{}_epochs_{}_{}.hdf5".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir']+'/logs/'+'/model/'+config['model_name']+'/'

    # Create save model directory
    if config['load_model_dir']=='None':
        config['load_model_dir'] = config['root_dir']+'/logs/'+'/model/'+config['model_name']+'/'
    
    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir']+'/logs/'+'/prediction/'+config['model_name']+'/test/'
    config['prediction_val_dir'] = config['root_dir']+'/logs/'+'/prediction/'+config['model_name']+'/validation/'
    
    config['visualization_dir'] = config['root_dir']+'/logs/'+'/visualization/'

    return config