from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/got10k_lmdb'
    settings.got10k_path = '/data1/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/data1/lasot_lmdb'
    settings.lasot_path = '/data1/lasot'
    settings.network_path = '/home/yufan/videotrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/data1/OTB100'
    settings.prj_dir = '/home/yufan/videotrack'
    settings.result_plot_path = '/home/yufan/videotrack/output/test/result_plots'
    settings.results_path = '/home/yufan/videotrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/yufan/videotrack/output'
    settings.segmentation_path = '/home/yufan/videotrack/output/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/trackingnet'
    settings.uav_path = '/data1/UAV123'
    settings.vot_path = '/data1/VOT2019'
    settings.youtubevos_dir = ''

    return settings

