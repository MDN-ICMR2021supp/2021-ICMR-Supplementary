import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.00004)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=10)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8)
    parser.add_argument(
        '--step_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=2)
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=8)

    
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--output', type=str, default="./output/default")
    
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=256) 
    parser.add_argument(
        '--feature_path',
        type=str,
        default="./data/thumos_feature/TSN_pretrain_avepool_allfrms_hdf5")

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)


    
    parser.add_argument('--max_duration', type=int, default=64) 
    parser.add_argument('--min_duration', type=int, default=0)  

    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5,
        help='the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )


    parser.add_argument(
           '--goi_samp',
           type=int,
           default=0)
    parser.add_argument(
           '--goi_style',
           type=int,
           default=1)

    parser.add_argument(
        '--kern_2d',
        type=int,
        default=3)
    parser.add_argument(
        '--pad_2d',
        type=int,
        default=1)


    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.46)


    parser.add_argument(
        '--override', default=False, action='store_true',
        help='Prevent use of cached data'
    )

    args = parser.parse_args()

    return args

