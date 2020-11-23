import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--input_sizes', type=list, default=[512, 640, 768, 896, 1024, 1280, 1408, 1536])
    parser.add_argument('--phi', type=int, default=0, help="0、1、2...、8")
    parser.add_argument('--use_mosaic', type=bool, default=True)
    parser.add_argument('--Cosine_lr', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--smoooth_label', type=int, default=0)
    parser.add_argument('--Use_Data_Loader', type=bool, default=True)
    parser.add_argument('--train_annotation_path', type=str, default='./data/identify_violence/train.txt')
    parser.add_argument('--classes_path', type=str, default='./model_data/coco_classes.txt', help=("./data/tmp.names"))
    parser.add_argument('--pretrain_dir', type=str, default='./pretrain_weights/')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--learning_rate_first_stage', type=float, default=1e-3)
    parser.add_argument('--opt_weight_decay', type=float, default=5e-4)
    parser.add_argument('--CosineAnnealingLR_T_max', type=int, default=5)
    parser.add_argument('--CosineAnnealingLR_eta_min', type=float, default=1e-5)
    parser.add_argument('--StepLR_step_size', type=int, default=1)
    parser.add_argument('--StepLR_gamma', type=float, default=0.9)
    parser.add_argument('--learning_rate_second_stage', type=float, default=1e-4)
    parser.add_argument('--Batch_size_first_stage', type=int, default=4)
    parser.add_argument('--Batch_size_second_stage', type=int, default=8)
    parser.add_argument('--Init_Epoch', type=int, default=0)
    parser.add_argument('--Freeze_Epoch', type=int, default=30)
    parser.add_argument('--Unfreeze_Epoch', type=int, default=200)
    parser.add_argument('--Save_num_epoch', type=int, default=5)

    parser.add_argument('--save_model_path', type=str, default='model_weight', help="saving of model's path")
    # inference
    parser.add_argument('--weight_path', type=str, default='./pretrain_weights/efficientdet-d0.pth')
    parser.add_argument('--confidence', type=float, default=0.4, help="Object confidence threshold")
    parser.add_argument('--nms_thres', type=float, default=0.3)

    args = parser.parse_args()

    return args
