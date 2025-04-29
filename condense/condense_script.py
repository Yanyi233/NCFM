def check_args(args):
    # 检查args中是否有define_model使用到的参数，如果没有则设置缺省值
    if not hasattr(args, 'dataset'):
        args.dataset = 'cifar10'  # 默认数据集
        if args.rank == 0:
            print(f"Warning: 'dataset' not found in args, using default: {args.dataset}")
    
    if not hasattr(args, 'norm_type'):
        args.norm_type = 'batch'  # 默认归一化类型
        if args.rank == 0:
            print(f"Warning: 'norm_type' not found in args, using default: {args.norm_type}")
    
    if not hasattr(args, 'net_type'):
        args.net_type = 'resnet'  # 默认网络类型
        if args.rank == 0:
            print(f"Warning: 'net_type' not found in args, using default: {args.net_type}")
    
    if not hasattr(args, 'nch'):
        args.nch = 3  # 默认通道数
        if args.rank == 0:
            print(f"Warning: 'nch' not found in args, using default: {args.nch}")
    
    if not hasattr(args, 'depth'):
        args.depth = 18  # 默认网络深度
        if args.rank == 0:
            print(f"Warning: 'depth' not found in args, using default: {args.depth}")
    
    if not hasattr(args, 'width'):
        args.width = 1.0  # 默认网络宽度
        if args.rank == 0:
            print(f"Warning: 'width' not found in args, using default: {args.width}")
    
    if not hasattr(args, 'nclass'):
        args.nclass = 10  # 默认类别数
        if args.rank == 0:
            print(f"Warning: 'nclass' not found in args, using default: {args.nclass}")
    
    if not hasattr(args, 'size'):
        args.size = 32  # 默认图像大小
        if args.rank == 0:
            print(f"Warning: 'size' not found in args, using default: {args.size}")
    
    if not hasattr(args, 'is_multilabel'):
        args.is_multilabel = False  # 默认为单标签分类
        if args.rank == 0:
            print(f"Warning: 'is_multilabel' not found in args, using default: {args.is_multilabel}")
    return args

def main_worker(args):
    
    args.class_list = distribute_class(args.nclass,args.debug)

    plotter = get_plotter(args)

    loader_real,_ = get_loader(args)


    aug, _ = diffaug(args)
    
    condenser = Condenser(args, nclass_list=args.class_list, nchannel=args.nch, hs=args.size, ws=args.size, device='cuda')
    for local_rank in range(args.local_world_size):
        if  args.local_rank == local_rank:
            condenser.load_condensed_data(loader_real, init_type=args.init,load_path=args.load_path)
            print(f"============RANK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Condensed Data==========================")
        dist.barrier()

    optim_img = get_optimizer(optimizer=args.optimizer, parameters=condenser.parameters(),lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    if args.sampling_net:
            sampling_net = SampleNet(feature_dim=2048)
            optim_sampling_net = get_optimizer(optimizer=args.optimizer, parameters=sampling_net,lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    else:
        sampling_net = None
        optim_sampling_net = None
    args = check_args(args)
    model_init, model_interval, model_final = get_feature_extractor(args)
    condenser.condense(args, plotter, loader_real, aug,optim_img, model_init, model_interval, model_final, sampling_net, optim_sampling_net)

    dist.destroy_process_group()



if __name__ == '__main__':
    import sys
    import os
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.diffaug import diffaug
    import torch.distributed as dist
    from  utils.ddp import distribute_class
    from  utils.utils import get_plotter,get_optimizer,get_loader,get_feature_extractor
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor
    from condenser.Condenser import Condenser
    from NCFM.SampleNet import SampleNet

    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"') 
    parser.add_argument('-i', '--ipc', type=int, default=1,required=True, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    parser.add_argument('--sampling_net', action='store_true',default=False,help='Enable sampling_net')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_script(args)

    main_worker(args)