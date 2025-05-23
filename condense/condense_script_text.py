import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.diffaug import diffaug
import torch.distributed as dist
from utils.ddp import distribute_class
from utils.utils_text import get_plotter, get_optimizer, get_loader, get_feature_extractor
from utils.init_script import init_script
import argparse
from argsprocessor.args import ArgsProcessor
from condenser.Condenser_text import Condenser
from NCFM.SampleNet import SampleNet

def check_args(args):
    # 检查args中是否有define_model使用到的参数，如果没有则设置缺省值
    if not hasattr(args, 'dataset'):
        args.dataset = 'reuters'  # 默认数据集
        if args.rank == 0:
            print(f"Warning: 'dataset' not found in args, using default: {args.dataset}")

    if not hasattr(args, 'norm_type'):
        args.norm_type = 'layernorm'  # 默认归一化类型
        if args.rank == 0:
            print(f"Warning: 'norm_type' not found in args, using default: {args.norm_type}")
    
    if not hasattr(args, 'net_type'):
        args.net_type = 'BERT'  # 默认网络类型
        if args.rank == 0:
            print(f"Warning: 'net_type' not found in args, using default: {args.net_type}")
    
    if not hasattr(args, 'depth'):
        args.depth = 12  # 默认网络深度
        if args.rank == 0:
            print(f"Warning: 'depth' not found in args, using default: {args.depth}")
    
    if not hasattr(args, 'nclass'):
        args.nclass = 90  # 默认类别数
        if args.rank == 0:
            print(f"Warning: 'nclass' not found in args, using default: {args.nclass}")
    
    if not hasattr(args, 'max_length'):
        args.max_length = 512  # 默认文本最大长度
        if args.rank == 0:
            print(f"Warning: 'max_length' not found in args, using default: {args.max_length}")
    
    if not hasattr(args, 'is_multilabel'):
        args.is_multilabel = True  # 默认为多标签分类
        if args.rank == 0:
            print(f"Warning: 'is_multilabel' not found in args, using default: {args.is_multilabel}")
    
    return args

def main_worker(args):
    
    args.class_list = distribute_class(args.nclass, args.debug)

    plotter = get_plotter(args)

    loader_real,_ = get_loader(args)


    # aug, _ = diffaug(args)
    aug = None
    
    # 初始化Condenser
    condenser = Condenser(args, nclass_list=args.class_list, device='cuda')
    
    # 加载合成数据
    for local_rank in range(args.local_world_size):
        if args.local_rank == local_rank:
            condenser.load_condensed_data(
                loader_real, 
                init_type=args.init,
                load_path=args.load_path
            )
            print(f"============RANK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Condensed Data==========================")
        dist.barrier()
    
    # 设置优化器
    optim_text = get_optimizer(
        optimizer=args.optimizer, 
        parameters=condenser.parameters(),
        lr=args.lr_img, 
        mom_img=args.mom_img,
        weight_decay=args.weight_decay,
        logger=args.logger
    )
    
    # 设置采样网络（如果需要）
    if args.sampling_net:
        sampling_net = SampleNet(feature_dim=768)  # BERT的隐藏维度是768
        optim_sampling_net = get_optimizer(
            optimizer=args.optimizer, 
            parameters=sampling_net,
            lr=args.lr_img, 
            mom_img=args.mom_img,
            weight_decay=args.weight_decay,
            logger=args.logger
        )
    else:
        sampling_net = None
        optim_sampling_net = None
    
    # 检查参数
    args = check_args(args)
    
    # 获取特征提取器
    model_init, model_interval, model_final = get_feature_extractor(args)
    condenser.condense(args, plotter, loader_real, aug, optim_text, model_init, model_interval, model_final, sampling_net, optim_sampling_net)

    dist.destroy_process_group()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"') 
    parser.add_argument('-i', '--ipc', type=int, default=10, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    parser.add_argument('--sampling_net', action='store_true',default=False,help='Enable sampling_net')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)
    args = args_processor.add_args_from_yaml(args)
    
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    init_script(args)
    # 格式化args输出
    args_str = "Experiment Configuration:\n"
    for arg_name, arg_value in sorted(vars(args).items()):
        if not arg_name.startswith('_'):  # 跳过私有属性
            args_str += f"  {arg_name}: {arg_value}\n"
    if args.rank == 0:
        args.logger(args_str)

    main_worker(args)