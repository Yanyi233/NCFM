import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils_text import update_feature_extractor
from utils.ddp import gather_save_visualize, sync_distributed_metric
from NCFM.NCFM import match_loss, cailb_loss, mutil_layer_match_loss, CFLossFunc
from NCFM.SampleNet import SampleNet
from utils.experiment_tracker import TimingTracker, get_time
from data.dataset import TensorDataset
from utils.utils_text import define_model
from .evaluate import evaluate_syn_data
from torch.utils.data import DistributedSampler, DataLoader
from .decode import decode
from .subsample import subsample
from .condense_transfom import get_train_transform
from .compute_loss import compute_match_loss, compute_calib_loss
from data.dataloader import MultiEpochsDataLoader
import torch.optim as optim
from data.dataloader import AsyncLoader
from tqdm import tqdm
import random
from transformers import AutoTokenizer

class Condenser:
    def __init__(self, args, nclass_list, device="cuda"):
        self.timing_tracker = TimingTracker(args.logger)
        self.args = args
        self.logger = args.logger
        self.ipc = args.ipc  # 每个类别的样本数
        self.nclass_list = nclass_list
        self.device = device
        self.nclass = len(nclass_list)
        
        # 初始化 BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # 初始化合成数据
        # 对于文本数据，我们需要存储 tokenized 后的 input_ids, attention_mask 等
        self.data = {
            'input_ids': torch.randint(
                low=0, 
                high=self.tokenizer.vocab_size,
                size=(self.nclass * self.ipc, args.max_length),
                dtype=torch.long,
                requires_grad=False,
                device=self.device
            ),
            'attention_mask': torch.ones(
                (self.nclass * self.ipc, args.max_length),
                dtype=torch.long,
                requires_grad=False,
                device=self.device
            )
        }
        
        # 对于多标签分类，标签是 multi-hot 向量
        self.targets = torch.zeros(
            (self.nclass * self.ipc, self.nclass),
            dtype=torch.float,
            requires_grad=False,
            device=self.device
        )
        
        # 为每个类别设置标签
        for i, c in enumerate(self.nclass_list):
            start_idx = i * self.ipc
            end_idx = (i + 1) * self.ipc
            self.targets[start_idx:end_idx, c] = 1.0
            
        # 记录每个类别的样本索引
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data['input_ids'].shape[0]):
            # 对于多标签情况，找到第一个为1的标签作为主类别
            main_class = torch.where(self.targets[i] == 1)[0][0].item()
            self.cls_idx[main_class].append(i)
            
        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode="bilinear")
        if dist.get_rank() == 0:
            self.logger(f"Factor: {self.factor} ({self.decode_type})")

    def load_condensed_data(self, loader, init_type="noise", load_path=None):
        if init_type == "random":
            if dist.get_rank() == 0:
                self.logger("===================Random initialize condensed===================")
            for c in self.nclass_list:
                # 使用 class_sample 按类获取样本
                data, target = loader.class_sample(c, self.ipc)
                # 将数据分配到对应类别的位置
                start_idx = self.ipc * self.nclass_list.index(c)
                end_idx = self.ipc * (self.nclass_list.index(c) + 1)
                
                # 确保数据格式正确
                if isinstance(data, dict):  # 如果数据是字典格式（包含input_ids和attention_mask）
                    self.data['input_ids'][start_idx:end_idx] = data['input_ids']
                    self.data['attention_mask'][start_idx:end_idx] = data['attention_mask']
                else:  # 如果数据是张量格式
                    self.data['input_ids'][start_idx:end_idx] = data
                    # 如果没有attention_mask，则全部设为1
                    self.data['attention_mask'][start_idx:end_idx] = torch.ones_like(data)
                
                # 更新标签
                self.targets[start_idx:end_idx] = target
                
        elif init_type == "noise":
            if dist.get_rank() == 0:
                self.logger("===================Noise initialize condensed dataset===================")
            pass
            
        elif init_type == "load":
            if load_path is None:
                raise ValueError("Please provide the path of the initialization data")
            if dist.get_rank() == 0:
                self.logger("==================designed path initialize condense dataset ===================")
            data_dict = torch.load(load_path)
            self.data = {k: v.to(self.device) for k, v in data_dict['data'].items()}
            self.targets = data_dict['targets'].to(self.device)

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def class_sample(self, c, max_size=10000):
        """按类别采样数据"""
        # 对于多标签情况，检查targets中第c个位置是否为1
        target_mask = self.targets[:, c] == 1
        data = {
            'input_ids': self.data['input_ids'][target_mask],
            'attention_mask': self.data['attention_mask'][target_mask]
        }
        target = self.targets[target_mask]
        
        # 如果样本数量超过max_size，则随机选择max_size个样本
        if len(data['input_ids']) > max_size:
            indices = torch.randperm(len(data['input_ids']))[:max_size]
            data = {
                'input_ids': data['input_ids'][indices],
                'attention_mask': data['attention_mask'][indices]
            }
            target = target[indices]
        
        return data, target

    def get_syndataLoader(self, args, augment=True):
        """获取合成数据的DataLoader"""
        # 对于文本数据，我们不需要图像相关的transform
        # 但可能需要文本特定的数据增强
        train_transform = None  # 文本数据增强可以在这里添加
        
        data_dec = []
        target_dec = []
        for c in self.nclass_list:
            # 对于多标签情况，检查targets中第c个位置是否为1
            target_mask = self.targets[:, c] == 1
            data = {
                'input_ids': self.data['input_ids'][target_mask].detach(),
                'attention_mask': self.data['attention_mask'][target_mask].detach()
            }
            target = self.targets[target_mask].detach()
            
            data_dec.append(data)
            target_dec.append(target)

        # 合并所有类别的数据
        combined_data = {
            'input_ids': torch.cat([d['input_ids'] for d in data_dec]),
            'attention_mask': torch.cat([d['attention_mask'] for d in data_dec])
        }
        combined_targets = torch.cat(target_dec)
        
        if args.rank == 0:
            print("Decode condensed data: ", combined_data['input_ids'].shape)
        
        # 创建数据集
        train_dataset = TensorDataset(combined_data, combined_targets, train_transform)
        
        # 创建DataLoader
        nw = 0 if not augment else args.workers
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=args.world_size, 
            rank=args.rank, 
            shuffle=True
        )
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=True
        )
        return train_loader

    def condense(
        self,
        args,
        plotter,
        loader_real,
        aug,
        optim_img,
        model_init,
        model_interval,
        model_final,
        sampling_net=None,
        optim_sampling_net=None,
    ):
        """训练合成数据"""
        loader_real = AsyncLoader(
            loader_real, args.class_list, args.batch_real, args.device
        )
        loader_syn = AsyncLoader(self, args.class_list, 100000, args.device)
        
        # 设置损失函数
        args.cf_loss_func = CFLossFunc(
            alpha_for_loss=args.alpha_for_loss, 
            beta_for_loss=args.beta_for_loss
        )
        
        # 设置学习率调度器
        if args.sampling_net:
            scheduler_sampling_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_img, mode="min", factor=0.5, patience=500, verbose=False
            )
        else:
            scheduler_sampling_net = None
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_img, mode="min", factor=0.5, patience=500, verbose=False
        )
        
        # 保存初始状态
        gather_save_visualize(self, args)
        
        if args.local_rank == 0:
            pbar = tqdm(range(1, args.niter))
        
        for it in range(args.niter):
            # 更新特征提取器
            model_init, model_final, model_interval = update_feature_extractor(
                args, model_init, model_final, model_interval, a=0, b=1
            )
            
            # 确保数据在有效范围内
            self.data['input_ids'] = torch.clamp(
                self.data['input_ids'], 
                min=0, 
                max=self.tokenizer.vocab_size-1
            )
            self.data['attention_mask'] = torch.clamp(
                self.data['attention_mask'], 
                min=0, 
                max=1
            )
            
            # 计算匹配损失
            match_loss_total, match_grad_mean = compute_match_loss(
                args,
                loader_real=loader_real,
                sample_fn=loader_syn.class_sample,
                aug_fn=aug,
                inner_loss_fn=match_loss if args.depth <= 5 else mutil_layer_match_loss,
                optim_img=optim_img,
                class_list=self.args.class_list,
                timing_tracker=self.timing_tracker,
                model_interval=model_interval,
                data_grad=self.data['input_ids'].grad,
                optim_sampling_net=optim_sampling_net,
                sampling_net=sampling_net
            )
            
            # 计算校准损失（如果需要）
            if args.iter_calib > 0:
                calib_loss_total, calib_grad_mean = compute_calib_loss(
                    sample_fn=loader_syn.class_sample,
                    aug_fn=aug,
                    inter_loss_fn=cailb_loss,
                    optim_img=optim_img,
                    iter_calib=args.iter_calib,
                    class_list=self.args.class_list,
                    timing_tracker=self.timing_tracker,
                    model_final=model_final,
                    calib_weight=args.calib_weight,
                    data_grad=self.data['input_ids'].grad,
                )
            else:
                calib_loss_total, calib_grad_mean = 0, 0
            
            # 同步分布式指标
            calib_loss_total, match_loss_total, match_grad_mean, calib_grad_mean = (
                sync_distributed_metric([
                    calib_loss_total,
                    match_loss_total,
                    match_grad_mean,
                    calib_grad_mean,
                ])
            )
            
            # 计算总梯度均值
            total_grad_mean = (
                match_grad_mean + calib_grad_mean
                if args.iter_calib > 0
                else match_grad_mean
            )
            
            # 计算当前损失
            current_loss = (
                (match_loss_total + calib_loss_total) / args.nclass
                if args.iter_calib > 0
                else match_loss_total / args.nclass
            )
            
            # 更新损失曲线
            plotter.update_match_loss(match_loss_total / args.nclass)
            if args.iter_calib > 0:
                plotter.update_calib_loss(calib_loss_total / args.nclass)
            
            # 同步和更新进度
            if it % args.it_log == 0:
                dist.barrier()
            if args.local_rank == 0:
                pbar.set_description(f"[Niter {it+1}/{args.niter+1}]")
                pbar.update(1)
            
            # 更新学习率
            scheduler.step(current_loss)
            if scheduler_sampling_net is not None:
                scheduler_sampling_net.step(current_loss)

    def evaluate(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("======================Start Evaluation ======================")
        results = []
        all_best_acc = 0
        for i in range(args.val_repeat):
            if args.rank == 0:
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
            model = define_model(
                args.dataset,
                args.norm_type,
                args.net_type,
                args.nch,
                args.depth,
                args.width,
                args.nclass,
                args.logger,
                args.size,
            ).to(args.device)
            best_acc, acc = evaluate_syn_data(
                args, model, syndataloader, val_loader, logger=args.logger
            )
            if all_best_acc < best_acc:
                all_best_acc = best_acc
            results.append(best_acc)
            if args.rank == 0:
                args.logger(
                    f"Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                )
        mean_result = np.mean(results)
        std_result = np.std(results)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(f"Evaluation Stop:")
            args.logger(
                f"Mean Accuracy: {mean_result:.3f}", f"Std Deviation: {std_result:.3f}"
            )
            args.logger(f"All result: {[f'{x:.3f}' for x in results]}")
            args.logger("=" * 50)

    def continue_learning(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("Start Continue Learning ......... :D ")
        mean_result_list = []
        std_result_list = []
        results = []
        all_best_acc = 0
        step_classes = len(self.nclass_list) // args.steps

        all_classes = list(range(self.nclass))
        for current_step in range(1, args.step + 1):
            classes_seen = random.sample(all_classes, current_step * step_classes)
            def get_loader_step(classes_seen, val_loader):
                val_data, val_targets = [], []

                for data, target in val_loader:
                    mask = torch.tensor(
                        [t.item() in classes_seen for t in target], device=target.device
                    )
                    val_data.append(data[mask])
                    val_targets.append(target[mask])

                val_data = torch.cat(val_data)
                val_targets = torch.cat(val_targets)

                val_dataset_step = TensorDataset(val_data, val_targets)
                val_loader_step = DataLoader(val_dataset_step, batch_size=128, shuffle=False)
                return val_loader_step

            val_loader_step = get_loader_step(classes_seen, val_loader)
            syndataloader = get_loader_step(classes_seen, syndataloader)
            for i in range(args.val_repeat):
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
                model = define_model(
                    args.dataset,
                    args.norm_type,
                    args.net_type,
                    args.nch,
                    args.depth,
                    args.width,
                    args.nclass,
                    args.logger,
                    args.size,
                ).to(args.device)
                best_acc, acc = evaluate_syn_data(
                    args, model, syndataloader, val_loader_step, logger=args.logger
                )
                if all_best_acc < best_acc:
                    all_best_acc = best_acc
                results.append(best_acc)
                if args.rank == 0:
                    args.logger(
                        f"Step {current_step},Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                    )
            mean_result = np.mean(results)
            std_result = np.std(results)
            mean_result_list.append(mean_result)
            std_result_list.append(std_result)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(
                f"All result: {[f'Step {i} Acc: {x:.3f}' for i, x in enumerate(mean_result_list)]}"
            )
            args.logger("=" * 50)
