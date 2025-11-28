import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils_text import update_feature_extractor
from utils.ddp import gather_save_visualize, sync_distributed_metric
from NCFM.NCFM_text import match_loss, cailb_loss, mutil_layer_match_loss, CFLossFunc
from NCFM.SampleNet import SampleNet
from utils.experiment_tracker import TimingTracker, get_time
from data.dataset import TensorDataset
from utils.utils_text import define_model, define_language_model
from .evaluate_text import evaluate_syn_data
from torch.utils.data import DistributedSampler, DataLoader
from .compute_loss_text import compute_match_loss, compute_calib_loss
from data.dataloader import MultiEpochsDataLoader
from data.dataloader import AsyncLoader
from tqdm import tqdm
import random

class Condenser:
    def __init__(self, args, nclass_list, device="cuda"):
        self.timing_tracker = TimingTracker(args.logger)
        self.args = args
        self.logger = args.logger
        self.ipc = args.ipc  # 每个类别的样本数
        self.nclass_list = nclass_list
        self.device = device
        self.nclass = len(nclass_list)  # 当前进程处理的类别数（15）
        self.total_nclass = args.nclass  # 总类别数（90）
        
        # 初始化合成数据
        # 使用正态分布初始化embedding，范围与BERT的embedding层初始化范围一致
        self.data = torch.randn(
            (self.nclass * self.ipc, args.max_length, 768),  # 768是BERT的隐藏层维度
            dtype=torch.float32,
            requires_grad=True,
            device=self.device
        )
        
        # attention_mask作为类属性单独存储
        self.attention_mask = torch.ones(
            (self.nclass * self.ipc, args.max_length),
            dtype=torch.long,
            requires_grad=False,
            device=self.device
        )
        
        # 修改标签维度为总类别数（90）
        self.targets = torch.zeros(
            (self.nclass * self.ipc, self.total_nclass),
            dtype=torch.float,
            requires_grad=False,
            device=self.device
        )
        
        # if dist.get_rank() == 0:
        #     print('--------------------------------')
        #     print(f"dist.get_world_size(): {dist.get_world_size()}")
        #     print(f"self.nclass (local): {self.nclass}")
        #     print(f"self.total_nclass: {self.total_nclass}")
        #     print(f"self.nclass_list: {self.nclass_list}")
        #     print(f"self.ipc: {self.ipc}")
        #     print(f"self.targets.shape: {self.targets.shape}")
        #     print(f"self.data.shape: {self.data.shape}")
        #     print('--------------------------------')
            
        # 为每个类别设置标签
        for i, c in enumerate(self.nclass_list):
            start_idx = i * self.ipc
            end_idx = (i + 1) * self.ipc
            self.targets[start_idx:end_idx, c] = 1.0
            
        # 记录每个类别的样本索引
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            # 对于多标签情况，找到第一个为1的标签作为主类别
            main_class = torch.where(self.targets[i] == 1)[0][0].item()
            # 将主类别映射到本地类别索引
            local_class_idx = self.nclass_list.index(main_class)
            self.cls_idx[local_class_idx].append(i)

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
                
                # 更新数据
                self.data[start_idx:end_idx] = data['embeddings']
                self.attention_mask[start_idx:end_idx] = data['attention_mask']
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
            self.data = data_dict[0].to(self.device)
            self.targets = data_dict[1].to(self.device)

    def parameters(self):
        """返回需要优化的参数"""
        return [self.data]

    def class_sample(self, c, max_size=10000):
        """按类别采样数据"""
        # 对于多标签情况，检查targets中第c个位置是否为1
        target_mask = self.targets[:, c] == 1
        data = self.data[target_mask]
        target = self.targets[target_mask]
        
        # 如果样本数量超过max_size，则随机选择max_size个样本
        if len(data) > max_size:
            indices = torch.randperm(len(data))[:max_size]
            data = data[indices]
            target = target[indices]
        
        return data, target

    def get_syndataLoader(self, args, augment=True):
        """获取合成数据的DataLoader"""
        # 对于文本数据，我们不需要图像相关的transform
        # 但可能需要文本特定的数据增强
        train_transform = None  # 文本数据增强可以在这里添加
        
        # data_dec = []
        # target_dec = []
        # for c in self.nclass_list:
        #     # 对于多标签情况，检查targets中第c个位置是否为1
        #     target_mask = self.targets[:, c] == 1
        #     data = self.data[target_mask].detach()
        #     target = self.targets[target_mask].detach()
            
        #     data_dec.append(data)
        #     target_dec.append(target)

        # # 合并所有类别的数据
        combined_data = self.data.detach().cpu()
        combined_targets = self.targets.detach().cpu()

        
        if args.rank == 0:
            print("Decode condensed data: ", combined_data.shape)
        
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
        optim,
        model_init,
        model_interval,
        model_final,
        sampling_net=None,
        optim_sampling_net=None,
        val_loader=None,
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
                optim_sampling_net, mode="min", factor=0.5, patience=500, verbose=False
            )
        else:
            scheduler_sampling_net = None
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=500, verbose=False
        )
        
        # 保存初始状态
        gather_save_visualize(self, args)
        eval_interval = getattr(args, "eval_interval", 0)
        
        if args.local_rank == 0:
            pbar = tqdm(range(args.niter), desc="Condensing", unit="iter")
        else:
            pbar = range(args.niter)
        
        for it in pbar:
            self.timing_tracker.start_step()

            # 更新特征提取器
            model_init, model_final, model_interval = update_feature_extractor(
                args, model_init, model_final, model_interval, a=0, b=1
            )
            self.timing_tracker.record("update_feature_extractor")

            # 计算匹配损失
            match_loss_total, match_grad_mean = compute_match_loss(
                args,
                loader_real=loader_real,
                sample_fn=loader_syn.class_sample,
                aug_fn=aug,
                inner_loss_fn=match_loss if args.depth <= 5 else mutil_layer_match_loss,
                optim=optim,
                class_list=self.args.class_list,
                timing_tracker=self.timing_tracker,
                model_interval=model_interval,
                data_grad=self.data.grad,
                optim_sampling_net=optim_sampling_net,
                sampling_net=sampling_net
            )
            self.timing_tracker.record("compute_match_loss")
            
            # 计算校准损失（如果需要）
            if args.iter_calib > 0:
                calib_loss_total, calib_grad_mean = compute_calib_loss(
                    sample_fn=loader_syn.class_sample,
                    aug_fn=aug,
                    inter_loss_fn=cailb_loss,
                    optim=optim,
                    iter_calib=args.iter_calib,
                    class_list=self.args.class_list,
                    timing_tracker=self.timing_tracker,
                    model_final=model_final,
                    calib_weight=args.calib_weight,
                    data_grad=self.data.grad,
                )
                self.timing_tracker.record("compute_calib_loss")
            else:
                calib_loss_total, calib_grad_mean = 0, 0
            
            # 同步分布式指标
            metrics_to_sync = [calib_loss_total, match_loss_total, match_grad_mean, calib_grad_mean]
            synced_metrics = sync_distributed_metric(metrics_to_sync)
            calib_loss_total, match_loss_total, match_grad_mean, calib_grad_mean = synced_metrics[0], synced_metrics[1], synced_metrics[2], synced_metrics[3]

            self.timing_tracker.record("sync_metrics")
            
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
            if args.rank == 0:
                plotter.update_match_loss(match_loss_total / args.nclass)
                if args.iter_calib > 0:
                    plotter.update_calib_loss(calib_loss_total / args.nclass)
            
            self.timing_tracker.record("update_plotter")

            # 日志、保存和进度条更新
            if (it + 1) % args.it_log == 0:
                dist.barrier()
                if args.rank == 0:
                    timing_stats = self.timing_tracker.report(reset=True)
                    current_lr = optim.param_groups[0]["lr"]
                    plotter.plot_and_save_loss_curve()

                    log_message_parts = [
                        f"\n{get_time()} (Iter {it+1:3d}/{args.niter})",
                        f"LR: {current_lr:.6f}",
                    ]
                    if args.iter_calib > 0:
                        log_message_parts.append(f"inter-loss: {calib_loss_total / args.nclass / args.iter_calib:.4f}")
                    log_message_parts.append(f"inner-loss: {match_loss_total / args.nclass:.4f}")
                    log_message_parts.append(f"grad-norm: {total_grad_mean / args.nclass:.7f}")
                    log_message_parts.append(f"Timing Stats: {timing_stats}")
                    
                    self.logger(" ".join(log_message_parts))

            if args.local_rank == 0:
                # pbar.set_description(f"[Iter {it+1}/{args.niter}] LR:{optim.param_groups[0]['lr']:.5f} MatchL:{match_loss_total/args.nclass:.3f} CalibL:{calib_loss_total/args.nclass:.3f if args.iter_calib > 0 else 0:.3f}")
                calib_l_formatted = calib_loss_total / args.nclass if args.iter_calib > 0 else 0.0
                pbar.set_description(
                    f"[Iter {it+1}/{args.niter}] LR:{optim.param_groups[0]['lr']:.5f} "
                    f"MatchL:{match_loss_total/args.nclass:.3f} "
                    f"CalibL:{calib_l_formatted:.3f}"
                )

            # 按指定迭代或间隔保存数据
            # hasattr(args, 'it_save') and args.it_save and (it + 1) in args.it_save:
            #    self.logger(f"Iteration {it + 1} is in args.it_save. Saving data.")
            #    gather_save_visualize(self, args, iteration=it)
            # elif (it + 1) % args.save_interval == 0: # 这行导致 AttributeError
            #    self.logger(f"Reached save interval at iteration {it + 1}. Saving data.")
            #    gather_save_visualize(self, args, iteration=it)
            
            should_save = False
            save_reason = ""

            if hasattr(args, 'it_save') and args.it_save and (it + 1) in args.it_save:
                should_save = True
                save_reason = f"Iteration {it + 1} is in args.it_save."
            
            # 仅当 save_interval 在 args 中定义且大于0时才检查基于间隔的保存
            current_save_interval = getattr(args, 'save_interval', 0) # 默认为0，表示不基于此间隔保存
            if not should_save and current_save_interval > 0 and (it + 1) % current_save_interval == 0:
                should_save = True
                save_reason = f"Reached save interval at iteration {it + 1}."

            if should_save:
                self.logger(f"{save_reason} Saving data.")
                gather_save_visualize(self, args, iteration=it)
            
            self.timing_tracker.record("save_data_if_needed")

            # 更新学习率
            scheduler.step(current_loss)
            if scheduler_sampling_net is not None:
                scheduler_sampling_net.step(current_loss)
            self.timing_tracker.record("scheduler_step")

            if (
                eval_interval
                and eval_interval > 0
                and val_loader is not None
                and (it + 1) % eval_interval == 0
            ):
                dist.barrier()
                if args.rank == 0:
                    self.logger(f"Starting evaluation at iteration {it + 1}")
                syndataloader = self.get_syndataLoader(args, args.augment)
                self.evaluate(args, syndataloader, val_loader)
                dist.barrier()

        # 训练结束后最后保存一次
        self.logger(f"Condensation finished after {args.niter} iterations. Final Gather and Save Data!")
        gather_save_visualize(self, args, iteration=args.niter -1)
        
        if args.local_rank == 0:
            pbar.close()

    def evaluate(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("======================Start Evaluation ======================")
        
        results_best_mAP_per_run = []  # Stores the best mAP from each repeat
        results_f1_at_best_mAP_per_run = [] # Stores the F1 when best mAP was achieved in each repeat

        overall_best_mAP_across_repeats = 0.0 # Tracks the absolute best mAP across all repeats
        f1_for_overall_best_mAP = 0.0         # F1 for the overall_best_mAP_across_repeats

        for i in range(args.val_repeat):
            if args.rank == 0:
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
            
            model = define_language_model(
                args.model_path, 
                args.net_type, 
                args.nclass 
            ).to(args.device)

            # evaluate_syn_data returns: best_mAP_this_run, f1_at_best_mAP_this_run, last_mAP, last_f1
            best_mAP_this_run, f1_at_best_mAP_this_run, last_mAP, last_f1 = evaluate_syn_data(
                args, model, syndataloader, val_loader, logger=args.logger
            )
            
            results_best_mAP_per_run.append(best_mAP_this_run)
            results_f1_at_best_mAP_per_run.append(f1_at_best_mAP_this_run)
            
            if best_mAP_this_run > overall_best_mAP_across_repeats:
                overall_best_mAP_across_repeats = best_mAP_this_run
                f1_for_overall_best_mAP = f1_at_best_mAP_this_run
            
            if args.rank == 0:
                log_metric_name = "mAP" if args.is_multilabel else "Top-1 Acc"
                args.logger(
                    f"Repeat {i+1}/{args.val_repeat} => Best {log_metric_name} for this run: {best_mAP_this_run:.4f} (F1 at this point: {f1_at_best_mAP_this_run:.4f})"
                )
                args.logger(
                    f"Last Epoch {log_metric_name}: {last_mAP:.4f}, Last Epoch F1: {last_f1:.4f}"
                )
                args.logger(
                    f"Overall Best {log_metric_name} so far: {overall_best_mAP_across_repeats:.4f} (F1 for this {log_metric_name}: {f1_for_overall_best_mAP:.4f})"
                )
        
        mean_best_mAP = np.mean(results_best_mAP_per_run)
        std_best_mAP = np.std(results_best_mAP_per_run)
        # Optionally, calculate mean of F1s at best mAPs if meaningful
        # mean_f1_at_best_mAP = np.mean(results_f1_at_best_mAP_per_run) 
        
        if args.rank == 0:
            log_metric_name = "mAP" if args.is_multilabel else "Top-1 Acc"
            args.logger("=" * 50)
            args.logger(f"Evaluation Stop:")
            args.logger(
                f"Mean Best {log_metric_name} over repeats: {mean_best_mAP:.4f}", 
                f"Std Dev of Best {log_metric_name}: {std_best_mAP:.4f}"
            )
            # args.logger(
            #     f"Mean F1 at Best {log_metric_name} over repeats: {mean_f1_at_best_mAP:.4f}" # Optional
            # )
            args.logger(f"All Best {log_metric_name} results per repeat: {[f'{x:.4f}' for x in results_best_mAP_per_run]}")
            args.logger(f"Corresponding F1s at these Best {log_metric_name}s: {[f'{x:.4f}' for x in results_f1_at_best_mAP_per_run]}")
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
