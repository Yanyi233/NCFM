# [CVPR2025 Rating: 5/5/5] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective 


<div align="center"> 
    <a href="https://doi.org/10.48550/arXiv.2502.20653"><img src="https://img.shields.io/badge/CVPR%202025-528dc7.svg" alt="Paper"></a>
    <a href="https://github.com/gszfwsb/NCFM/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/gszfwsb/NCFM?color=green"> </a>
</div>

Official PyTorch implementation of the paper ["Dataset Distillation with Neural Characteristic Function"](https://arxiv.org/abs/2502.20653) (NCFM) in CVPR 2025.


## :fire: News

- [2025/03/02] The code of our paper has been released.  
- [2025/02/27] Our NCFM paper has been accepted to CVPR 2025 (Rating: 555). Thanks!  


## :rocket: Pipeline

Here's an overview of the process behind our **Neural Characteristic Function Matching (NCFM)** method:

![Figure 1](./asset/figure1.png?raw=true)





## :mag: TODO

- [x] Distillation code
- [x] Evaluation code
- [ ] Config files
- [ ] Pretrained models
- [ ] Distilled datasets
- [ ] Continual learning code
- [ ] Project page




## 🛠️ Getting Started

To get started with NCFM, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/gszfwsb/NCFM.git
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```
3. Pretrain the models yourself, or download the **pretrained_models** from [Google Drive](https://drive.google.com/drive/folders/1HT_eUbTWOVXvBov5bM90b169jdy2puOh?usp=drive_link). 
```sh
cd pretrain
torchrun --nproc_per_node={n_gpus} --nnodes=1 pretrain_script.py --gpu={gpu_ids} --config_path=../config/{dataset}.yaml

```

4. Condense
```sh
cd condense 
torchrun --nproc_per_node={n_gpus} --nnodes=1 condense_script.py --gpu={gpu_ids} --ipc={ipc} --config_path=../config/{dataset}.yaml

```
5. Evaluation
```sh
cd evaluation 
torchrun --nproc_per_node={n_gpus} --nnodes=1 evaluation_script.py --gpu={gpu_ids} --ipc={ipc}  --config_path=../config/imagenet-1k.yaml --load_path= {distillation_dataset.pt}
```

### :blue_book: Example Usage

1. CIFAR-10

```sh
#ipc50
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=50 --config_path=../config/cifar10.yaml
```

2. CIFAR-100

```sh
#ipc10
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=10 --config_path=../config/cifar100.yaml
```




## 📂 File Structure 
<pre>
NCFM 
├── <span style="color:blue;">NCFM</span>
│   ├── NCFM.py
│   └── SampleNet.py
├── <span style="color:red;">README.md</span>
├── <span style="color:purple;">argsprocessor</span>
│   └── args.py
├── <span style="color:orange;">condense</span>
│   ├── condense_script.py
│   ├── imagenet-1k_preparation.py
├── <span style="color:teal;">condenser</span>
│   ├── Condenser.py
│   ├── compute_loss.py
│   ├── condense_transfom.py
│   ├── decode.py
│   ├── evaluate.py
│   └── subsample.py
├── <span style="color:darkcyan;">config</span>
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   ├── imagefruit.yaml (TBD)
│   ├── imagemeow.yaml (TBD)
│   ├── imagenet-1k.yaml (TBD)
│   ├── imagenette.yaml (TBD)
│   ├── imagesquawk.yaml (TBD)
│   ├── imagewoof.yaml (TBD)
│   ├── imageyellow.yaml (TBD)
│   └── tinyimagenet.yaml (TBD)
├── <span style="color:brown;">data</span>
│   ├──  __init__.py
│   ├── augment.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── dataset_statistics.py
│   ├── save_img.py
│   └── transform.py
├── <span style="color:darkgreen;">evaluation</span>
│   ├── evaluation_script.py
├── <span style="color:indigo;">imagenet_subset</span>
│   ├── class100.txt
│   ├── classimagefruit.txt
│   ├── classimagemeow.txt
│   ├── classimagenette.txt
│   ├── classimagesquawk.txt
│   ├── classimagewoof.txt
│   └── classimageyellow.txt
├── <span style="color:darkblue;">models</span>
│   ├── convnet.py
│   ├── densenet_cifar.py
│   ├── network.py
│   ├── resnet.py
│   └── resnet_ap.py
├── <span style="color:darkred;">pretrain</span>
│   ├── pretrain_script.py
├── <span style="color:darkorange;">requirements.txt</span>
└── <span style="color:darkslategray;">utils</span>
    ├── __init__.py
    ├── ddp.py
    ├── diffaug.py
    ├── experiment_tracker.py
    ├── init_script.py
    ├── mix_cut_up.py
    ├── train_val.py
    └── utils.py
</pre>

## :postbox: Contact
If you have any questions, please contact [Shaobo Wang](https://gszfwsb.github.io/)(`shaobowang1009@sjtu.edu.cn`).

## :pushpin: Citation
If you find NCM useful for your research and applications, please cite using this BibTeX:

```bibtex
@inproceedings{
  wang2025dataset,
  title={Dataset Distillation with Neural Characteristic Function: A Minmax Perspective},
  author={Shaobo Wang and Yicun Yang and Zhiyuan Liu and Chenghao Sun and Xuming Hu and Conghui He and Linfeng Zhang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
