# acl2025-contrastive-perplexity-replication
acl2025-contrastive-perplexity的复现

源代码：https://github.com/SAP-samples/acl2025-contrastive-perplexity
复现只做了CP这个方法有效性的简单验证，如有不对的地方请指出

环境配置及部署请看上面连接中源代码的README.md

下面是我做复现时遇到的问题和解决方法：

1、数据集的预处理：
<img width="1207" height="402" alt="image" src="https://github.com/user-attachments/assets/5e22e405-1b76-4ec2-aa3c-58ee55b5086b" />

外面那个README.md中训练数据这部分没给出，完整训练数据制作流程在scripts文件夹那个README.md中

2、AUTO_GPTQ安装后显示未安装的问题

遇到了WARNING - CUDA kernels for auto_gptq are not installed, this will result in very slow inference speed.这个问题，导致推理速度极慢

具体解决步骤请看：https://blog.csdn.net/wi162yyxq/article/details/141422519

3、模型补丁问题，以train_mistral_hard_negatives.py为例

在这里为模型打补丁
<img width="812" height="266" alt="image" src="https://github.com/user-attachments/assets/90ddf315-0024-4923-bd6c-c5604dceea8b" />

还要将output['loss']改为output.loss
<img width="875" height="330" alt="image" src="https://github.com/user-attachments/assets/ad7a9851-520b-4a68-8beb-6f5ae5ecff39" />

不然会显示TypeError: forward() got an unexpected keyword argument 'loss_reduction'等问题

4、训练生成的模型adapter_model.safetensors是LoRA适配器，compute_perplexity.py不支持--lora_path，它只能加载完整模型，但训练后只有LoRA适配器

这个问题请使用我提供的merge_lora.py脚本来解决

python merge_lora.py


然后再使用

python scripts/compute_perplexity.py \
  --model_path autodl-tmp/mistral-detox-merged \
  --data_dir data/paraphrase/safeNLP_processed \
  --num_toxic_samples 4 \
  --num_neutral_samples 0

来计算困惑度





