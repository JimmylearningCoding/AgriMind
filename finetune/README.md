# README

# 模型微调过程说明文档（以1.8B为例子）

## 1. 环境与数据准备

### 1.1 环境准备
确保环境可以用到Xtuner工具

## 2. 模型训练

### 2.1 数据准备
```shell
# 利用合并脚本将自我认知数据集合和专业领域数据进行合并
python /root/AgriMind/data_generation/combine_json1.py 
```
得到/dataset/final_merged_data.json1数据用于模型微调

### 2.2 基座模型准备

首先将开发提供的1.8B放到项目微调文件夹下。

```shell
cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/AgriMind/finetune/model/Internlm2-1_8B/
```

### 2.3 下载配置文件

```shell
# 使用XTuner工具，它提供多个模型的配置文件方便我们微调
# 想看看支持什么模型，可以输入
xtuner list-cfg
# 找到Intenrlm2-1.8b模型里支持的配置文件
xtuner list-cfg -p internlm2_1_8b

# 创建一个专门存放config的文件夹(项目下finetune文件路径下)
mkdir config
mkdir internlm2_1_8b 
# 使用Xtuner命令 copy-cfg 将 config文件复制到指定的位置
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/AgriMind/finetune/config/internlm2_1_8b/

```



### 2.4 修改配置参数

修改config下的`internlm2_1_8b_qlora_alpaca_e3_copy.py` 配置参数

```python
# Model

pretrained_model_name_or_path = '/root/AgriMind/finetune/model/Internlm2-1_8B'
use_varlen_attn = False

# Data

alpaca_en_path = '/root/AgriMind/dataset/final_merged_data.jsonl'
prompt_template = PROMPT_TEMPLATE.default
max_length = 2048
max_length = 1024
pack_to_max_length = True

# parallel
@@ -57,7 +57,7 @@
evaluation_freq = 500
SYSTEM = SYSTEM_TEMPLATE.alpaca
evaluation_inputs = [
    
    '请介绍一下你自己', 'please introduce yourself'
]

#######################################################################
@@ -99,10 +99,10 @@
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=alpaca_en_path),
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=alpaca_map_fn,
    dataset_map_fn=None,
```

### 2.5 模型训练

```shell
# 使用xtuner train命令并且使用deepseed来加速训练
xtuner train /root/AgriMind/finetune/config/internlm2_1_8b/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/AgriMind/finetune/train/internlm2_1_8b --deepspeed deepspeed_zero2

```

### 2.6 转换模型格式至HuggingFace

1. 创建HuggingFace格式存储目录

```shell
mkdir -p /root/AgriMind/finetune/huggingface/1_8b
```

2. 模型转换：使用提供的配置和权重文件进行模型转换

```shell
# 使用xtuner convert pth_to_hf命令
xtuner convert pth_to_hf /root/AgriMind/finetune/config/internlm2_1_8b/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/AgriMind/finetune/train/internlm2_1_8b/iter_1525.pth /root/AgriMind/finetune/huggingface/1_8b --fp32
```

3. 合并模型：合并模型并解决依赖关系

```shell
mkdir -p /root/AgriMind/finetune/final_model/AgriMind2_1_8b
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /root/AgriMind/finetune/model/internlm2_1_8b /root/AgriMind/finetune/huggingface/1_8b /root/AgriMind/finetune/final_model/AgriMind2_1
```



4. 测试模型：启动对话来测试下微调后模型效果

```shell
xtuner chat /root/AgriMind/finetune/final_model/AgriMind2_1_8b --prompt-template internlm2_chat
```



