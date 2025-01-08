# 模型微调说明过程文档（以Qwen2.5_7b_instruct作为例子）

## 🤖 1. 环境准备
1.1 Ubuntu
- Python 3.10.16
- Pytorch 2.5.1
- GPU A100(24GB)

1.2 LLaMA-Factory框架
```shell
git clone https://github.com/hiyouga/LLaMA-Factory.git  
cd LLaMA-Factory   
pip install -e .  
```
1.3 训练数据集准备  
举个例子：
将训练数据放在 `LLaMA-Factory/data`里面  
然后修改数据配置文件： `LLaMA-Factory/data/dataset_info.json` 
```shell
"xxx": {
  "file_name": "xxx.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

## ⚙️ 2. 模型准备
```shell
pip install -U huggingface_hub
pip install huggingface-cli

huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir '这里自己指定地址'

```


## 🛠️ 3. 模型微调
3.1 微调方法一：可视化微调    
[方法一](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)   
3.2 微调方法二：配置文件微调    
[方法二](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html) 

## ⏬ 4. 模型推理与部署  
4.1
LLaMA-Factory推理可以使用 `llamafactory-cli chat inference_config.yaml`或者`llamafactory-cli webchat inference_config.yaml`进行推理与模型进行对话。  

```shell
model_name_or_path: /group_share/models/Qwen2.5-7B-Instruct
adapter_name_or_path: /group_share/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/train_2025-01-07-12-21-07
template: qwen
infer_backend: vllm  # choices: [huggingface, vllm]
trust_remote_code: true
finetuning_type: lora
```   
### 4.2 把原模型和LoRA适配器进行合并    
使用 `llamafactory-cli export merge_config.yaml` 指令来合并模型。   
```shell
model_name_or_path: /group_share/models/Qwen2.5-7B-Instruct
adapter_name_or_path: /group_share/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/train_2025-01-07-12-21-07
template: qwen
infer_backend: vllm  # choices: [huggingface, vllm]
trust_remote_code: true
finetuning_type: lora

### export
export_dir: /group_share/models/AgriMind_Qwen2.5_7b_instruct_v1
export_size: 2
export_device: cpu
export_legacy_format: false

```    
### 4.3 将模型以api形式进行部署（其他部署形式可以看官方文档）  
启动类似 `API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/llama3_lora_sft.yaml` 指令，这里的配置文件按自身配置进行修改。  

写一个api_call_exmple.py 来测试一下   
```shell
# api_call_example.py
from openai import OpenAI 
client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
messages = [{"role": "user", "content": "俺家养了几头牛，最近发现它们长得不快，饲料也吃了不少，但就是不见长肉。俺听说饲料配比很重要，可俺不懂咋配，能不能教教俺咋弄？"}]
result = client.chat.completions.create(messages=messages, model="/group_share/models/Qwen2.5-7B-Instruct")
print(result.choices[0].message)
```  

对应同样的输入   
使用通用大模型的效果：    
![before_finetuning](normal.png)  


微调之后的效果：更加亲切直白，跟农民伯伯的距离感没那么强  
![compare](after_finetuning.png)  