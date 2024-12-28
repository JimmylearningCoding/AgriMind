import json


file1_path = "/root/AgriMind/dataset/congnitive_data.jsonl"
file2_path = "/root/AgriMind/dataset/new_xtuner_finetune_agri_data.jsonl"
output_path = "/root/AgriMind/dataset/final_merged_data.jsonl"

# 合并文件的函数
def merge_jsonl_files(file1, file2, output):
    with open(output, 'w', encoding="utf-8") as f1:
        # 处理第一个文件
        with open(file1, 'r', encoding="utf-8") as f2:
            for line in f2:
                json_obj = json.loads(line.strip()) # 解析json对象
                f1.write(json.dumps(json_obj,ensure_ascii=False) + '\n') # 写入新文件

        # 处理第二个文件
        with open(file2, 'r', encoding="utf-8") as f3:
            for line in f3:
                json_obj = json.loads(line.strip()) # 解析json对象
                f1.write(json.dumps(json_obj,ensure_ascii=False) + '\n') # 写入新文件

merge_jsonl_files(file1_path, file2_path, output_path)

print(f"两个文件已经合并到{output_path}")