
import json


input_json_filename = '/root/AgriMind/dataset/finetune_agri_data.json'
output_json_filename = '/root/AgriMind/dataset/new_xtuner_finetune_agri_data.json'

# 读取原始JSON文件
with open(input_json_filename, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 转换数据格式
converted_data = [
    {"conversation": [
        {
            "instruction": "你是一个农业领域的小助手，帮助用户解答农作物种植、病虫害防治、农药化肥使用和农业法律问题。你的回答需要准确、专业，并考虑用户的实际需求。",
            "input": item["input"],
            "output": item["output"]
        }
    ]} for item in data
]

# 写入转换后的JSON文件
with open(output_json_filename, 'w', encoding='utf-8') as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=4)

print(f"转换完成，JSON文件已保存为 '{output_json_filename}'")

# 转换下json为jsonl文件
output_jsonl_filename = '/root/AgriMind/dataset/new_xtuner_finetune_agri_data.jsonl'
with open(output_jsonl_filename, 'w', encoding='utf-8') as f:
    for item in converted_data:
        json_str = json.dumps(item, ensure_ascii=False)
        f.write(json_str + '\n')
print(f"转换完成，JSON文件转换为了JSONl文件格式：'{output_jsonl_filename}")