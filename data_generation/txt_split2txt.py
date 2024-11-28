import os
import re
import tiktoken
import unicodedata

# 初始化分词器
try:
    encoder = tiktoken.encoding_for_model("gpt-4")
except Exception:
    encoder = tiktoken.get_encoding("cl100k_base")

# 清理文本函数
def clean_text(text):
    # 移除不可见字符
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # 合并多余换行
    text = re.sub(r'\n+', '\n', text)
    return text

# 分段函数
def split_text_line_by_line(file_path, max_token_length, encoder, output_dir):
    split_texts = []
    current_text = ""
    line_number = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            line = clean_text(line.strip())  # 清理每一行

            # 检查当前段是否超长
            if len(encoder.encode(current_text + line)) > max_token_length:
                # 保存当前段落到列表
                split_texts.append(current_text)
                current_text = line
            else:
                current_text += line + "\n"

    # 保存最后一段
    if current_text:
        split_texts.append(current_text)

    # 写入分段文件
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(split_texts):
        with open(os.path.join(output_dir, f"part_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(segment)

    print(f"分段完成，总段数: {len(split_texts)}")

# 示例：调用分段处理
input_file = "/root/origin_data/1. 作物病虫害防治技术.txt"
output_dir = "/root/origin_data/plant disease and pest"
max_token_length = 12000

split_text_line_by_line(input_file, max_token_length, encoder, output_dir)
