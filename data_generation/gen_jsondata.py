from doctest import OutputChecker
import sys
import random 
import os
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="")

# 获取输入文件(用一个数组接)
def get_text_files(directory):
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    return [os.path.join(directory, f) for f in txt_files]

# 读取输入文件
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 编写prompt    
def return_random_prompt(content):
    system_prompt = "根据下面提供有关农业领域文本，请你仔细通读全文，你需要依据该文本：\n\n######\n{}######\n尽可能给出多样化的问题和对应的回答。我们将用于人工评估GLM-4模型对问答对数据的完成情况。要求:\n".format(content)
    system_prompt += "1. 生成问题有价值且遵守该文本信息，回答准确专业。\n"
    system_prompt += "2. 生成问答对不能重复。\n"
    system_prompt += "3. 问题多样化，同个问题可以换成不同表述方式，但意思保持不变。\n"
    system_prompt += "4. 为问题生成作为<input>，不应该只包含简单的占位符。<input>应提供实质性的内容问题，具有挑战性。字数不超过" + str(random.randint(80, 120)) + "字。\n"
    system_prompt += "5. <output>应该是对问题的适当且真实的回答，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复，但不能胡编乱造。<output>的内容应少于" + str(random.randint(512,1024)) + "字。\n\n"
    system_prompt += "请给出满足条件的10条JSON格式数据，并存储在一个列表中，便于整理使用，不要输出无法的字符，只要列表形式存储JSON数据\n"
    return system_prompt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python XXX.py <output_file>")
        exit(1)

    output_file = open(sys.argv[1], 'w', encoding='utf-8')

    MAX_EPOCHS = 1

    txt_files = get_text_files('/root/origin_data/uasge_guidelines')
    
    for file_path in txt_files:
        print(f"正在处理文件：{file_path}")
        content = read_file(file_path)

        for k in range(MAX_EPOCHS):
            response = client.chat.completions.create(
                model = "glm-4-Flash",
                messages=[{
                    "role":"user",
                    "content":return_random_prompt(content)
                }],
                top_p=0.7,
                temperature=0.9,
                stream=False,
                max_tokens=2500,
            )
            output_file.write(response.choices[0].message.content + '\n')

    output_file.close()



