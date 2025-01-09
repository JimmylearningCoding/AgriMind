import json
from openai import OpenAI
import re

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

def remove_unwanted_punctuation(text):
    # 定义需要保留的标点符号（可以根据需求调整）
    allowed_punctuation = "，。！？、：；“”‘’（）《》【】"
    # 去除不需要的标点符号
    cleaned_text = re.sub(fr'[^\w\s{allowed_punctuation}]', '', text)
    return cleaned_text

def extract_qa(dialogue):
    # 使用正则表达式匹配问题和回答
    question_match = re.search(r"问题：(.*?)\n\n回答：", dialogue, re.DOTALL)
    answer_match = re.search(r"回答：(.*)", dialogue, re.DOTALL)
    
    if question_match and answer_match:
        question = question_match.group(1).strip()  # 提取问题
        answer = answer_match.group(1).strip()      # 提取回答
        return question, answer
    else:
        return None, None

def generate_based_prompt(topic,subtopic,num_samples=120):
    
    prompt = f"""
            请你基于下述在农业生产领域中关于{topic}(具体问题：{subtopic})遇到的问题，制作问题与回答对。
            你需要模仿受教程低的农民去提出问题，确保你的问题：
            1. 模仿教育程度低的农民或农事工作人员，以他们的口吻去进行提问。
            2. 提问一般是简单直接直白的。
            
            然后，你需要给出一条专业并且容易理解的回复以帮助受教育水平较低的农民伯伯解决问题，确保你的回复：
            1. 用通俗易懂的语言进行描述 
            2. 尽量尝试用简单的描述替代专业术语 
            3. 分步骤列出具体操作 
            4. 加入视觉引导或实物参考 
            5. 提供背景信息，增强理解
            6. 提供效果预期
            确保你的回复流畅地将各部分相互连接：
            问题：[问题]
            回答：[回答]

            """
    dialogues = []

    for _ in range(num_samples):
        response = client.chat.completions.create(
            model="deepseek-chat",  # 自定义模型名称
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # 控制生成文本的长度
            temperature=1  # 控制生成文本的多样性
        )
        
        dialogues.append(response.choices[0].message.content.strip())
    
    return dialogues

# arr = generate_based_prompt("农田管理")

# cleaned_arr = [remove_unwanted_punctuation(dialogue) for dialogue in arr]

# 定义主类别和子类别
categories = {
    "农作物问题": ["病害诊断", "施肥建议", "种植技术"],
    "畜牧问题": ["疾病诊断", "饲料配比", "养殖环境管理"],
    "农具与农业机械使用问题": ["基础操作指南", "常见故障诊断", "日常保养与维护"]
}

questions = []
answers = []

alpaca_data = []

for  topic, subtopics in categories.items():
    for subtopic in subtopics:
        print(f"生成 {topic} - {subtopic} 的数据中...")
        dialogues = generate_based_prompt(topic,subtopic)

        clean_arr = [remove_unwanted_punctuation(dialogue) for dialogue in dialogues]

        for dialogue in clean_arr:
            question, answer = extract_qa(dialogue)
            if question and answer:
                alpaca_data.append({
                    "instruction": f"请帮助回答一个关于{topic}中{subtopic}的问题。",
                    "input":question,
                    "output":answer
                })

output_file = "alpaca_finetune_data.json"
with open(output_file,'w',encoding="utf-8") as f:
    json.dump(alpaca_data,f,ensure_ascii=False,indent=4)





# for i in range(len(questions)):
#     print(questions[i])
#     print("\n")
#     print(answers[i])

    
# questions = []
# answers = []

# for dialogue in cleaned_arr:
#     question, answer = extract_qa(dialogue)
#     if question and answer:
#         questions.append(question)
#         answers.append(answer)


# print(questions)
# print("\n")
# print(answers)