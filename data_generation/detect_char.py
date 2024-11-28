import tiktoken
encoder = tiktoken.encoding_for_model("gpt-4")
sample_text = """这是一个测试。玉米病虫害防治技术
第一节 玉米害虫防治技术
一、形态特点与发生情况
二、防治措施
第二节 玉米病害防治技术
一、病害症状与发病规律
二、防治措施

第七章 马铃薯病虫害防治技术
第一节 马铃薯害虫防治技术
一、形态特点
二、发生情况与防治措施
第二节 马铃薯病害的防治技术
一、症状特点
二、发病规律与防治措施

第八章 棉花病虫害防治技术

第一节 棉花害虫防治技术
一、形态特点与发生情况
二、防治措
第二节 棉花病害防治技术 """
sample_text = sample_text.replace('\n',"") 
sample_text = sample_text.replace(" ","")
tokens = encoder.encode(sample_text)
print(f"Tokens: {tokens}")

decoded_text = encoder.decode(tokens)
print(f"Decoded Text: {decoded_text}")