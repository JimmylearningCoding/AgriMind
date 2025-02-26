from serpapi import GoogleSearch
from langchain.agents import initialize_agent, Tool
from models.chat_model import ChatModel
from config import Config
class NewsModel:
    def __init__(self):
        self.agriculture_news_tool = Tool(
            name="Agriculture News Query Tool",
            func=self.get_agriculture_news,
            description="用于查询近期的农业新闻热点"
        )
        self.llm = ChatModel().llm
        self.tools = [self.agriculture_news_tool]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )

    def get_agriculture_news(self, query: str = ""):
        params = {
            "engine": "baidu",
            "q": "近期热点农业新闻",
            "api_key": Config.SERPAPI_API_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        new_list = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                if "snippet" in result:
                    new_list.append(f"新闻摘要:{result['snippet']}\n链接：{result['link']}")
        return "\n\n".join(new_list) if new_list else "无法获取农业新闻信息，请稍后再试"

    def run(self, query):
        return self.agent.run(query)