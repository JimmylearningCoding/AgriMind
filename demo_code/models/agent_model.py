from langchain.agents import initialize_agent, Tool
from models.weather_model import WeatherModel
from models.chat_model import ChatModel

class AgentModel:
    def __init__(self):
        self.weather_tool = Tool(
            name="Weather Query Tool",
            func=WeatherModel().get_weather_by_coordinates,
            description="用于查询某个地点的天气状况"
        )
        self.llm = ChatModel().llm
        self.tools = [self.weather_tool]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )

    def run(self, query):
        return self.agent.run(query)