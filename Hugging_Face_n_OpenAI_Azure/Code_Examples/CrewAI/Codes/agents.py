from crewai import Agent
from textwrap import dedent
#from langchain_openai.chat_models import AzureChatOpenAI
#the above is incompatible with CrewAI 0.157
#CrewAI now provides its own LLM wrapper
from crewai import LLM
from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools
#from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")

class TravelAgents:
    def __init__(self):
            """
            Initializes the TravelAgents class by setting up two OpenAI GPT models:
            - GPT-3.5 model (`gpt-3.5-turbo`) for generating text-based responses.
            - Both models are initialized with a temperature of 0.7 to control randomness 
              in responses.
            
            The models will be used by the different agents (travel planner, 
            city selection expert, and local tour guide) for generating responses based 
            on their specific goals and tasks.
            """
            #self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            #self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            self.OpenAIGPT41 = LLM(
            model="azure/gpt-4.1",
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0
            )
            
            '''
            self.OpenAIGPT4 = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                deployment_name="gpt-4",
                temperature=0,
            )
            '''
            '''
            self.OpenAIGPT41 = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                deployment_name="gpt-4.1",
                temperature=0,
            )
            '''

            """      
            #when using AzureOpenAI
            from openai import AzureOpenAI
            self.OpenAIGPT35 = AzureOpenAI(
            api_key=os.getenv("API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment="gpt-4.1",
            temperature=0.7
            )
            """

    def expert_travel_agent(self):  
        """
        Sets up the 'Expert Travel Agent' to help plan a 7-day trip.

        This agent can:
        - Build a detailed 7-day itinerary for the trip.
        - Suggest a budget for the entire journey.
        - Offer packing tips.
        - Share important safety advice.

        The agent uses the following tools:
        - `SearchTools.search_internet`: To look up useful travel info online.
        - `CalculatorTools.calculate`: To handle any necessary calculations (like costs).

        The agent works with the GPT-3.5 model (`self.OpenAIGPT35`) to create responses and complete tasks.

        Returns:
            Agent: The fully configured Expert Travel Agent.
        """
        return Agent(
                    role="Expert Travel Agent",
                    backstory= f"""I'm a seasoned expert in travel planning and logistics. With decades of experience, I'm here to help plan your perfect trip.""",
                    goal=f"""Plan a 7-day trip with day-by-day details, including a budget, packing tips, and safety recommendations.""",
                    #temp disable tools
                    # tools=[
                    #     SearchTools.search_internet,
                    #     CalculatorTools.calculate
                    #     ],
                    tools = [],
                    verbose=True,
                    llm=self.OpenAIGPT41,
                )


    def city_selection_expert(self):
        """
        Creates a 'City Selection Expert' agent to help choose the best travel destinations.

        This agent can:
        - Analyze travel data to recommend cities based on various factors.
        - Take into account weather, season, prices, and the preferences of the traveler.

        The agent uses the following tool:
        - `SearchTools.search_internet`: To gather city-related information online.

        The agent uses the GPT-3.5 model (`self.OpenAIGPT35`) to generate recommendations and respond to queries.

        Returns:
            Agent: The fully set-up City Selection Expert.
        """
        return Agent(
                    role="City Selection Expert",
                    backstory=dedent(f"""A pro at analyzing travel data to pick the best destinations based on a 
                                            variety of factors.""" ),
                    goal=dedent(f"""
                        Find the top cities to visit, considering weather, seasons, budget, and what the traveler enjoys."""
                    ),
                    #tools=[SearchTools.search_internet],
                    tools = [],
                    verbose=True,
                    llm=self.OpenAIGPT41,
                )
        
    def local_tour_guide(self):
        """
        Sets up a 'Local Tour Guide' agent to provide detailed information about a city.

        This agent can:
        - Share insights about the city's top attractions, customs, and local highlights.
        - Offer the best tips for experiencing the city like a local.

        The agent uses the following tool:
        - `SearchTools.search_internet`: To find local information and details about the city.

        The agent uses the GPT-3.5 model (`self.OpenAIGPT35`) to provide responses and recommendations.

        Returns:
            Agent: The fully set-up Local Tour Guide.
        """
        return Agent(
                role="Local Tour Guide",
                    backstory=dedent(f"""
                                     A knowledgeable local guide who knows everything about the city's attractions, 
                                     culture, and hidden gems."""),
                    goal=dedent(f"""   
                                Give the best insights and recommendations about the selected city for an unforgettable 
                                experience."""),
                    #tools=[SearchTools.search_internet],
                    tools = [],
                    verbose=True,
                    llm=self.OpenAIGPT41,
                )

'''
To fix tools usage
Using crew ai official tools
pip install crewai-tools
from crewai_tools import SerperDevTool
search_tool = SerperDevTool()


'''
