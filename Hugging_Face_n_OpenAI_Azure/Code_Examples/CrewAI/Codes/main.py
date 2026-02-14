'''
Docstring for CrewAI.Codes.main
Versions:
crewai                                   0.157.0
langchain                                1.2.3
langchain-classic                        1.0.1
langchain-community                      0.4.1
langchain-core                           1.2.7
langchain-huggingface                    0.3.1
langchain-openai                         1.1.7
langchain-text-splitters                 1.1.0
langgraph                                1.0.7
langgraph-checkpoint                     2.1.1
langgraph-prebuilt                       1.0.7
langgraph-sdk                            0.3.4
langsmith                                0.4.8
openai                                   2.17.0
pyautogen                                0.7.6
pybase64                                 1.4.2
pycparser                                2.22
pydantic                                 2.11.7
pydantic_core                            2.33.2
pydantic-settings                        2.10.1
'''

from textwrap import dedent
from crew import TripCrew

if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')

    origin = input(dedent("From where will you be traveling from?\n"))
    cities = input(dedent("What are the cities options you are interested in visiting?\n"))
    date_range = input(dedent("What is the date range you are interested in traveling?\n"))
    interests = input(dedent("What are some of your high-level interests and hobbies?\n"))

    trip_crew = TripCrew(origin, cities, date_range, interests)
    result = trip_crew.run()

    print("\n\n########################")
    print("## Here is your Trip Plan")
    print("########################\n")
    print(result)
