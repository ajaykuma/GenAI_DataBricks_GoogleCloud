from crewai import Crew
from agents import TravelAgents
from tasks import TravelTasks

from dotenv import load_dotenv
load_dotenv()

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests
    
    def run(self):
        """
            Executes the trip planning process by:

            1. Initializing specialized travel agents.
            2. Assigning relevant tasks to each agent based on their expertise.
            3. Creating a Crew to coordinate and execute tasks.
            4. Running the Crew to generate a structured travel itinerary.

            Returns:
                str: A comprehensive travel plan, including city selection, 
                    itinerary details, and local insights.
        """
        
        # Initialize travel agents and tasks
        travel_agents = TravelAgents()
        travel_tasks = TravelTasks()

        # Create agent instances
        itinerary_planner = travel_agents.expert_travel_agent()
        destination_analyst = travel_agents.city_selection_expert()
        local_expert = travel_agents.local_tour_guide()

        # Assign tasks to agents
        itinerary_task = travel_tasks.plan_itinerary(
            itinerary_planner,
            self.cities,
            self.date_range,
            self.interests
        )

        destination_selection_task = travel_tasks.identify_city(
            destination_analyst,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        city_guide_task = travel_tasks.gather_city_info(
            local_expert,
            self.cities,
            self.date_range,
            self.interests
        )

        # Create and run the Crew
        # travel_crew = Crew(
        #     agents=[itinerary_planner, destination_analyst, local_expert],
        #     tasks=[itinerary_task, destination_selection_task, city_guide_task],
        #     verbose=True,
        # )

        #Changing execution order to ensure 
        #City selection → city guide → itinerary planning

        '''
        We should pass context between tasks
        as CrewAI supports context chaining.
        '''
        travel_crew = Crew(
        agents=[destination_analyst, local_expert, itinerary_planner],
        tasks=[destination_selection_task, city_guide_task, itinerary_task],
        verbose=True,
        )

        result = travel_crew.kickoff()
        return result.raw


