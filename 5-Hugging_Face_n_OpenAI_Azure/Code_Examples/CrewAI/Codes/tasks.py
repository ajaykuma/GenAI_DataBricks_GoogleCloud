from crewai import Task
from textwrap import dedent

'''
CrewAI now uses expected_output to:

• improve agent reasoning
• validate outputs
• improve multi-agent coordination
• enable structured workflows

'''
class TravelTasks:

    def plan_itinerary(self, agent, city, travel_dates, interests):

        return Task(
        description=dedent(f"""
        **Task**: Develop a 7-Day Travel Itinerary

        **City**: {city}
        **Travel Dates**: {travel_dates}
        **Traveler Interests**: {interests}

        Create a detailed 7-day itinerary including:

        - Daily schedule
        - Specific attractions
        - Actual hotels
        - Actual restaurants
        - Activities matching traveler interests
        - Budget breakdown
        - Packing suggestions
        - Safety recommendations
        """),

        expected_output=dedent("""
        A complete 7-day travel itinerary with daily breakdown, hotels,
        restaurants, attractions, budget, packing, and safety tips.
        """),

        agent=agent
    )

    def identify_city(self, agent, origin, cities, interests, travel_dates):

        return Task(
            description=dedent(f"""
            **Task**: Identify the Best Travel Destination

            Analyze the following:

            Origin: {origin}
            Candidate Cities: {cities}
            Interests: {interests}
            Travel Dates: {travel_dates}

            Determine the best city based on:

            - Weather
            - Events
            - Travel costs
            - Attractions
            - Interest alignment
            """),

            expected_output=dedent("""
            Selected city and detailed reasoning including weather,
            events, cost, and attractions.
            """),

            agent=agent
            )
    
    def gather_city_info(self, agent, city, travel_dates, interests):

        return Task(
        description=dedent(f"""
        **Task**: Create a detailed city guide

        City: {city}
        Travel Dates: {travel_dates}
        Interests: {interests}

        Provide:

        - Top attractions
        - Cultural insights
        - Events
        - Travel tips
        - Estimated costs
        """),

        expected_output=dedent("""
        A complete city guide with attractions, culture, events,
        travel tips, and cost estimates.
        """),
        
        agent=agent
    )


