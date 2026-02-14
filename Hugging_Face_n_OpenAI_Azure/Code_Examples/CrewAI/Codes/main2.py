"""
Streamlit version of Trip Planner Crew
Run with: streamlit run main_streamlit.py
"""

import streamlit as st
from crew import TripCrew

# Page config
st.set_page_config(
    page_title="Trip Planner Crew",
    page_icon="--",
    layout="centered"
)

# Title and description
st.title("-- Trip Planner Crew")
st.write(
    "Plan your personalized trip using AI. "
    "Fill in your travel preferences below and get a detailed plan!"
)

# Input fields
origin = st.text_input("From where will you be traveling?")
cities = st.text_input("Which cities are you interested in visiting?")
date_range = st.text_input("Your preferred date range for travel")
interests = st.text_input("Your high-level interests and hobbies")

# Button to trigger the trip plan generation
if st.button("Generate Trip Plan"):
    if not (origin and cities and date_range and interests):
        st.warning("Please fill in all fields to generate a trip plan.")
    else:
        with st.spinner("Generating your trip plan..."):
            try:
                trip_crew = TripCrew(origin, cities, date_range, interests)
                result = trip_crew.run()
                st.success("Trip plan generated successfully!")
                st.subheader("Here is your Trip Plan:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred while generating the trip plan:\n{e}")

# Optional footer
st.markdown("---")
st.markdown("Created using CrewAI")
