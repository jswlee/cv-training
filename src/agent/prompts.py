"""
System prompts and templates for the Beach Conditions Agent
"""

SYSTEM_PROMPT = """You are a helpful beach conditions assistant for Kāʻanapali Beach in Maui, Hawaii. 

Your role is to analyze current beach conditions using real-time data and provide helpful recommendations to visitors. You have access to tools that can:

1. Capture current snapshots from the beach livestream
2. Detect and count people in the water and on the beach
3. Analyze weather conditions including cloud coverage and rain
4. Classify different areas of the beach using computer vision

When responding to users:
- Be friendly, informative, and helpful
- Focus on safety considerations (crowding, weather conditions)
- Provide specific, actionable recommendations
- Use the actual data from your analysis tools
- Consider factors like time of day, weather, and crowd levels
- Be honest about limitations of the analysis

Always use your tools to get current, real data before making recommendations. Never make assumptions or use outdated information.

Example response style:
"Based on the current snapshot from Kāʻanapali Beach: I can see 12 people in the water and 8 people on the beach. The sky shows about 25% cloud coverage with no signs of rain. The conditions look great for a beach visit! The water doesn't appear too crowded, making it a good time for swimming."
"""

ANALYSIS_PROMPT = """Analyze the current beach conditions and provide a comprehensive assessment.

Current data:
- People in water: {people_in_water}
- People on beach: {people_on_beach}
- Total people visible: {total_people}
- Cloud coverage: {cloud_coverage}%
- Weather condition: {weather_condition}
- Is raining: {is_raining}
- Visibility: {visibility}

Please provide:
1. A summary of current conditions
2. Assessment of crowding levels
3. Weather suitability for beach activities
4. Any safety considerations
5. Overall recommendation for visiting the beach now
"""

CROWDING_ASSESSMENT = {
    "very_low": "The beach appears very quiet with minimal crowds",
    "low": "The beach has light crowds - great for a peaceful visit",
    "moderate": "The beach has moderate crowds - still comfortable",
    "high": "The beach is quite busy - expect crowds",
    "very_high": "The beach is very crowded - consider visiting later"
}

WEATHER_RECOMMENDATIONS = {
    "clear": "Perfect weather for all beach activities",
    "partly_cloudy": "Good conditions with some cloud cover providing natural shade",
    "cloudy": "Overcast but still suitable for beach activities",
    "overcast": "Cloudy conditions - good for those who prefer less sun",
    "rainy": "Not ideal for beach activities due to rain"
}

SAFETY_CONSIDERATIONS = [
    "Check current water conditions and surf reports",
    "Apply sunscreen regularly, even on cloudy days",
    "Stay hydrated and seek shade during peak sun hours",
    "Be aware of your surroundings and follow local guidelines",
    "Consider crowd levels for social distancing if needed"
]

def get_crowding_level(total_people: int) -> str:
    """Determine crowding level based on people count"""
    if total_people <= 5:
        return "very_low"
    elif total_people <= 15:
        return "low"
    elif total_people <= 30:
        return "moderate"
    elif total_people <= 50:
        return "high"
    else:
        return "very_high"

def format_response_template(analysis_data: dict) -> str:
    """Format the analysis data into a response template"""
    
    people_data = analysis_data.get('people', {})
    weather_data = analysis_data.get('weather', {})
    
    total_people = people_data.get('total_people', 0)
    people_in_water = people_data.get('people_in_water', 0)
    people_on_beach = people_data.get('people_on_beach', 0)
    
    cloud_coverage = weather_data.get('cloud_coverage_percent', 0)
    weather_condition = weather_data.get('weather_condition', 'unknown')
    is_raining = weather_data.get('is_raining', False)
    visibility = weather_data.get('visibility', 'unknown')
    
    crowding_level = get_crowding_level(total_people)
    
    return ANALYSIS_PROMPT.format(
        people_in_water=people_in_water,
        people_on_beach=people_on_beach,
        total_people=total_people,
        cloud_coverage=cloud_coverage,
        weather_condition=weather_condition.replace('_', ' ').title(),
        is_raining="Yes" if is_raining else "No",
        visibility=visibility.title()
    )
