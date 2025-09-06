"""
LangGraph agent nodes for beach conditions analysis
"""
import logging
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .prompts import SYSTEM_PROMPT, format_response_template

logger = logging.getLogger(__name__)

class BeachAgentState:
    """State management for the beach conditions agent"""
    
    def __init__(self):
        self.messages: List = []
        self.current_analysis: Dict[str, Any] = {}
        self.user_query: str = ""
        self.response: str = ""
        self.tools_used: List[str] = []
        self.error: str = ""

def extract_intent_node(state: BeachAgentState) -> BeachAgentState:
    """
    Extract user intent and determine what analysis is needed
    """
    try:
        logger.info("Extracting user intent")
        
        # Get the latest user message
        user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not user_messages:
            state.error = "No user message found"
            return state
        
        latest_message = user_messages[-1].content.lower()
        state.user_query = latest_message
        
        # Determine what analysis is needed based on keywords
        needs_current_data = any(keyword in latest_message for keyword in [
            "current", "now", "today", "right now", "at the moment", "currently"
        ])
        
        asks_about_people = any(keyword in latest_message for keyword in [
            "people", "crowd", "busy", "crowded", "swimmers", "visitors"
        ])
        
        asks_about_weather = any(keyword in latest_message for keyword in [
            "weather", "rain", "sunny", "cloudy", "cloud", "sky", "storm"
        ])
        
        asks_for_recommendation = any(keyword in latest_message for keyword in [
            "should", "recommend", "good time", "visit", "go", "beach"
        ])
        
        # Set flags for what analysis to perform
        state.needs_snapshot = needs_current_data or asks_for_recommendation
        state.needs_people_analysis = asks_about_people or asks_for_recommendation
        state.needs_weather_analysis = asks_about_weather or asks_for_recommendation
        
        logger.info(f"Intent analysis - Snapshot: {state.needs_snapshot}, "
                   f"People: {state.needs_people_analysis}, Weather: {state.needs_weather_analysis}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in extract_intent_node: {e}")
        state.error = str(e)
        return state

def capture_snapshot_node(state: BeachAgentState) -> BeachAgentState:
    """
    Capture current beach snapshot if needed
    """
    try:
        if not getattr(state, 'needs_snapshot', True):
            logger.info("Skipping snapshot capture - not needed")
            return state
        
        logger.info("Capturing beach snapshot")
        
        # This will be called by the tool
        state.tools_used.append("capture_snapshot")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in capture_snapshot_node: {e}")
        state.error = str(e)
        return state

def analyze_people_node(state: BeachAgentState) -> BeachAgentState:
    """
    Analyze people in the beach image
    """
    try:
        if not getattr(state, 'needs_people_analysis', True):
            logger.info("Skipping people analysis - not needed")
            return state
        
        logger.info("Analyzing people in beach image")
        
        # This will be called by the tool
        state.tools_used.append("analyze_people")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in analyze_people_node: {e}")
        state.error = str(e)
        return state

def analyze_weather_node(state: BeachAgentState) -> BeachAgentState:
    """
    Analyze weather conditions in the beach image
    """
    try:
        if not getattr(state, 'needs_weather_analysis', True):
            logger.info("Skipping weather analysis - not needed")
            return state
        
        logger.info("Analyzing weather conditions")
        
        # This will be called by the tool
        state.tools_used.append("analyze_weather")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in analyze_weather_node: {e}")
        state.error = str(e)
        return state

def generate_response_node(state: BeachAgentState) -> BeachAgentState:
    """
    Generate final response based on analysis results
    """
    try:
        logger.info("Generating response based on analysis")
        
        if state.error:
            state.response = f"I apologize, but I encountered an error while analyzing the beach conditions: {state.error}"
            return state
        
        if not state.current_analysis:
            state.response = "I wasn't able to gather current beach condition data. Please try again in a moment."
            return state
        
        # Format the analysis data for the LLM
        analysis_prompt = format_response_template(state.current_analysis)
        
        # Add system context
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {state.user_query}"),
            HumanMessage(content=analysis_prompt)
        ]
        
        # This will be handled by the LLM call in the graph
        state.messages.extend(messages)
        
        return state
        
    except Exception as e:
        logger.error(f"Error in generate_response_node: {e}")
        state.error = str(e)
        state.response = "I apologize, but I encountered an error while generating my response."
        return state

def should_capture_snapshot(state: BeachAgentState) -> str:
    """Conditional edge: determine if snapshot capture is needed"""
    return "capture" if getattr(state, 'needs_snapshot', True) else "analyze_people"

def should_analyze_people(state: BeachAgentState) -> str:
    """Conditional edge: determine if people analysis is needed"""
    return "people" if getattr(state, 'needs_people_analysis', True) else "analyze_weather"

def should_analyze_weather(state: BeachAgentState) -> str:
    """Conditional edge: determine if weather analysis is needed"""
    return "weather" if getattr(state, 'needs_weather_analysis', True) else "generate_response"

def format_final_response(state: BeachAgentState, llm_response: str) -> str:
    """
    Format the final response with additional context
    """
    try:
        # Add timestamp and data source info
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        footer = f"\n\n*Analysis based on live beach camera data captured at {timestamp} HST*"
        
        if state.tools_used:
            tools_info = f"\n*Tools used: {', '.join(state.tools_used)}*"
            footer += tools_info
        
        return llm_response + footer
        
    except Exception as e:
        logger.error(f"Error formatting final response: {e}")
        return llm_response
