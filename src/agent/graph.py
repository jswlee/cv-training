"""
LangGraph implementation for the Beach Conditions Agent
"""
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from .nodes import (
    BeachAgentState,
    extract_intent_node,
    capture_snapshot_node,
    analyze_people_node,
    analyze_weather_node,
    generate_response_node,
    should_capture_snapshot,
    should_analyze_people,
    should_analyze_weather,
    format_final_response
)

logger = logging.getLogger(__name__)

class BeachConditionsAgent:
    """LangGraph-based agent for beach conditions analysis"""
    
    def __init__(self, config: Dict[str, Any], tools: Dict[str, Any]):
        """
        Initialize the beach conditions agent
        
        Args:
            config: Configuration dictionary
            tools: Dictionary of available tools
        """
        self.config = config
        self.tools = tools
        self.agent_config = config.get('agent', {})
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.agent_config.get('model', 'gpt-3.5-turbo'),
            temperature=self.agent_config.get('temperature', 0.7),
            max_tokens=self.agent_config.get('max_tokens', 500)
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        try:
            # Create the graph
            workflow = StateGraph(BeachAgentState)
            
            # Add nodes
            workflow.add_node("extract_intent", self._extract_intent_wrapper)
            workflow.add_node("capture_snapshot", self._capture_snapshot_wrapper)
            workflow.add_node("analyze_people", self._analyze_people_wrapper)
            workflow.add_node("analyze_weather", self._analyze_weather_wrapper)
            workflow.add_node("generate_response", self._generate_response_wrapper)
            
            # Set entry point
            workflow.set_entry_point("extract_intent")
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "extract_intent",
                should_capture_snapshot,
                {
                    "capture": "capture_snapshot",
                    "analyze_people": "analyze_people"
                }
            )
            
            workflow.add_conditional_edges(
                "capture_snapshot",
                should_analyze_people,
                {
                    "people": "analyze_people",
                    "analyze_weather": "analyze_weather"
                }
            )
            
            workflow.add_conditional_edges(
                "analyze_people",
                should_analyze_weather,
                {
                    "weather": "analyze_weather",
                    "generate_response": "generate_response"
                }
            )
            
            workflow.add_edge("analyze_weather", "generate_response")
            workflow.add_edge("generate_response", END)
            
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    def _extract_intent_wrapper(self, state: BeachAgentState) -> BeachAgentState:
        """Wrapper for extract_intent_node"""
        return extract_intent_node(state)
    
    def _capture_snapshot_wrapper(self, state: BeachAgentState) -> BeachAgentState:
        """Wrapper for capture_snapshot_node with tool integration"""
        try:
            state = capture_snapshot_node(state)
            
            if "capture_snapshot" in state.tools_used:
                # Call the capture tool
                capture_tool = self.tools.get('capture_snapshot')
                if capture_tool:
                    result = capture_tool()
                    if 'error' not in result:
                        state.snapshot_path = result.get('snapshot_path')
                        logger.info(f"Snapshot captured: {state.snapshot_path}")
                    else:
                        state.error = result['error']
                        logger.error(f"Snapshot capture failed: {state.error}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in capture snapshot wrapper: {e}")
            state.error = str(e)
            return state
    
    def _analyze_people_wrapper(self, state: BeachAgentState) -> BeachAgentState:
        """Wrapper for analyze_people_node with tool integration"""
        try:
            state = analyze_people_node(state)
            
            if "analyze_people" in state.tools_used:
                # Call the people analysis tool
                people_tool = self.tools.get('analyze_people')
                if people_tool:
                    snapshot_path = getattr(state, 'snapshot_path', None)
                    if snapshot_path:
                        result = people_tool(snapshot_path)
                        if 'error' not in result:
                            if 'people' not in state.current_analysis:
                                state.current_analysis['people'] = {}
                            state.current_analysis['people'].update(result)
                            logger.info(f"People analysis completed: {result}")
                        else:
                            state.error = result['error']
                            logger.error(f"People analysis failed: {state.error}")
                    else:
                        state.error = "No snapshot available for people analysis"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in analyze people wrapper: {e}")
            state.error = str(e)
            return state
    
    def _analyze_weather_wrapper(self, state: BeachAgentState) -> BeachAgentState:
        """Wrapper for analyze_weather_node with tool integration"""
        try:
            state = analyze_weather_node(state)
            
            if "analyze_weather" in state.tools_used:
                # Call the weather analysis tool
                weather_tool = self.tools.get('analyze_weather')
                if weather_tool:
                    snapshot_path = getattr(state, 'snapshot_path', None)
                    if snapshot_path:
                        result = weather_tool(snapshot_path)
                        if 'error' not in result:
                            if 'weather' not in state.current_analysis:
                                state.current_analysis['weather'] = {}
                            state.current_analysis['weather'].update(result)
                            logger.info(f"Weather analysis completed: {result}")
                        else:
                            state.error = result['error']
                            logger.error(f"Weather analysis failed: {state.error}")
                    else:
                        state.error = "No snapshot available for weather analysis"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in analyze weather wrapper: {e}")
            state.error = str(e)
            return state
    
    def _generate_response_wrapper(self, state: BeachAgentState) -> BeachAgentState:
        """Wrapper for generate_response_node with LLM integration"""
        try:
            state = generate_response_node(state)
            
            if state.messages and not state.error:
                # Call LLM to generate response
                try:
                    response = self.llm.invoke(state.messages)
                    llm_response = response.content
                    
                    # Format final response
                    state.response = format_final_response(state, llm_response)
                    
                    # Add AI message to conversation
                    state.messages.append(AIMessage(content=state.response))
                    
                    logger.info("Response generated successfully")
                    
                except Exception as llm_error:
                    logger.error(f"LLM error: {llm_error}")
                    state.error = f"Error generating response: {llm_error}"
                    state.response = "I apologize, but I'm having trouble generating a response right now. Please try again."
            
            return state
            
        except Exception as e:
            logger.error(f"Error in generate response wrapper: {e}")
            state.error = str(e)
            state.response = "I apologize, but I encountered an error while generating my response."
            return state
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return the agent's response
        
        Args:
            user_message: User's input message
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing user message: {user_message}")
            
            # Initialize state
            initial_state = BeachAgentState()
            initial_state.messages = [HumanMessage(content=user_message)]
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Prepare response
            response_data = {
                'response': final_state.response,
                'analysis_data': final_state.current_analysis,
                'tools_used': final_state.tools_used,
                'error': final_state.error if hasattr(final_state, 'error') else None,
                'snapshot_path': getattr(final_state, 'snapshot_path', None)
            }
            
            logger.info("Message processing completed")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your request. Please try again.",
                'analysis_data': {},
                'tools_used': [],
                'error': str(e),
                'snapshot_path': None
            }
    
    def get_conversation_history(self, state: BeachAgentState) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        try:
            history = []
            for message in state.messages:
                if isinstance(message, HumanMessage):
                    history.append({
                        'role': 'user',
                        'content': message.content
                    })
                elif isinstance(message, AIMessage):
                    history.append({
                        'role': 'assistant',
                        'content': message.content
                    })
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

def create_agent(config: Dict[str, Any], tools: Dict[str, Any]) -> BeachConditionsAgent:
    """
    Factory function to create a beach conditions agent
    
    Args:
        config: Configuration dictionary
        tools: Available tools dictionary
        
    Returns:
        Configured BeachConditionsAgent instance
    """
    try:
        agent = BeachConditionsAgent(config, tools)
        logger.info("Beach conditions agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise
