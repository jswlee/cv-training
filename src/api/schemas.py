"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message to the beach conditions agent")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Agent's response to the user")
    analysis_data: Dict[str, Any] = Field(default_factory=dict, description="Analysis data used in response")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used in analysis")
    snapshot_path: Optional[str] = Field(None, description="Path to snapshot image if captured")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")
    error: Optional[str] = Field(None, description="Error message if any")

class SnapshotRequest(BaseModel):
    """Request model for snapshot capture"""
    force_new: bool = Field(False, description="Force capture of new snapshot even if recent one exists")

class SnapshotResponse(BaseModel):
    """Response model for snapshot capture"""
    snapshot_path: str = Field(..., description="Path to captured snapshot")
    timestamp: datetime = Field(..., description="Capture timestamp")
    image_size: Dict[str, int] = Field(..., description="Image dimensions")
    file_size_mb: float = Field(..., description="File size in MB")

class PeopleAnalysisResponse(BaseModel):
    """Response model for people analysis"""
    total_people: int = Field(..., description="Total number of people detected")
    people_in_water: int = Field(..., description="Number of people in water")
    people_on_beach: int = Field(..., description="Number of people on beach")
    people_other: int = Field(..., description="Number of people in other areas")
    confidence_stats: Dict[str, float] = Field(..., description="Detection confidence statistics")
    detections: Dict[str, List[Dict[str, Any]]] = Field(..., description="Detailed detection data")

class WeatherAnalysisResponse(BaseModel):
    """Response model for weather analysis"""
    cloud_coverage_percent: float = Field(..., description="Cloud coverage percentage")
    is_raining: bool = Field(..., description="Whether it's currently raining")
    rain_confidence: float = Field(..., description="Rain detection confidence")
    weather_condition: str = Field(..., description="Overall weather condition")
    visibility: str = Field(..., description="Visibility assessment")
    summary: str = Field(..., description="Human-readable weather summary")

class BeachConditionsResponse(BaseModel):
    """Complete beach conditions response"""
    people: PeopleAnalysisResponse
    weather: WeatherAnalysisResponse
    snapshot_info: SnapshotResponse
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

class ROIZone(BaseModel):
    """ROI zone definition"""
    name: str = Field(..., description="Zone name (water, beach, other)")
    polygon: List[List[int]] = Field(..., description="Polygon coordinates [[x,y], ...]")
    area: float = Field(..., description="Zone area in pixels")

class ROIResponse(BaseModel):
    """Response model for ROI detection"""
    image_width: int = Field(..., description="Image width")
    image_height: int = Field(..., description="Image height")
    water_polygons: List[List[List[int]]] = Field(..., description="Water region polygons")
    beach_polygons: List[List[List[int]]] = Field(..., description="Beach region polygons")
    zones_detected: int = Field(..., description="Total number of zones detected")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Status of individual services")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
