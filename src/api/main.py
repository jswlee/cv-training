"""
FastAPI main application for Beach Conditions Agent
"""
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    ChatRequest, ChatResponse, SnapshotRequest, SnapshotResponse,
    PeopleAnalysisResponse, WeatherAnalysisResponse, BeachConditionsResponse,
    ROIResponse, HealthResponse, ErrorResponse
)
from src.api.tools import create_tools
from src.agent.graph import create_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BeachConditionsAPI:
    """Main FastAPI application for beach conditions"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the API application"""
        self.config_path = config_path
        self.config = self._load_config()
        self.api_config = self.config.get('api', {})
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Beach Conditions Agent API",
            description="AI-powered beach conditions analysis for Kāʻanapali Beach",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize tools and agent
        self.tools = None
        self.agent = None
        self._initialize_components()
        
        # Setup routes
        self._setup_routes()
        
        # Mount static files for UI
        self._setup_static_files()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize tools and agent"""
        try:
            logger.info("Initializing beach analysis tools...")
            self.tools = create_tools(self.config_path)
            
            logger.info("Initializing LangGraph agent...")
            self.agent = create_agent(self.config, self.tools)
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=FileResponse)
        async def serve_ui():
            """Serve the main UI page"""
            ui_path = Path("src/ui/templates/index.html")
            if ui_path.exists():
                return FileResponse(ui_path)
            return {"message": "Beach Conditions Agent API", "docs": "/docs"}
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Main chat endpoint for beach conditions queries"""
            try:
                logger.info(f"Processing chat request: {request.message}")
                
                # Process message through agent
                result = self.agent.process_message(request.message)
                
                response = ChatResponse(
                    response=result['response'],
                    analysis_data=result['analysis_data'],
                    tools_used=result['tools_used'],
                    snapshot_path=result['snapshot_path'],
                    session_id=request.session_id,
                    error=result['error']
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/snapshot", response_model=SnapshotResponse)
        async def capture_snapshot(request: SnapshotRequest = SnapshotRequest()):
            """Capture a new beach snapshot"""
            try:
                logger.info("Capturing beach snapshot")
                
                result = self.tools['capture_snapshot']()
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                response = SnapshotResponse(
                    snapshot_path=result['snapshot_path'],
                    timestamp=datetime.now(),
                    image_size={
                        'width': result['image_width'],
                        'height': result['image_height']
                    },
                    file_size_mb=result['file_size_mb']
                )
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error capturing snapshot: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/snapshot/latest", response_model=SnapshotResponse)
        async def get_latest_snapshot():
            """Get the latest available snapshot"""
            try:
                logger.info("Getting latest snapshot")
                
                result = self.tools['get_latest_snapshot']()
                
                if 'error' in result:
                    raise HTTPException(status_code=404, detail=result['error'])
                
                response = SnapshotResponse(
                    snapshot_path=result['snapshot_path'],
                    timestamp=datetime.now(),
                    image_size={
                        'width': result['image_width'],
                        'height': result['image_height']
                    },
                    file_size_mb=result['file_size_mb']
                )
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting latest snapshot: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze/people")
        async def analyze_people(image_path: str):
            """Analyze people in a beach image"""
            try:
                logger.info(f"Analyzing people in: {image_path}")
                
                result = self.tools['analyze_people'](image_path)
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                return PeopleAnalysisResponse(**result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error analyzing people: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze/weather")
        async def analyze_weather(image_path: str):
            """Analyze weather conditions in a beach image"""
            try:
                logger.info(f"Analyzing weather in: {image_path}")
                
                result = self.tools['analyze_weather'](image_path)
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                return WeatherAnalysisResponse(**result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error analyzing weather: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/conditions", response_model=BeachConditionsResponse)
        async def get_beach_conditions(force_new: bool = False):
            """Get complete current beach conditions"""
            try:
                logger.info("Getting complete beach conditions")
                
                result = self.tools['analyze_complete_conditions'](force_new)
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                response = BeachConditionsResponse(
                    people=PeopleAnalysisResponse(**result['people']),
                    weather=WeatherAnalysisResponse(**result['weather']),
                    snapshot_info=SnapshotResponse(
                        snapshot_path=result['snapshot_info']['snapshot_path'],
                        timestamp=datetime.now(),
                        image_size={
                            'width': result['snapshot_info']['image_width'],
                            'height': result['snapshot_info']['image_height']
                        },
                        file_size_mb=result['snapshot_info']['file_size_mb']
                    )
                )
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting beach conditions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/roi/detect")
        async def detect_roi(image_path: str):
            """Detect ROI zones in a beach image"""
            try:
                logger.info(f"Detecting ROI zones in: {image_path}")
                
                result = self.tools['detect_roi_zones'](image_path)
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                return ROIResponse(**result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error detecting ROI: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            try:
                # Check component status
                services = {
                    "api": "healthy",
                    "agent": "healthy" if self.agent else "error",
                    "tools": "healthy" if self.tools else "error"
                }
                
                # Test snapshot capability
                try:
                    latest_result = self.tools['get_latest_snapshot']()
                    services["snapshot"] = "healthy" if 'error' not in latest_result else "warning"
                except:
                    services["snapshot"] = "error"
                
                overall_status = "healthy" if all(s in ["healthy", "warning"] for s in services.values()) else "error"
                
                return HealthResponse(
                    status=overall_status,
                    version="1.0.0",
                    services=services
                )
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                return HealthResponse(
                    status="error",
                    version="1.0.0",
                    services={"api": "error"}
                )
        
        @self.app.get("/image/{filename}")
        async def serve_image(filename: str):
            """Serve snapshot images"""
            try:
                image_path = Path("data/snapshots") / filename
                if not image_path.exists():
                    raise HTTPException(status_code=404, detail="Image not found")
                
                return FileResponse(image_path)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error serving image: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_static_files(self):
        """Setup static file serving for UI"""
        try:
            static_dir = Path("src/ui/static")
            if static_dir.exists():
                self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
                logger.info("Static files mounted")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
    
    def run(self):
        """Run the FastAPI server"""
        try:
            host = self.api_config.get('host', '0.0.0.0')
            port = self.api_config.get('port', 8000)
            reload = self.api_config.get('reload', False)
            
            logger.info(f"Starting Beach Conditions API on {host}:{port}")
            
            uvicorn.run(
                "src.api.main:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
            
        except Exception as e:
            logger.error(f"Error running server: {e}")
            raise

# Global app instance
app_instance = None

def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and return FastAPI app instance"""
    global app_instance
    if app_instance is None:
        app_instance = BeachConditionsAPI(config_path)
    return app_instance.app

# Create app instance
app = create_app()

def main():
    """Main entry point"""
    try:
        # Set up environment
        os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key-here")
        
        # Create and run app
        beach_api = BeachConditionsAPI()
        beach_api.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()
