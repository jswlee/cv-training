"""
Snapshot capture module integrating with existing YouTube capture functionality
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import yaml

# Add the parent directory to path to import the existing capture script
sys.path.append(str(Path(__file__).parent.parent.parent))
from livestream_capture_youtube import YouTubeCapture

logger = logging.getLogger(__name__)

class BeachSnapshotCapture:
    """Beach-specific snapshot capture with configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize capture with configuration"""
        self.config = self._load_config(config_path)
        self.camera_config = self.config['camera']
        self.paths_config = self.config['paths']
        
        # Initialize YouTube capture
        self.youtube_capture = YouTubeCapture(
            url=self.camera_config['stream_url'],
            output_dir=self.paths_config['snapshots'],
            interval=self.camera_config['capture_interval']
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def capture_current_snapshot(self) -> str:
        """
        Capture a single snapshot from the beach livestream
        
        Returns:
            str: Path to the captured image file
        """
        try:
            logger.info("Capturing current beach snapshot...")
            
            # Use the existing capture functionality with retries
            snapshot_path = self.youtube_capture.capture_snapshot(
                max_retries=self.camera_config['max_retries']
            )
            
            if snapshot_path:
                logger.info(f"Successfully captured snapshot: {snapshot_path}")
                return snapshot_path
            else:
                raise Exception("Failed to capture snapshot after retries")
                
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            raise
    
    def get_latest_snapshot(self) -> str:
        """
        Get the path to the most recent snapshot
        
        Returns:
            str: Path to the latest snapshot file
        """
        try:
            snapshots_dir = Path(self.paths_config['snapshots'])
            if not snapshots_dir.exists():
                raise FileNotFoundError("Snapshots directory does not exist")
            
            # Find the most recent snapshot
            snapshot_files = list(snapshots_dir.glob("youtube_snapshot_*.jpg"))
            if not snapshot_files:
                raise FileNotFoundError("No snapshots found")
            
            latest_snapshot = max(snapshot_files, key=os.path.getctime)
            logger.info(f"Latest snapshot: {latest_snapshot}")
            return str(latest_snapshot)
            
        except Exception as e:
            logger.error(f"Error getting latest snapshot: {e}")
            raise
    
    def cleanup_old_snapshots(self, keep_count: int = 10):
        """
        Clean up old snapshots, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent snapshots to keep
        """
        try:
            snapshots_dir = Path(self.paths_config['snapshots'])
            snapshot_files = list(snapshots_dir.glob("youtube_snapshot_*.jpg"))
            
            if len(snapshot_files) <= keep_count:
                return
            
            # Sort by creation time and remove oldest
            snapshot_files.sort(key=os.path.getctime, reverse=True)
            files_to_remove = snapshot_files[keep_count:]
            
            for file_path in files_to_remove:
                file_path.unlink()
                logger.info(f"Removed old snapshot: {file_path}")
                
        except Exception as e:
            logger.error(f"Error cleaning up snapshots: {e}")

def main():
    """Test the capture functionality"""
    capture = BeachSnapshotCapture()
    
    try:
        # Capture a new snapshot
        snapshot_path = capture.capture_current_snapshot()
        print(f"Captured: {snapshot_path}")
        
        # Get latest snapshot
        latest = capture.get_latest_snapshot()
        print(f"Latest: {latest}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
