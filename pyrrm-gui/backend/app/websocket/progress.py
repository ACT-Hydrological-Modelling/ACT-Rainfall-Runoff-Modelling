"""
WebSocket progress manager for real-time calibration updates.

Provides a connection manager for broadcasting calibration progress
to connected WebSocket clients.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import WebSocket


@dataclass
class CalibrationProgress:
    """Stores calibration progress state."""
    experiment_id: str
    iteration: int = 0
    total_iterations: Optional[int] = None
    best_objective: float = 0.0
    best_parameters: Dict[str, float] = field(default_factory=dict)
    current_objective: float = 0.0
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "best_objective": self.best_objective,
            "best_parameters": self.best_parameters,
            "current_objective": self.current_objective,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "progress_percent": (
                self.iteration / self.total_iterations * 100 
                if self.total_iterations and self.total_iterations > 0 
                else None
            )
        }


class ProgressManager:
    """
    Manages WebSocket connections and progress updates.
    
    Handles:
    - Tracking active WebSocket connections per experiment
    - Broadcasting progress updates to all connected clients
    - Storing latest progress state for new connections
    """
    
    def __init__(self):
        # Map of experiment_id -> list of connected WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # Map of experiment_id -> latest progress state
        self.progress_state: Dict[str, CalibrationProgress] = {}
    
    async def connect(self, websocket: WebSocket, experiment_id: str):
        """
        Accept a new WebSocket connection for an experiment.
        
        Args:
            websocket: The WebSocket connection
            experiment_id: ID of the experiment to track
        """
        await websocket.accept()
        
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = []
        
        self.active_connections[experiment_id].append(websocket)
        
        # Send current state if available
        if experiment_id in self.progress_state:
            await websocket.send_json({
                "type": "progress",
                "data": self.progress_state[experiment_id].to_dict()
            })
    
    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to remove
            experiment_id: ID of the experiment
        """
        if experiment_id in self.active_connections:
            if websocket in self.active_connections[experiment_id]:
                self.active_connections[experiment_id].remove(websocket)
            
            # Clean up empty lists
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]
    
    async def broadcast(self, experiment_id: str, message: Dict[str, Any]):
        """
        Broadcast a message to all connections for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            message: Message to send
        """
        if experiment_id in self.active_connections:
            disconnected = []
            
            for websocket in self.active_connections[experiment_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, experiment_id)
    
    async def update_progress(
        self,
        experiment_id: str,
        iteration: int,
        best_objective: float,
        best_parameters: Dict[str, float],
        current_objective: Optional[float] = None,
        total_iterations: Optional[int] = None
    ):
        """
        Update and broadcast calibration progress.
        
        Args:
            experiment_id: ID of the experiment
            iteration: Current iteration number
            best_objective: Best objective value so far
            best_parameters: Best parameter values so far
            current_objective: Current iteration's objective value
            total_iterations: Total expected iterations (if known)
        """
        # Update state
        if experiment_id not in self.progress_state:
            self.progress_state[experiment_id] = CalibrationProgress(
                experiment_id=experiment_id,
                started_at=datetime.utcnow()
            )
        
        progress = self.progress_state[experiment_id]
        progress.iteration = iteration
        progress.best_objective = best_objective
        progress.best_parameters = best_parameters
        progress.current_objective = current_objective or best_objective
        progress.total_iterations = total_iterations
        progress.updated_at = datetime.utcnow()
        
        # Broadcast
        await self.broadcast(experiment_id, {
            "type": "progress",
            "data": progress.to_dict()
        })
    
    async def send_completed(self, experiment_id: str, success: bool, message: str = ""):
        """
        Send completion notification.
        
        Args:
            experiment_id: ID of the experiment
            success: Whether calibration completed successfully
            message: Optional message (error message if failed)
        """
        await self.broadcast(experiment_id, {
            "type": "completed",
            "data": {
                "success": success,
                "message": message
            }
        })
        
        # Clean up state
        if experiment_id in self.progress_state:
            del self.progress_state[experiment_id]
    
    def get_progress(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress state for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Progress state dict or None
        """
        if experiment_id in self.progress_state:
            return self.progress_state[experiment_id].to_dict()
        return None
    
    def has_connections(self, experiment_id: str) -> bool:
        """Check if there are active connections for an experiment."""
        return (
            experiment_id in self.active_connections and
            len(self.active_connections[experiment_id]) > 0
        )


# Global progress manager instance
progress_manager = ProgressManager()
