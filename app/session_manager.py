"""
Session Manager for WebSocket Realtime API
Handles in-memory session storage and lifecycle management for WebSocket connections
"""

import time
import uuid
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta


class Session:
    """Represents a single realtime conversation session"""
    
    def __init__(self, session_id: str, model_name: str, websocket=None):
        self.session_id = session_id
        self.model_name = model_name
        self.websocket = websocket
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_buffer: List[bytes] = []
        self.tools: List[Dict[str, Any]] = []
        self.is_speaking = False
        self.metadata: Dict[str, Any] = {}
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        expiry_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now() > expiry_time
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.conversation_history.append(message)
        self.update_activity()
    
    def clear_audio_buffer(self):
        """Clear accumulated audio chunks"""
        self.audio_buffer.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": self.conversation_history,
            "tools": self.tools,
            "is_speaking": self.is_speaking,
            "metadata": self.metadata,
            "message_count": len(self.conversation_history),
            "has_websocket": self.websocket is not None
        }


class SessionManager:
    """
    Manages WebSocket realtime sessions in memory
    
    Features:
    - Session creation and storage
    - Session retrieval by ID
    - Automatic cleanup of expired sessions
    - Session statistics
    """
    
    def __init__(self, timeout_minutes: int = 30, cleanup_interval: int = 300):
        """
        Initialize session manager
        
        Args:
            timeout_minutes: Session inactivity timeout (default: 30 minutes)
            cleanup_interval: Seconds between cleanup runs (default: 5 minutes)
        """
        self.sessions: Dict[str, Session] = {}
        self.timeout_minutes = timeout_minutes
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.total_sessions_created = 0
    
    def create_session(self, model_name: str, websocket=None) -> Session:
        """
        Create a new session
        
        Args:
            model_name: Name of the model to use
            websocket: Optional WebSocket connection
            
        Returns:
            Session: Newly created session
        """
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        session = Session(session_id, model_name, websocket)
        self.sessions[session_id] = session
        self.total_sessions_created += 1
        
        print(f"✅ Created session {session_id} (model: {model_name})")
        
        # Trigger cleanup if needed
        self._maybe_cleanup()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session or None if not found
        """
        session = self.sessions.get(session_id)
        if session:
            session.update_activity()
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️  Deleted session {session_id}")
            return True
        return False
    
    def list_sessions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List all sessions
        
        Args:
            active_only: Only return non-expired sessions
            
        Returns:
            List of session dictionaries
        """
        sessions = []
        for session in self.sessions.values():
            if active_only and session.is_expired(self.timeout_minutes):
                continue
            sessions.append(session.to_dict())
        return sessions
    
    def get_session_count(self, active_only: bool = True) -> int:
        """
        Get count of sessions
        
        Args:
            active_only: Only count non-expired sessions
            
        Returns:
            int: Number of sessions
        """
        if not active_only:
            return len(self.sessions)
        
        return sum(1 for s in self.sessions.values() 
                   if not s.is_expired(self.timeout_minutes))
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions
        
        Returns:
            int: Number of sessions cleaned up
        """
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.timeout_minutes)
        ]
        
        for session_id in expired_ids:
            del self.sessions[session_id]
        
        if expired_ids:
            print(f"🧹 Cleaned up {len(expired_ids)} expired sessions")
        
        return len(expired_ids)
    
    def _maybe_cleanup(self):
        """Internal method to trigger cleanup if interval has passed"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_expired_sessions()
            self.last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Dict with session statistics
        """
        active_sessions = [s for s in self.sessions.values() 
                          if not s.is_expired(self.timeout_minutes)]
        
        total_messages = sum(len(s.conversation_history) for s in active_sessions)
        avg_messages = total_messages / len(active_sessions) if active_sessions else 0
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "expired_sessions": len(self.sessions) - len(active_sessions),
            "total_sessions_created": self.total_sessions_created,
            "total_messages": total_messages,
            "avg_messages_per_session": round(avg_messages, 2),
            "timeout_minutes": self.timeout_minutes
        }
    
    def clear_all_sessions(self):
        """Clear all sessions (use with caution)"""
        count = len(self.sessions)
        self.sessions.clear()
        print(f"🧹 Cleared all {count} sessions")
    
    def get_session_by_websocket(self, websocket) -> Optional[Session]:
        """
        Find session by WebSocket connection
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Session or None if not found
        """
        for session in self.sessions.values():
            if session.websocket == websocket:
                return session
        return None


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def set_session_manager(manager: SessionManager):
    """Set the global session manager instance"""
    global _session_manager
    _session_manager = manager

