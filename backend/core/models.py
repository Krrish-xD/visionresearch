from sqlalchemy import Column, String, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .database import Base

class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id = Column(String, primary_key=True, index=True) # task_id
    parent_task_id = Column(String, index=True, nullable=True) # For video batches
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    status = Column(String, default="pending") # pending, complete, error
    error = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    total_time_ms = Column(Float, nullable=True)
    
    # Store the final aggregated structured output from AnalysisResult
    final_results = Column(JSON, nullable=True)
    
    # Persistent chat history: list of {role: "user"|"assistant", content: str}
    chat_history = Column(JSON, nullable=True, default=list)
