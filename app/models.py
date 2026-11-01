from datetime import datetime
from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base

class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    project_id: Mapped[str] = mapped_column(String(100), index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    identity_score: Mapped[float] = mapped_column(Float)
    defect_score: Mapped[float] = mapped_column(Float)
    adherence_score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.utcnow)
    
class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    project_id: Mapped[str] = mapped_column(String(100), index=True)
    feedback: Mapped[str] = mapped_column(String(30))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.utcnow)
    
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(50), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    feature_schema_version: Mapped[str] = mapped_column(String(20))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.utcnow)
    
class RolloutState(Base):
    __tablename__ = "rollout_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(50))
    stage: Mapped[str] = mapped_column(String(30))
    canary_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.utcnow)