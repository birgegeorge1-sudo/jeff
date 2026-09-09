from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import create_engine, String, Text, DateTime, Integer, ForeignKey, UniqueConstraint, Float, Boolean, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship
from .config import MYSQL_URL

if not MYSQL_URL:
    raise RuntimeError('MYSQL_URL is required')

engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=1800, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

class Base(DeclarativeBase): pass

class Contact(Base):
    __tablename__='contacts'
    id: Mapped[int]=mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str]=mapped_column(String(160), unique=True, index=True)
    display_name: Mapped[str]=mapped_column(String(160))
    notes: Mapped[str]=mapped_column(Text, default='')
    relationship_stage: Mapped[str]=mapped_column(String(40), default='new')
    vibe_score: Mapped[float]=mapped_column(Float, default=50.0)
    engagement_score: Mapped[float]=mapped_column(Float, default=50.0)
    date_readiness: Mapped[float]=mapped_column(Float, default=0.0)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    messages=relationship('Message', back_populates='contact', cascade='all, delete-orphan')
    memories=relationship('Memory', back_populates='contact', cascade='all, delete-orphan')
    events=relationship('ConversationEvent', back_populates='contact', cascade='all, delete-orphan')

class Message(Base):
    __tablename__='messages'
    id: Mapped[int]=mapped_column(Integer, primary_key=True)
    contact_id: Mapped[int]=mapped_column(ForeignKey('contacts.id'), index=True)
    sender: Mapped[str]=mapped_column(String(30))
    content: Mapped[str]=mapped_column(Text)
    platform: Mapped[str]=mapped_column(String(50), default='manual')
    external_message_id: Mapped[Optional[str]]=mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    contact=relationship('Contact', back_populates='messages')

class Memory(Base):
    __tablename__='memories'
    id: Mapped[int]=mapped_column(Integer, primary_key=True)
    contact_id: Mapped[int]=mapped_column(ForeignKey('contacts.id'), index=True)
    key: Mapped[str]=mapped_column(String(140))
    value: Mapped[str]=mapped_column(Text)
    importance: Mapped[int]=mapped_column(Integer, default=3)
    source_message_id: Mapped[Optional[int]]=mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    contact=relationship('Contact', back_populates='memories')
    __table_args__=(UniqueConstraint('contact_id','key',name='uq_contact_memory_key'),)

class ConversationEvent(Base):
    __tablename__='conversation_events'
    id: Mapped[int]=mapped_column(Integer, primary_key=True)
    contact_id: Mapped[int]=mapped_column(ForeignKey('contacts.id'), index=True)
    event_type: Mapped[str]=mapped_column(String(60))
    detail: Mapped[str]=mapped_column(Text, default='')
    score: Mapped[Optional[float]]=mapped_column(Float, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    contact=relationship('Contact', back_populates='events')

class RuntimeState(Base):
    __tablename__='runtime_state'
    id: Mapped[int]=mapped_column(Integer, primary_key=True)
    active_contact_id: Mapped[Optional[int]]=mapped_column(Integer, nullable=True)

def init_db(): Base.metadata.create_all(engine)

def normalize_name(name:str)->str: return ' '.join(name.strip().split()).casefold()

def get_or_create_contact(db,name:str)->Contact:
    canonical=normalize_name(name)
    c=db.scalar(select(Contact).where(Contact.canonical_name==canonical))
    if c:return c
    c=Contact(canonical_name=canonical,display_name=name.strip())
    db.add(c);db.commit();db.refresh(c);return c

def set_active_contact(db,contact:Contact):
    s=db.get(RuntimeState,1)
    if not s:
        s=RuntimeState(id=1,active_contact_id=contact.id);db.add(s)
    else:s.active_contact_id=contact.id
    db.commit()

def get_active_contact(db):
    s=db.get(RuntimeState,1)
    return db.get(Contact,s.active_contact_id) if s and s.active_contact_id else None

def save_message(db,contact_id:int,sender:str,content:str,platform='manual',external_message_id=None)->Message:
    m=Message(contact_id=contact_id,sender=sender,content=content,platform=platform,external_message_id=external_message_id)
    db.add(m);db.commit();db.refresh(m);return m

def recent_messages(db,contact_id:int,limit:int=40):
    rows=list(db.scalars(select(Message).where(Message.contact_id==contact_id).order_by(Message.created_at.desc()).limit(limit)))
    rows.reverse();return rows

def upsert_memory(db,contact_id,key,value,importance=3,source_message_id=None):
    row=db.scalar(select(Memory).where(Memory.contact_id==contact_id,Memory.key==key))
    if row:
        row.value=value;row.importance=importance;row.source_message_id=source_message_id
    else:db.add(Memory(contact_id=contact_id,key=key,value=value,importance=importance,source_message_id=source_message_id))
    db.commit()

def get_memories(db,contact_id,limit=50):
    return list(db.scalars(select(Memory).where(Memory.contact_id==contact_id).order_by(Memory.importance.desc(),Memory.updated_at.desc()).limit(limit)))

def add_event(db,contact_id,event_type,detail='',score=None):
    db.add(ConversationEvent(contact_id=contact_id,event_type=event_type,detail=detail,score=score));db.commit()

def find_contact(db, name: str):
    return db.scalar(select(Contact).where(Contact.canonical_name == normalize_name(name)))

def delete_contact(db, contact_id: int) -> bool:
    contact = db.get(Contact, contact_id)
    if not contact:
        return False
    state = db.get(RuntimeState, 1)
    if state and state.active_contact_id == contact_id:
        state.active_contact_id = None
    db.delete(contact)
    db.commit()
    return True
