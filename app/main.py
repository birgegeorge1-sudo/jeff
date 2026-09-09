import os, re
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from .config import HOST, PORT, MAX_HISTORY
from .db import SessionLocal, init_db, get_or_create_contact, find_contact, set_active_contact, get_active_contact, save_message, recent_messages, get_memories, upsert_memory, delete_contact
from .db import add_event, Contact
from .engine import generate_reply
from .persona import load_persona
from .planner import boredom_heuristic, question_frequency

app=FastAPI(title='Relationship Copilot',version='1.0.0')

@contextmanager
def db_session():
    db=SessionLocal()
    try: yield db
    finally: db.close()

class CommandIn(BaseModel): command:str=Field(min_length=1,max_length=220)
class MessageIn(BaseModel): contact:str=Field(min_length=1,max_length=160); message:str=Field(min_length=1,max_length=12000); platform:str='manual'; external_message_id:str|None=None
class SendDraftIn(BaseModel): contact:str=Field(min_length=1,max_length=160); content:str=Field(min_length=1,max_length=12000); platform:str='manual'

@app.on_event('startup')
def startup(): init_db()

@app.get('/health')
def health(): return {'ok':True,'service':'relationship-copilot','version':'1.0.0'}

@app.get('/persona')
def persona(): return load_persona()

@app.get('/conversations')
def conversations():
    with db_session() as db:
        rows=list(db.scalars(select(Contact).order_by(Contact.display_name)))
        active=get_active_contact(db)
        return [{'id':r.id,'name':r.display_name,'active':bool(active and active.id==r.id),'stage':r.relationship_stage,'vibe':r.vibe_score,'engagement':r.engagement_score,'date_readiness':r.date_readiness} for r in rows]

@app.get('/conversations/{name}')
def conversation(name:str):
    with db_session() as db:
        c=find_contact(db,name)
        if not c: raise HTTPException(404, f'No conversation exists for {name}.')
        msgs=recent_messages(db,c.id,100);mems=get_memories(db,c.id,100)
        return {'id':c.id,'name':c.display_name,'stage':c.relationship_stage,'vibe':c.vibe_score,'engagement':c.engagement_score,'date_readiness':c.date_readiness,
                'messages':[{'id':m.id,'sender':m.sender,'content':m.content,'platform':m.platform,'created_at':m.created_at.isoformat()} for m in msgs],
                'memories':[{'key':m.key,'value':m.value,'importance':m.importance} for m in mems]}

@app.post('/command')
def command(body:CommandIn):
    cmd=body.command.strip()
    # !newconvo_Name creates/activates a conversation. !contconvo_Name continues an existing one.
    if cmd.lower().startswith('!deleteconvo_'):
        name=cmd[len('!deleteconvo_'):].strip()
        if not name: raise HTTPException(400,'A name is required.')
        with db_session() as db:
            c=find_contact(db,name)
            if not c: raise HTTPException(404,f'No conversation exists for {name}.')
            deleted=delete_contact(db,c.id)
            return {'ok':deleted,'mode':'delete','deleted_contact':name}
    for prefix,mode in (('!newconvo_','new'),('!contconvo_','continue')):
        if cmd.lower().startswith(prefix):
            name=cmd[len(prefix):].strip()
            if not name: raise HTTPException(400,'A name is required.')
            with db_session() as db:
                c=get_or_create_contact(db,name)
                if mode=='continue' and not recent_messages(db,c.id,1):
                    raise HTTPException(404,f'No existing conversation for {c.display_name}. Use !newconvo_{c.display_name} first.')
                set_active_contact(db,c)
                add_event(db,c.id,'conversation_activated',mode)
                return {'ok':True,'mode':mode,'active_contact':c.display_name,'contact_id':c.id}
    raise HTTPException(400,'Unknown command. Use !newconvo_Name, !contconvo_Name, or !deleteconvo_Name.')

@app.post('/message')
def incoming_message(body:MessageIn):
    with db_session() as db:
        active=get_active_contact(db); c=get_or_create_contact(db,body.contact)
        if not active or active.id!=c.id:
            raise HTTPException(409,{'error':'inactive_contact','message':f'{c.display_name} is not active.','command':f'!contconvo_{c.display_name}'})
        incoming=save_message(db,c.id,'incoming',body.message,body.platform,body.external_message_id)
        history=recent_messages(db,c.id,MAX_HISTORY)
        heuristic=boredom_heuristic(history)
        result=generate_reply(load_persona(),c,history,get_memories(db,c.id,50),body.message)
        result.setdefault('boredom',{})['heuristic']=heuristic
        result['coaching_note']=(result.get('coaching_note','') + f" Question frequency in recent drafts: {question_frequency(history):.0%}.").strip()
        vibe=result.get('vibe',{}) or {}; boredom=result.get('boredom',{}) or {}
        c.relationship_stage=result.get('relationship_stage',c.relationship_stage)
        c.vibe_score=float(vibe.get('momentum',c.vibe_score) or c.vibe_score)
        c.engagement_score=float(vibe.get('reciprocity',c.engagement_score) or c.engagement_score)
        c.date_readiness=float(result.get('date_readiness',c.date_readiness) or c.date_readiness)
        add_event(db,c.id,'analysis',detail=str(boredom.get('reasons',[])),score=float(boredom.get('score',0) or 0))
        for item in result.get('memory_updates',[])[:12]:
            key=str(item.get('key','')).strip(); val=str(item.get('value','')).strip()
            if key and val: upsert_memory(db,c.id,key,val,max(1,min(5,int(item.get('importance',3) or 3))),incoming.id)
        db.commit()
        return {'contact':c.display_name,'active_contact_id':c.id,'draft':result,'guard':'Draft only — review before sending.'}

@app.post('/draft-sent')
def draft_sent(body:SendDraftIn):
    with db_session() as db:
        active=get_active_contact(db);c=get_or_create_contact(db,body.contact)
        if not active or active.id!=c.id: raise HTTPException(409,'Contact is not active.')
        m=save_message(db,c.id,'me',body.content,body.platform)
        return {'ok':True,'message_id':m.id}

if __name__=='__main__':
    import uvicorn
    uvicorn.run('app.main:app',host=HOST,port=PORT,reload=False)
