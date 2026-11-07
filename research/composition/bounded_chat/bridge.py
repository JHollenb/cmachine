"""Bind complete persistent-process argv vectors at the public Saturn panel boundary."""
from saturn_bridge import CInvocation,Session,SurfaceManifest,SlotSpec,TransitionSpec,digest
class ChatPanel(CInvocation):
    def __init__(self,commands,request,controller):
        self.commands=[list(map(str,c)) for c in commands]
        if not self.commands or self.commands!=request['execution_commands']:raise ValueError('Persistent command binding differs')
        super().__init__([],None,request,controller=controller);self.execution={'adapter':'tcm-chat-panel-v1','commands':self.commands,'scope':'sequential persistent C/client process invocations'};self.model_identity=digest({'adapter':'tcm-chat-panel-v1','request':self.request,'commands':self.commands})
    def surface(self,state):return SurfaceManifest(slots=(SlotSpec('request',role='raw-input-and-execution-binding',writable=False,persistence='authoritative',producer='caller',consumer='persistent chat processes'),SlotSpec('result',role='complete-panel-result',writable=False,persistence='derived',producer='persistent chat processes',consumer='comparison')),state_schema='tcm-chat-panel-v1',consumer='complete panel',horizon='seven sequential process invocations')
    def transition(self,state):return TransitionSpec(operation='execute-c-chat-panel-v1',reads=('request',),writes=('result',),consumer='complete panel',footprint='declared')
def invoke(commands,request,controller):
    adapter=ChatPanel(commands,request,controller);session=Session(adapter,{'request':dict(request),'result':None});before=session.capture(retain=False);receipt=session.continue_();after=session.capture(retain=False);row=receipt.to_dict()
    if row['parent']!=before.fingerprint or row['result']!=after.fingerprint or adapter.calls!=1:raise RuntimeError('Persistent panel receipt did not close')
    return session.read('result'),{'receipt':row,'before':before.fingerprint,'after':after.fingerprint,'request':request,'scope':'seven sequential persistent process invocations; internal token ownership audited separately'}
