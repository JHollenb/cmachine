"""Public Saturn boundary for the declared three-process weight-storage panel."""
from saturn_bridge import CInvocation,Session,SurfaceManifest,SlotSpec,TransitionSpec,digest
class CPanel(CInvocation):
    def __init__(self,commands,request,controller):
        self.commands=[list(map(str,c)) for c in commands]
        if len(self.commands)!=3 or self.commands!=request['execution_commands']:raise ValueError('Panel command binding differs')
        super().__init__([],None,request,controller=controller)
        self.execution={'adapter':'tcm-process-panel-v1','commands':self.commands,'scope':'three sequential C process invocations'}
        self.model_identity=digest({'adapter':'tcm-process-panel-v1','request':self.request,'commands':self.commands})
    def surface(self,state):
        return SurfaceManifest(slots=(SlotSpec('request',role='raw-input-and-program-binding',writable=False,persistence='authoritative',producer='caller',consumer='three C processes'),SlotSpec('result',role='complete-panel-result',writable=False,persistence='derived',producer='three C processes',consumer='comparison')),state_schema='tcm-weight-panel-v1',consumer='complete panel',horizon='three sequential process invocations')
    def transition(self,state):return TransitionSpec(operation='execute-c-weight-panel-v1',reads=('request',),writes=('result',),consumer='complete panel',footprint='declared')
def invoke(commands,output,request,controller):
    adapter=CPanel(commands,request,controller);session=Session(adapter,{'request':dict(request),'result':None});before=session.capture(retain=False);receipt=session.continue_();after=session.capture(retain=False);row=receipt.to_dict()
    if row['parent']!=before.fingerprint or row['result']!=after.fingerprint or adapter.calls!=1:raise RuntimeError('Panel receipt failed to close')
    return session.read('result'),{'receipt':row,'before':before.fingerprint,'after':after.fingerprint,'request':request,'scope':'three sequential C invocations; internal token/layer boundaries tested separately'}
