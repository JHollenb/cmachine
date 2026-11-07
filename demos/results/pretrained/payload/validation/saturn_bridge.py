"""Public Saturn receipt around a complete C process invocation.

The typed boundary is the declared raw request and complete returned result.
Internal row/token checkpoint closure is measured by C separately. This wrapper
does not claim that every CUDA operation or internal C slot has a Saturn receipt.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from saturn_pub import Adapter,ExecutionPoint,NumericalContract,Session,SlotSpec,SurfaceManifest,TransitionSpec
from saturn_pub.values import digest


class CInvocation(Adapter):
    def __init__(self,command,output,request,controller=None):
        self.command=list(map(str,command));self.output=Path(output) if output is not None else None;self.request=dict(request);self.controller=controller
        self.request_digest=digest(self.request);self.model_identity=digest({"adapter":"tcm-process-v1","request":self.request})
        self.execution={"adapter":"tcm-process-v1","command":self.command,"scope":"whole C invocation"}
        self.calls=0

    def validate(self,state):
        if set(state)!={"request","result"} or digest(state["request"])!=self.request_digest:
            raise ValueError("C invocation request changed")
        if state["result"] is not None and not isinstance(state["result"],dict):
            raise TypeError("C result must be an object")

    def boundary(self,state):return "complete" if state["result"] is not None else "entry"
    def addresses(self,state):return ("request","result")
    def point(self,state):
        return ExecutionPoint(family="c-state-machine",logical_step=0,phase=self.boundary(state),operator="tcm",index=0,
                              edge="after" if state["result"] is not None else "before",local_clock=0)
    def surface(self,state):
        return SurfaceManifest(slots=(SlotSpec("request",role="raw-input-and-program-binding",writable=False,persistence="authoritative",producer="caller",consumer="C machine"),
                                      SlotSpec("result",role="complete-C-result",writable=False,persistence="derived",producer="C machine",consumer="comparison")),
                               state_schema="tcm-invocation-v1",consumer="complete result",horizon="one process invocation")
    def transition(self,state):
        return TransitionSpec(operation="execute-c-machine-v1",reads=("request",),writes=("result",),consumer="complete result",footprint="declared")
    def numerical_contract(self):
        return NumericalContract(mode="behavior-observed",program="tcm",comparator="native outputs reported separately")
    def advance(self,state):
        if self.calls or state["result"] is not None:raise RuntimeError("C invocation already executed")
        if self.controller is not None:
            result=self.controller(state["request"])
        else:
            process=subprocess.run(self.command,capture_output=True,text=True,timeout=180)
            if process.returncode:raise RuntimeError(f"C process failed ({process.returncode}): {process.stderr[-8000:]}")
            result=json.loads(self.output.read_text())
        self.calls+=1
        return {**state,"result":result}


def invoke(command,output,request,controller=None):
    adapter=CInvocation(command,output,request,controller)
    session=Session(adapter,{"request":dict(request),"result":None})
    before=session.capture(retain=False)
    receipt=session.continue_()
    after=session.capture(retain=False)
    row=receipt.to_dict()
    if row["parent"]!=before.fingerprint or row["result"]!=after.fingerprint or adapter.calls!=1:
        raise RuntimeError("C invocation receipt failed to close")
    return session.read("result"),{"receipt":row,"before":before.fingerprint,"after":after.fingerprint,
                                   "request":request,"scope":"whole C invocation; internal boundaries tested separately"}
