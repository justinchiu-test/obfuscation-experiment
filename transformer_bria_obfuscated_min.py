import inspect as I
from typing import Any as Y,Dict as D,List as L,Optional as O,Tuple as T,Union as U
import numpy as np
import torch as t
import torch.nn as n
import torch.nn.functional as F
from ...configuration_utils import ConfigMixin as M,register_to_config as r
from ...loaders import FromOriginalModelMixin as X,PeftAdapterMixin as P
from ...utils import USE_PEFT_BACKEND as B,logging,scale_lora_layers as sl,unscale_lora_layers as ul
from ...utils.torch_utils import maybe_allow_in_graph as g
from ..attention import AttentionModuleMixin as A,FeedForward as f
from ..attention_dispatch import dispatch_attention_fn as d
from ..cache_utils import CacheMixin as C
from ..embeddings import TimestepEmbedding as E,apply_rotary_emb as a,get_timestep_embedding as e
from ..modeling_outputs import Transformer2DModelOutput as R
from ..modeling_utils import ModelMixin as m
from ..normalization import AdaLayerNormContinuous as N,AdaLayerNormZero as Z,AdaLayerNormZeroSingle as S
l=logging.get_logger(__name__)
def _(at,h,eh=None):
 q=at.to_q(h)
 k=at.to_k(h)
 v=at.to_v(h)
 eq=ek=ev=None
 if eh is not None and at.added_kv_proj_dim is not None:
  eq=at.add_q_proj(eh)
  ek=at.add_k_proj(eh)
  ev=at.add_v_proj(eh)
 return q,k,v,eq,ek,ev
def __(at,h,eh=None):
 q,k,v=at.to_qkv(h).chunk(3,dim=-1)
 eq=ek=ev=(None,)
 if eh is not None and hasattr(at,"to_added_qkv"):eq,ek,ev=at.to_added_qkv(eh).chunk(3,dim=-1)
 return q,k,v,eq,ek,ev
def ___(at,h,eh=None):
 if at.fused_projections:return __(at,h,eh)
 return _(at,h,eh)
def o(d:int,p:U[np.ndarray,int],th:float=10000.0,u=False,lf=1.0,nf=1.0,ri=True,fd=t.float32):
 assert d%2==0
 if isinstance(p,int):p=t.arange(p)
 if isinstance(p,np.ndarray):p=t.from_numpy(p)
 th=th*nf
 fr=(1.0/(th**(t.arange(0,d,2,dtype=fd,device=p.device)[:(d//2)]/d))/lf)
 fr=t.outer(p,fr)
 if u and ri:
  fc=fr.cos().repeat_interleave(2,dim=1).float()
  fs=fr.sin().repeat_interleave(2,dim=1).float()
  return fc,fs
 elif u:
  fc=t.cat([fr.cos(),fr.cos()],dim=-1).float()
  fs=t.cat([fr.sin(),fr.sin()],dim=-1).float()
  return fc,fs
 else:
  frc=t.polar(t.ones_like(fr),fr)
  return frc
class Q:
 _attention_backend=None
 def __init__(s):
  if not hasattr(F,"scaled_dot_product_attention"):raise ImportError(f"{s.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
 def __call__(s,at,h:t.Tensor,eh:t.Tensor=None,am:O[t.Tensor]=None,ir:O[t.Tensor]=None)->t.Tensor:
  q,k,v,eq,ek,ev=___(at,h,eh)
  q=q.unflatten(-1,(at.heads,-1))
  k=k.unflatten(-1,(at.heads,-1))
  v=v.unflatten(-1,(at.heads,-1))
  q=at.norm_q(q)
  k=at.norm_k(k)
  if at.added_kv_proj_dim is not None:
   eq=eq.unflatten(-1,(at.heads,-1))
   ek=ek.unflatten(-1,(at.heads,-1))
   ev=ev.unflatten(-1,(at.heads,-1))
   eq=at.norm_added_q(eq)
   ek=at.norm_added_k(ek)
   q=t.cat([eq,q],dim=1)
   k=t.cat([ek,k],dim=1)
   v=t.cat([ev,v],dim=1)
  if ir is not None:
   q=a(q,ir,sequence_dim=1)
   k=a(k,ir,sequence_dim=1)
  h=d(q,k,v,attn_mask=am,backend=s._attention_backend)
  h=h.flatten(2,3)
  h=h.to(q.dtype)
  if eh is not None:
   eh,h=h.split_with_sizes([eh.shape[1],h.shape[1]-eh.shape[1]],dim=1)
   h=at.to_out[0](h)
   h=at.to_out[1](h)
   eh=at.to_add_out(eh)
   return h,eh
  else:return h
class W(t.nn.Module,A):
 _default_processor_cls=Q
 _available_processors=[Q]
 def __init__(s,qd:int,hs:int=8,dh:int=64,dr:float=0.0,bi:bool=False,akp:O[int]=None,apb:O[bool]=True,ob:bool=True,ep:float=1e-5,od:int=None,cpo:O[bool]=None,po:bool=False,ea:bool=True,pr=None):
  super().__init__()
  s.head_dim=dh
  s.inner_dim=od if od is not None else dh*hs
  s.query_dim=qd
  s.use_bias=bi
  s.dropout=dr
  s.out_dim=od if od is not None else qd
  s.context_pre_only=cpo
  s.pre_only=po
  s.heads=od//dh if od is not None else hs
  s.added_kv_proj_dim=akp
  s.added_proj_bias=apb
  s.norm_q=t.nn.RMSNorm(dh,eps=ep,elementwise_affine=ea)
  s.norm_k=t.nn.RMSNorm(dh,eps=ep,elementwise_affine=ea)
  s.to_q=t.nn.Linear(qd,s.inner_dim,bias=bi)
  s.to_k=t.nn.Linear(qd,s.inner_dim,bias=bi)
  s.to_v=t.nn.Linear(qd,s.inner_dim,bias=bi)
  if not s.pre_only:
   s.to_out=t.nn.ModuleList([])
   s.to_out.append(t.nn.Linear(s.inner_dim,s.out_dim,bias=ob))
   s.to_out.append(t.nn.Dropout(dr))
  if akp is not None:
   s.norm_added_q=t.nn.RMSNorm(dh,eps=ep)
   s.norm_added_k=t.nn.RMSNorm(dh,eps=ep)
   s.add_q_proj=t.nn.Linear(akp,s.inner_dim,bias=apb)
   s.add_k_proj=t.nn.Linear(akp,s.inner_dim,bias=apb)
   s.add_v_proj=t.nn.Linear(akp,s.inner_dim,bias=apb)
   s.to_add_out=t.nn.Linear(s.inner_dim,qd,bias=ob)
  if pr is None:pr=s._default_processor_cls()
  s.set_processor(pr)
 def forward(s,h:t.Tensor,eh:O[t.Tensor]=None,am:O[t.Tensor]=None,ir:O[t.Tensor]=None,**k)->t.Tensor:
  ap=set(I.signature(s.processor.__call__).parameters.keys())
  qa={"ip_adapter_masks","ip_hidden_states"}
  uk=[x for x,_ in k.items()if x not in ap and x not in qa]
  if len(uk)>0:l.warning(f"attention_kwargs {uk} are not expected by {s.processor.__class__.__name__} and will be ignored.")
  k={x:w for x,w in k.items()if x in ap}
  return s.processor(s,h,eh,am,ir,**k)
class V(t.nn.Module):
 def __init__(s,th:int,ad:L[int]):
  super().__init__()
  s.theta=th
  s.axes_dim=ad
 def forward(s,i:t.Tensor)->t.Tensor:
  na=i.shape[-1]
  co=[]
  so=[]
  p=i.float()
  im=i.device.type=="mps"
  fd=t.float32 if im else t.float64
  for j in range(na):
   c,si=o(s.axes_dim[j],p[:,j],theta=s.theta,ri=True,u=True,fd=fd)
   co.append(c)
   so.append(si)
  fc=t.cat(co,dim=-1).to(i.device)
  fs=t.cat(so,dim=-1).to(i.device)
  return fc,fs
class K(n.Module):
 def __init__(s,nc:int,fsc:bool,dfs:float,sc:int=1,tt=10000):
  super().__init__()
  s.num_channels=nc
  s.flip_sin_to_cos=fsc
  s.downscale_freq_shift=dfs
  s.scale=sc
  s.time_theta=tt
 def forward(s,ts):
  te=e(ts,s.num_channels,flip_sin_to_cos=s.flip_sin_to_cos,downscale_freq_shift=s.downscale_freq_shift,scale=s.scale,max_period=s.time_theta)
  return te
class H(n.Module):
 def __init__(s,ed,tt):
  super().__init__()
  s.time_proj=K(num_channels=256,fsc=True,dfs=0,tt=tt)
  s.timestep_embedder=E(in_channels=256,time_embed_dim=ed)
 def forward(s,ts,dt):
  tp=s.time_proj(ts)
  te=s.timestep_embedder(tp.to(dtype=dt))
  return te
class J(t.nn.Module):
 def __init__(s,th:int,ad:L[int]):
  super().__init__()
  s.theta=th
  s.axes_dim=ad
 def forward(s,i:t.Tensor)->t.Tensor:
  na=i.shape[-1]
  co=[]
  so=[]
  p=i.float()
  im=i.device.type=="mps"
  fd=t.float32 if im else t.float64
  for j in range(na):
   c,si=o(s.axes_dim[j],p[:,j],theta=s.theta,ri=True,u=True,fd=fd)
   co.append(c)
   so.append(si)
  fc=t.cat(co,dim=-1).to(i.device)
  fs=t.cat(so,dim=-1).to(i.device)
  return fc,fs
@g
class G(n.Module):
 def __init__(s,di:int,na:int,ah:int,qn:str="rms_norm",ep:float=1e-6):
  super().__init__()
  s.norm1=Z(di)
  s.norm1_context=Z(di)
  s.attn=W(query_dim=di,akp=di,dh=ah,hs=na,od=di,cpo=False,bi=True,pr=Q(),ep=ep)
  s.norm2=n.LayerNorm(di,elementwise_affine=False,eps=1e-6)
  s.ff=f(dim=di,dim_out=di,activation_fn="gelu-approximate")
  s.norm2_context=n.LayerNorm(di,elementwise_affine=False,eps=1e-6)
  s.ff_context=f(dim=di,dim_out=di,activation_fn="gelu-approximate")
 def forward(s,h:t.Tensor,eh:t.Tensor,tb:t.Tensor,ir:O[T[t.Tensor,t.Tensor]]=None,ak:O[D[str,Y]]=None)->T[t.Tensor,t.Tensor]:
  nh,gm,sm,scm,glm=s.norm1(h,emb=tb)
  ne,cgm,csm,cscm,cglm=s.norm1_context(eh,emb=tb)
  ak=ak or {}
  ao=s.attn(hidden_states=nh,encoder_hidden_states=ne,image_rotary_emb=ir,**ak)
  if len(ao)==2:ato,cao=ao
  elif len(ao)==3:ato,cao,iao=ao
  ato=gm.unsqueeze(1)*ato
  h=h+ato
  nh=s.norm2(h)
  nh=nh*(1+scm[:,None])+sm[:,None]
  fo=s.ff(nh)
  fo=glm.unsqueeze(1)*fo
  h=h+fo
  if len(ao)==3:h=h+iao
  cao=cgm.unsqueeze(1)*cao
  eh=eh+cao
  ne=s.norm2_context(eh)
  ne=ne*(1+cscm[:,None])+csm[:,None]
  cfo=s.ff_context(ne)
  eh=eh+cglm.unsqueeze(1)*cfo
  if eh.dtype==t.float16:eh=eh.clip(-65504,65504)
  return eh,h
@g
class q(n.Module):
 def __init__(s,di:int,na:int,ah:int,mr:float=4.0):
  super().__init__()
  s.mlp_hidden_dim=int(di*mr)
  s.norm=S(di)
  s.proj_mlp=n.Linear(di,s.mlp_hidden_dim)
  s.act_mlp=n.GELU(approximate="tanh")
  s.proj_out=n.Linear(di+s.mlp_hidden_dim,di)
  pr=Q()
  s.attn=W(query_dim=di,dh=ah,hs=na,od=di,bi=True,pr=pr,ep=1e-6,po=True)
 def forward(s,h:t.Tensor,eh:t.Tensor,tb:t.Tensor,ir:O[T[t.Tensor,t.Tensor]]=None,ak:O[D[str,Y]]=None)->T[t.Tensor,t.Tensor]:
  ts=eh.shape[1]
  h=t.cat([eh,h],dim=1)
  rs=h
  nh,gt=s.norm(h,emb=tb)
  mh=s.act_mlp(s.proj_mlp(nh))
  ak=ak or {}
  ao=s.attn(hidden_states=nh,image_rotary_emb=ir,**ak)
  h=t.cat([ao,mh],dim=2)
  gt=gt.unsqueeze(1)
  h=gt*s.proj_out(h)
  h=rs+h
  if h.dtype==t.float16:h=h.clip(-65504,65504)
  eh,h=h[:,:ts],h[:,ts:]
  return eh,h
class w(m,M,P,X,C):
 _supports_gradient_checkpointing=True
 @r
 def __init__(s,patch_size:int=1,in_channels:int=64,num_layers:int=19,num_single_layers:int=38,attention_head_dim:int=128,num_attention_heads:int=24,joint_attention_dim:int=4096,pooled_projection_dim:int=None,guidance_embeds:bool=False,axes_dims_rope:L[int]=[16,56,56],rope_theta=10000,time_theta=10000):
  super().__init__()
  s.out_channels=in_channels
  s.inner_dim=s.config.num_attention_heads*s.config.attention_head_dim
  s.pos_embed=V(theta=rope_theta,ad=axes_dims_rope)
  s.time_embed=H(embedding_dim=s.inner_dim,tt=time_theta)
  if guidance_embeds:s.guidance_embed=H(embedding_dim=s.inner_dim)
  s.context_embedder=n.Linear(s.config.joint_attention_dim,s.inner_dim)
  s.x_embedder=t.nn.Linear(s.config.in_channels,s.inner_dim)
  s.transformer_blocks=n.ModuleList([G(di=s.inner_dim,na=s.config.num_attention_heads,ah=s.config.attention_head_dim)for i in range(s.config.num_layers)])
  s.single_transformer_blocks=n.ModuleList([q(di=s.inner_dim,na=s.config.num_attention_heads,ah=s.config.attention_head_dim)for i in range(s.config.num_single_layers)])
  s.norm_out=N(s.inner_dim,s.inner_dim,elementwise_affine=False,eps=1e-6)
  s.proj_out=n.Linear(s.inner_dim,patch_size*patch_size*s.out_channels,bias=True)
  s.gradient_checkpointing=False
 def forward(s,h:t.Tensor,eh:t.Tensor=None,pp:t.Tensor=None,ts:t.LongTensor=None,ii:t.Tensor=None,ti:t.Tensor=None,gu:t.Tensor=None,ak:O[D[str,Y]]=None,rd:bool=True,cbs=None,csbs=None)->U[T[t.Tensor],R]:
  if ak is not None:
   ak=ak.copy()
   ls=ak.pop("scale",1.0)
  else:ls=1.0
  if B:sl(s,ls)
  else:
   if ak is not None and ak.get("scale",None)is not None:l.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")
  h=s.x_embedder(h)
  ts=ts.to(h.dtype)
  if gu is not None:gu=gu.to(h.dtype)
  else:gu=None
  tb=s.time_embed(ts,dtype=h.dtype)
  if gu:tb+=s.guidance_embed(gu,dtype=h.dtype)
  eh=s.context_embedder(eh)
  if len(ti.shape)==3:ti=ti[0]
  if len(ii.shape)==3:ii=ii[0]
  i=t.cat((ti,ii),dim=0)
  ir=s.pos_embed(i)
  for ib,b in enumerate(s.transformer_blocks):
   if t.is_grad_enabled()and s.gradient_checkpointing:eh,h=s._gradient_checkpointing_func(b,h,eh,tb,ir,ak)
   else:eh,h=b(hidden_states=h,encoder_hidden_states=eh,temb=tb,image_rotary_emb=ir)
   if cbs is not None:
    ic=len(s.transformer_blocks)/len(cbs)
    ic=int(np.ceil(ic))
    h=h+cbs[ib//ic]
  for ib,b in enumerate(s.single_transformer_blocks):
   if t.is_grad_enabled()and s.gradient_checkpointing:eh,h=s._gradient_checkpointing_func(b,h,eh,tb,ir,ak)
   else:eh,h=b(hidden_states=h,encoder_hidden_states=eh,temb=tb,image_rotary_emb=ir)
   if csbs is not None:
    ic=len(s.single_transformer_blocks)/len(csbs)
    ic=int(np.ceil(ic))
    h[:,eh.shape[1]:,...]=h[:,eh.shape[1]:,...]+csbs[ib//ic]
  h=s.norm_out(h,tb)
  ou=s.proj_out(h)
  if B:ul(s,ls)
  if not rd:return(ou,)
  return R(sample=ou)