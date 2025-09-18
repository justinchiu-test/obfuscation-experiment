from typing import Optional as O,Union as U
import torch as T
from torch import nn as N
from ...cache_utils import Cache as C,DynamicCache as D
from ...generation import GenerationMixin as G
from ...masking_utils import create_causal_mask as m
from ...modeling_layers import GenericForQuestionAnswering as Q,GenericForSequenceClassification as S,GenericForTokenClassification as K
from ...modeling_outputs import BaseModelOutputWithPast as B,CausalLMOutputWithPast as L
from ...modeling_utils import PreTrainedModel as P
from ...processing_utils import Unpack as X
from ...utils import TransformersKwargs as W,auto_docstring as a,can_return_tuple as r,logging
from ...utils.generic import check_model_inputs as c
from .configuration_llama import LlamaConfig as F
from ..shared_library import rotate_half,apply_rotary_pos_emb,repeat_kv,eager_attention_forward,RMSNorm as R,BaseMLP as M,BaseRotaryEmbedding as E,BaseAttention as A,BaseDecoderLayer as Y
l=logging.get_logger(__name__)
class Z(R):pass
class I(E):pass
class J(M):
 def __init__(s,g):super().__init__(g,mlp_bias=g.mlp_bias)
class H(A):
 def __init__(s,g:F,i:int):super().__init__(config=g,layer_idx=i,attention_bias=g.attention_bias,sliding_window=None)
class V(Y):
 def __init__(s,g:F,i:int):super().__init__(config=g,layer_idx=i,norm_class=Z,mlp_class=J,attention_class=H)
@a
class o(P):
 config:F
 base_model_prefix="model"
 supports_gradient_checkpointing=True
 _no_split_modules=["LlamaDecoderLayer"]
 _skip_keys_device_placement=["past_key_values"]
 _supports_flash_attn=True
 _supports_sdpa=True
 _supports_flex_attn=True
 _can_compile_fullgraph=True
 _supports_attention_backend=True
 _can_record_outputs={"hidden_states":V,"attentions":H}
@a
class u(o):
 def __init__(s,g:F):
  super().__init__(g)
  s.padding_idx=g.pad_token_id
  s.vocab_size=g.vocab_size
  s.embed_tokens=N.Embedding(g.vocab_size,g.hidden_size,s.padding_idx)
  s.layers=N.ModuleList([V(g,i)for i in range(g.num_hidden_layers)])
  s.norm=Z(g.hidden_size,eps=g.rms_norm_eps)
  s.rotary_emb=I(config=g)
  s.gradient_checkpointing=False
  s.post_init()
 @c
 @a
 def forward(s,input_ids:O[T.LongTensor]=None,attention_mask:O[T.Tensor]=None,position_ids:O[T.LongTensor]=None,past_key_values:O[C]=None,inputs_embeds:O[T.FloatTensor]=None,cache_position:O[T.LongTensor]=None,use_cache:O[bool]=None,**k:X[W])->B:
  if(input_ids is None)^(inputs_embeds is not None):raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
  if inputs_embeds is None:inputs_embeds:T.Tensor=s.embed_tokens(input_ids)
  if use_cache and past_key_values is None:past_key_values=D(config=s.config)
  if cache_position is None:
   p=past_key_values.get_seq_length()if past_key_values is not None else 0
   cache_position:T.Tensor=T.arange(p,p+inputs_embeds.shape[1],device=inputs_embeds.device)
  if position_ids is None:position_ids=cache_position.unsqueeze(0)
  f=m(config=s.config,input_embeds=inputs_embeds,attention_mask=attention_mask,cache_position=cache_position,past_key_values=past_key_values,position_ids=position_ids)
  h=inputs_embeds
  e=s.rotary_emb(h,position_ids)
  for d in s.layers[:s.config.num_hidden_layers]:h=d(h,attention_mask=f,position_ids=position_ids,past_key_values=past_key_values,cache_position=cache_position,position_embeddings=e,**k)
  h=s.norm(h)
  return B(last_hidden_state=h,past_key_values=past_key_values)
@a
class t(o,G):
 _tied_weights_keys=["lm_head.weight"]
 _tp_plan={"lm_head":"colwise_rep"}
 _pp_plan={"lm_head":(["hidden_states"],["logits"])}
 def __init__(s,g):
  super().__init__(g)
  s.model=u(g)
  s.vocab_size=g.vocab_size
  s.lm_head=N.Linear(g.hidden_size,g.vocab_size,bias=False)
  s.post_init()
 def set_decoder(s,d):s.model=d
 def get_decoder(s):return s.model
 @r
 @a
 def forward(s,input_ids:O[T.LongTensor]=None,attention_mask:O[T.Tensor]=None,position_ids:O[T.LongTensor]=None,past_key_values:O[C]=None,inputs_embeds:O[T.FloatTensor]=None,labels:O[T.LongTensor]=None,use_cache:O[bool]=None,cache_position:O[T.LongTensor]=None,logits_to_keep:U[int,T.Tensor]=0,**k:X[W])->L:
  """
  Example:

  ```python
  >>> from transformers import AutoTokenizer, LlamaForCausalLM

  >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
  >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

  >>> prompt = "Hey, are you conscious? Can you talk to me?"
  >>> inputs = tokenizer(prompt, return_tensors="pt")

  >>> # Generate
  >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
  >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
  ```"""
  o:B=s.model(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds,use_cache=use_cache,cache_position=cache_position,**k)
  h=o.last_hidden_state
  i=slice(-logits_to_keep,None)if isinstance(logits_to_keep,int)else logits_to_keep
  g=s.lm_head(h[:,i,:])
  n=None
  if labels is not None:n=s.loss_function(logits=g,labels=labels,vocab_size=s.config.vocab_size,**k)
  return L(loss=n,logits=g,past_key_values=o.past_key_values,hidden_states=o.hidden_states,attentions=o.attentions)
class b(S,o):...
class x(Q,o):base_model_prefix="transformer"
class y(K,o):...
__all__=["t","u","o","b","x","y"]