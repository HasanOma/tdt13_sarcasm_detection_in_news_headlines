/* Minimal BERT uncased tokenizer (basic + WordPiece), matching the
   Python DistilBertTokenizer used in this repo. Vocab from bert_vocab.json. */
"use strict";
function bertTokenizerFactory(vocab){
  const UNK="[UNK]", MAX_CHARS=100;
  const isWs=c=>c===" "||c==="\t"||c==="\n"||c==="\r"||/\p{Zs}/u.test(c);
  const isCtrl=c=>{
    if(c==="\t"||c==="\n"||c==="\r") return false;
    return /\p{Cc}|\p{Cf}/u.test(c);
  };
  const isPunct=c=>{
    const cp=c.codePointAt(0);
    if((cp>=33&&cp<=47)||(cp>=58&&cp<=64)||(cp>=91&&cp<=96)||(cp>=123&&cp<=126)) return true;
    return /\p{P}/u.test(c);
  };
  const isCJK=cp=>(cp>=0x4E00&&cp<=0x9FFF)||(cp>=0x3400&&cp<=0x4DBF)||(cp>=0x20000&&cp<=0x2A6DF)||
    (cp>=0x2A700&&cp<=0x2B73F)||(cp>=0x2B740&&cp<=0x2B81F)||(cp>=0x2B820&&cp<=0x2CEAF)||
    (cp>=0xF900&&cp<=0xFAFF)||(cp>=0x2F800&&cp<=0x2FA1F);

  function cleanText(text){
    let out="";
    for(const c of text){
      const cp=c.codePointAt(0);
      if(cp===0||cp===0xFFFD||isCtrl(c)) continue;
      out+=isWs(c)?" ":c;
    }
    return out;
  }
  function padCJK(text){
    let out="";
    for(const c of text){
      out+=isCJK(c.codePointAt(0))?" "+c+" ":c;
    }
    return out;
  }
  function stripAccents(t){
    return t.normalize("NFD").replace(/\p{Mn}/gu,"");
  }
  function splitOnPunc(t){
    const out=[];let cur="";
    for(const c of t){
      if(isPunct(c)){ if(cur)out.push(cur); out.push(c); cur=""; }
      else cur+=c;
    }
    if(cur)out.push(cur);
    return out;
  }
  function basicTokenize(text){
    text=padCJK(cleanText(text));
    const split=text.split(" ").filter(Boolean);
    const out=[];
    for(let tok of split){
      tok=stripAccents(tok.toLowerCase());
      out.push(...splitOnPunc(tok));
    }
    return out.join(" ").split(" ").filter(Boolean);
  }
  function wordpiece(token){
    if([...token].length>MAX_CHARS) return [UNK];
    const chars=[...token];
    const out=[];let start=0;
    while(start<chars.length){
      let end=chars.length, cur=null;
      while(start<end){
        let sub=chars.slice(start,end).join("");
        if(start>0) sub="##"+sub;
        if(Object.prototype.hasOwnProperty.call(vocab,sub)){ cur=sub; break; }
        end--;
      }
      if(cur===null) return [UNK];
      out.push(cur); start=end;
    }
    return out;
  }
  function encode(text, maxLen){
    const pieces=[];
    for(const t of basicTokenize(text)) pieces.push(...wordpiece(t));
    const body=pieces.slice(0, maxLen-2);
    const ids=[vocab["[CLS]"], ...body.map(p=>vocab[p]), vocab["[SEP]"]];
    const mask=new Array(ids.length).fill(1);
    while(ids.length<maxLen){ ids.push(vocab["[PAD]"]); mask.push(0); }
    return {ids, mask};
  }
  return {encode, basicTokenize, wordpiece};
}
if(typeof module!=="undefined") module.exports=bertTokenizerFactory;
