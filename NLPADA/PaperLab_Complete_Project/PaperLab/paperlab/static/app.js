const $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, body) {
 const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-PaperLab':'1'},body:JSON.stringify(body)});
 const data = await response.json(); if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)); return data;
}
function message(id, text, error=false){ $(id).textContent=text; $(id).classList.toggle('error',error); }
async function status(){
 const data=await api('/api/status');
 $('doc-list').innerHTML=data.documents.map(d=>`<a class="doc" href="/api/papers/${encodeURIComponent(d.name)}" target="_blank" rel="noopener">▤ ${esc(d.name)}<small>${d.searchable_pages} searchable pages</small></a>`).join('');
 $('stats').innerHTML=[[data.documents.length,'papers'],[data.searchable_pages,'pages'],[data.chunks,'passages']].map(([v,l])=>`<div class="stat"><strong>${v}</strong><span>${l}</span></div>`).join('');
 if(data.issues.length)message('upload-status',data.issues.join(' '),true);
}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!=='view-'+b.dataset.view);
 document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n===b));
 if(b.dataset.view==='results')loadResults();
}));
document.querySelectorAll('[data-question]').forEach(b=>b.addEventListener('click',()=>{$('question').value=b.dataset.question; $('question').focus();}));
$('upload').addEventListener('change',async e=>{
 const file=e.target.files[0];if(!file)return;
 message('upload-status','Reading and indexing PDF…');
 try{
  if(file.size>20*1024*1024)throw Error('Choose a PDF no larger than 20 MiB.');
  const r=await fetch('/api/upload',{method:'POST',headers:{'X-PaperLab':'1','X-Filename':file.name,'Content-Type':'application/pdf'},body:file});
  const d=await r.json();if(!r.ok)throw Error(d.detail);
  message('upload-status',d.status==='already_present'?'This paper is already in your collection.':'Paper added. Your collection is ready.');await status();
 }catch(err){message('upload-status',err.message,true);}finally{e.target.value='';}
});
let latest=null;
function saveAnswer(format){
 if(!latest)return;
 const d=latest;
 const text=format==='json'?JSON.stringify(d,null,2):`# PaperLab evidence report\n\nQuestion: ${d.question}\n\nStatus: ${d.status}\n\n${d.text}\n\nCitation: ${d.citation||'None'}\n\nEvidence: ${d.quote||'None'}\n\nMode: ${d.generator}; retriever: ${d.retriever}\n\n${d.verification}\n`;
 const url=URL.createObjectURL(new Blob([text],{type:format==='json'?'application/json':'text/markdown'}));const a=document.createElement('a');a.href=url;a.download=`paperlab_answer.${format}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('ask-form').addEventListener('submit',async e=>{
 e.preventDefault();$('ask-button').disabled=true;message('ask-status','Searching your collection. The first FLAN request may take longer while the model downloads.');$('answer').innerHTML='';
 try{
  const d=await api('/api/ask',{question:$('question').value,retriever:$('retriever').value,generator:$('generator').value,k:Number($('k').value)});latest=d;
  const accepted=d.status==='answered';
  const match=d.citation?d.hits.find(h=>`${h.passage.file}, p. ${h.passage.page}`===d.citation):null;
  const citation=match?`<a target="_blank" rel="noopener" href="/api/papers/${encodeURIComponent(match.passage.file)}#page=${match.passage.page}">${esc(d.citation)} ↗</a>`:'';
  $('answer').innerHTML=`<article class="card"><span class="tag ${accepted?'':'warn'}">${accepted?(d.generator==='extractive'?'Relevant evidence':'Literal span verified'):'No verified answer'}</span><span class="meta">${Math.round(d.latency_ms)} ms · ${esc(d.retriever)} · ${esc(d.generator)}</span><p class="answer-text">${esc(d.text)}</p>${citation}${d.quote?`<blockquote>${esc(d.quote)}</blockquote>`:''}<p class="meta">${esc(d.verification)}</p><button class="secondary" id="save-md">Download Markdown</button><button class="secondary" id="save-json">Download JSON</button><details><summary>Inspect ${d.hits.length} retrieved passages</summary>${d.hits.map(h=>`<div class="hit"><b>${esc(h.passage.file)} · page ${h.passage.page} · chunk ${h.passage.chunk}</b><p>${esc(h.passage.text)}</p><span class="meta">Ranking score ${h.score.toFixed(3)} · Query term coverage ${(h.coverage*100).toFixed(1)}% (not confidence)</span></div>`).join('')}</details></article>`;
  $('save-md').onclick=()=>saveAnswer('md');$('save-json').onclick=()=>saveAnswer('json');message('ask-status','');
 }catch(err){message('ask-status',err.message,true);}finally{$('ask-button').disabled=false;}
});
$('lab-form').addEventListener('submit',async e=>{
 e.preventDefault();const button=e.target.querySelector('button');button.disabled=true;message('lab-status','Tokenizing and sampling…');
 try{const d=await api('/api/lab',{text:$('lab-text').value});$('lab-output').innerHTML=`<div class="card"><h2>Custom byte-level BPE</h2><p>${d.utf8_bytes} UTF-8 bytes → ${d.bpe_tokens} tokens · ${d.vocabulary} vocabulary entries</p><pre>${esc(JSON.stringify(d.ids))}</pre><p class="meta">Round trip: ${esc(d.round_trip)}</p><h2>Mini-GPT continuation</h2><pre>${esc(d.sample)}</pre><p class="meta">${d.parameters.toLocaleString()} trainable parameters. ${esc(d.warning)}</p></div>`;message('lab-status','');}catch(err){message('lab-status',err.message,true);}finally{button.disabled=false;}
});
async function loadResults(){
 try{
  const d=await api('/api/results');let html='';
  if(d.training){const t=d.training;html+=`<div class="card"><h2>Mini-GPT training</h2><p>${t.parameters.toLocaleString()} parameters · ${t.steps} optimizer steps · ${t.documents.train} training documents · ${t.documents.validation} validation documents</p><div class="table-wrap"><table><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss</th></tr></thead><tbody>${t.history.map(h=>`<tr><td>${h.step}</td><td>${h.train_loss.toFixed(3)}</td><td>${h.validation_loss.toFixed(3)}</td></tr>`).join('')}</tbody></table></div><p class="meta">Saved step ${t.best_step} · Perplexity ${t.validation_perplexity.toFixed(2)} · Loss in nats per BPE token. This is not QA accuracy.</p></div>`;}
  if(d.benchmark){const b=d.benchmark;html+=`<div class="card"><h2>Retrieval comparison</h2><p class="meta">${esc(b.scope)}</p><div class="table-wrap"><table><thead><tr><th>Retriever</th><th>Page Hit@3</th><th>Page MRR@3</th><th>Median ms</th></tr></thead><tbody>${Object.entries(b.retrieval).map(([name,v])=>`<tr><td>${esc(name)}</td><td>${(v.metrics['page_hit@3']*100).toFixed(1)}%</td><td>${v.metrics['page_mrr@3'].toFixed(3)}</td><td>${v.latency_ms.median.toFixed(3)}</td></tr>`).join('')}</tbody></table></div><h2>Answer evaluation</h2><div class="table-wrap"><table><thead><tr><th>Engine</th><th>Exact match</th><th>Answer + page</th><th>Abstention on unknowns</th></tr></thead><tbody>${Object.entries(b.answers).map(([name,v])=>`<tr><td>${esc(name)}</td><td>${(v.metrics.answer_exact_match*100).toFixed(1)}%</td><td>${(v.metrics.answer_and_page_correct*100).toFixed(1)}%</td><td>${(v.metrics.unanswerable_abstention_rate*100).toFixed(1)}%</td></tr>`).join('')}</tbody></table></div><p class="meta">Offline evidence returns sentences, so short-answer exact match penalizes its different output format. See the guide and per-question results before comparing engines. ${esc(b.limitations)}</p><details><summary>Inspect the full benchmark</summary><pre>${esc(JSON.stringify(b,null,2))}</pre></details></div>`;}
  $('results-output').innerHTML=html||'<div class="card">No saved runs yet. Run the training and benchmark commands in the README.</div>';
 }catch(err){$('results-output').textContent=err.message;}
}
status().catch(e=>message('ask-status',e.message,true));
