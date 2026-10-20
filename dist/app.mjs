import { EQ, displayEquation, typeset } from './math-render.mjs';
import { TASKS, DEFAULT_CONFIG, ACTIONS, softmax, kl, mean, clippedSurrogate, createLanguageState, languageMetrics, prepareBatch, languageObjective, optimizeEpoch, optimizeBatch, tokenTrace, createGrid, gridStep, epsilonPolicy, gae } from './engine.mjs';
import { STAGES, ALGORITHMS, CONCEPTS, DIALOGS, TERMS } from './content.mjs';
import { createUpdateSequence, currentUpdateFrame, updateComplete } from './update-steps.mjs';
import { renderComputationGraph } from './graph.mjs';
import { SvgMotion, traceFlow, openDetail, closeDetail, switchSurface, waitForMotion, motionPreference } from './motion.mjs';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const f = (n, digits = 3) => Number(Number(n).toFixed(digits)).toFixed(digits).replace('-', '−');
const signed = (n, digits = 3) => (n > 0 ? '+' : '') + f(n, digits);
const percent = p => (p * 100).toFixed(1) + '%';
const responseName = i => String.fromCharCode(65 + i);
const textOf = response => response.tokens.filter(t => t !== '<eos>').join('');
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[c]);
const icons = {
  play:'<path d="m6 3 12 9-12 9z"/>',pause:'<path d="M8 4v16M16 4v16"/>',step:'<path d="m5 4 11 8L5 20zM20 4v16"/>',reset:'<path d="M3 11a9 9 0 1 1 2 7M3 4v7h7"/>',spark:'<path d="m12 2 2.5 7.5L22 12l-7.5 2.5L12 22l-2.5-7.5L2 12l7.5-2.5z"/>',grid:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',help:'<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 0 1 6 0c0 2-3 2-3 4M12 17h.01"/>',sliders:'<path d="M4 6h5m4 0h7M4 12h9m4 0h3M4 18h3m4 0h9"/><circle cx="11" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="9" cy="18" r="2"/>',download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',close:'<path d="m6 6 12 12M6 18 18 6"/>',bulb:'<path d="M9 18h6M10 21h4M8 14a6 6 0 1 1 8 0c-1 1-1 2-1 2H9s0-1-1-2"/>',layers:'<path d="m3 7 9-5 9 5-9 5zM3 12l9 5 9-5M3 17l9 5 9-5"/>',agent:'<rect x="4" y="6" width="16" height="14" rx="4"/><path d="M12 2v4M8 12h.01M16 12h.01M9 16h6"/>',goal:'<path d="m12 2 3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1z"/>',trap:'<path d="m12 3 10 18H2zM12 9v5M12 17h.01"/>'
};
function icon(name) { return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name]}</svg>`; }
function hydrateIcons(root = document) { root.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon)); }
hydrateIcons();

let activeLab = 'language', method = 'grpo', language = createLanguageState(), selectedResponse = 0;
let requestedLab='language',labUpdate=Promise.resolve();
let config = { ...DEFAULT_CONFIG }, batch = prepareBatch(language, method, config), phase = 0, viewedStage = 0;
let languageController = null, gridTimer = null, toastTimer = null;
let detailOrigin = null;
let graphEpoch = 0,previewGradient=false,updateSequence=null;
const graphMotion=new SvgMotion($('#computation-graph')),chartMotion=new SvgMotion($('#training-chart'));
let grid = createGrid(), selectedCell = grid.start;
let gridConfig = { method: 'q', epsilon: 0.2, alpha: 0.3, gamma: 0.95, stepReward: -0.04, horizon: 80 };
let atlasFamily = 'all', atlasId = 'ppo';

function toast(message) {
  clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false;
  toastTimer = setTimeout(() => $('#toast').hidden = true, 3800);
}
function showDialog(title, body, kicker = 'A CLOSER LOOK') {
  openDetail($('#detail-dialog'),()=>{
    $('#dialog-title').textContent = title; $('#dialog-body').innerHTML = body; $('#dialog-kicker').textContent = kicker;
  },detailOrigin);
}
function showNamedDialog(name) {
  if (name === 'reward') { showRewards(); return; }
  if (name === 'tokens') { showTokenDetail(); return; }
  const [title, body] = DIALOGS[name]; showDialog(title, body);
}
document.addEventListener('click',event=>{const source=event.target.closest('button,.g-interactive,[role="button"]');detailOrigin=source?source.getBoundingClientRect():null;},true);
$('#dialog-close').addEventListener('click', () => closeDetail($('#detail-dialog')));
$('#detail-dialog').addEventListener('cancel',event=>{event.preventDefault();closeDetail(event.currentTarget);});
$('#detail-dialog').addEventListener('click', event => { if (event.target === event.currentTarget && (event.clientX < event.currentTarget.getBoundingClientRect().left || event.clientX > event.currentTarget.getBoundingClientRect().right || event.clientY < event.currentTarget.getBoundingClientRect().top || event.clientY > event.currentTarget.getBoundingClientRect().bottom)) closeDetail(event.currentTarget); });

function setLab(lab) {
  if(requestedLab===lab)return labUpdate;
  requestedLab=lab;
  stopLanguage(); stopGrid();
  labUpdate=switchSurface(()=>{
  if(requestedLab!==lab)return;
  activeLab = lab;
  for (const key of ['language', 'grid']) {
    $(`#${key}-lab`).hidden = key !== lab;
    $(`#tab-${key}`).classList.toggle('active', key === lab);
    $(`#tab-${key}`).setAttribute('aria-selected', String(key === lab));
  }
  if (lab === 'grid') renderGrid(); else renderLanguage({animate:false});
  $('.graph-context').textContent = lab === 'grid' ? '5×5 确定性网格 · 4 个动作' : '四个候选回答 · 可训练的 softmax 策略';
  renderChart({animate:false});
  syncLabIndicator();
  });
  return labUpdate;
}
function syncLabIndicator(){const nav=$('.workspace-nav'),selected=$(`#tab-${activeLab}`),a=nav.getBoundingClientRect(),b=selected.getBoundingClientRect();nav.style.setProperty('--tab-left',b.left-a.left+'px');nav.style.setProperty('--tab-width',b.width+'px');}
$('#tab-language').addEventListener('click', () => setLab('language'));
$('#tab-grid').addEventListener('click', () => setLab('grid'));
$$('.workspace-tab').forEach(el=>el.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();const next=requestedLab==='grid'?'language':'grid';setLab(next).then(()=>{if(requestedLab===next)$(`#tab-${next}`).focus();});}}));

function stopLanguage({settle=true}={}) {
  languageController?.abort();languageController=null;
  if(settle)graphMotion.settle();
  $('#llm-auto').innerHTML = '连续训练';
  $('#llm-auto').classList.remove('running');
  $('#llm-step').disabled=false;$('#llm-round').disabled=false;
  $('#computation-graph').removeAttribute('aria-busy');
  $$('.stage.running').forEach(node=>node.classList.remove('running'));
  $('.model-badge').classList.remove('is-playing');
}
function resetLanguage(taskKey = language.taskKey, scores = null) {
  stopLanguage({settle:false}); language = createLanguageState(taskKey);
  if (scores) language.scores = scores;
  selectedResponse = 0; phase = 0; viewedStage = 0;
  collectBatch();
  renderLanguage(); renderChart();
}
function setMethod(next) {
  if (method === next) return;
  method = next; resetLanguage(); syncParameters();
}
$$('[data-method]').forEach(el => el.addEventListener('click', () => setMethod(el.dataset.method)));
$('#task-select').addEventListener('change', event => resetLanguage(event.target.value));
$('#llm-reset').addEventListener('click', () => { resetLanguage(); toast('策略、训练轨迹与随机种子已重置。'); });
async function playStage(stage,signal){
  phase=stage;viewedStage=stage;renderLanguage();
  if(!await waitForMotion(90,signal))return false;
  return traceFlow($('#computation-graph'),stage,signal);
}
function collectBatch(){
  batch=prepareBatch(language,method,config);graphEpoch=0;previewGradient=false;updateSequence=null;
}
async function playUpdateAtom(signal){
  if(!updateSequence||updateComplete(updateSequence))updateSequence=createUpdateSequence(language,batch);
  updateSequence.cursor++;
  const frame=currentUpdateFrame(updateSequence);
  phase=3;viewedStage=3;previewGradient=true;
  if(updateComplete(updateSequence)){
    optimizeEpoch(language,batch);graphEpoch=batch.epochDetails.length-1;renderChart();
  }
  renderLanguage();
  let flow=Promise.resolve(true);
  if(frame.kind==='sample')flow=traceFlow($('#computation-graph'),3,signal,{direction:'backward',duration:340,keys:['ratio','objective','reduce'].map(part=>`flow:sample:${frame.sample}:${part}`)});
  if(frame.kind==='parameter'||frame.kind==='probability')flow=traceFlow($('#computation-graph'),4,signal,{duration:340,keys:[`flow:${frame.kind==='parameter'?'parameter':'probability'}:${frame.parameter}`]});
  const [,completed]=await Promise.all([graphMotion.pending,flow]);
  if(signal.aborted||!completed)return false;
  return waitForMotion(160,signal);
}
async function advanceLanguage(signal){
  if(phase===4){collectBatch();return playStage(0,signal);}
  if(phase<2)return playStage(phase+1,signal);
  if(!batch.optimized)return playUpdateAtom(signal);
  previewGradient=false;updateSequence=null;
  return playStage(4,signal);
}
async function runLanguageRound(signal){
  if(batch.optimized){if(phase===3){previewGradient=false;updateSequence=null;return playStage(4,signal);}collectBatch();phase=0;}
  if(phase<3){
    for(let stage=phase;stage<3;stage++)if(!await playStage(stage,signal))return false;
  }
  while(!batch.optimized)if(!await playUpdateAtom(signal))return false;
  previewGradient=false;updateSequence=null;
  return playStage(4,signal);
}
async function startLanguagePlayback(operation){
  stopLanguage({settle:false});
  const controller=new AbortController();languageController=controller;
  $('#llm-auto').innerHTML=`${icon('pause')} 暂停`;$('#llm-auto').classList.add('running');
  $('#llm-step').disabled=true;$('#llm-round').disabled=true;$('#computation-graph').setAttribute('aria-busy','true');
  await graphMotion.pending;
  if(!controller.signal.aborted)await operation(controller.signal);
  if(languageController===controller){stopLanguage();renderStages();}
}
$('#llm-step').addEventListener('click',()=>startLanguagePlayback(advanceLanguage));
$('#llm-round').addEventListener('click',()=>startLanguagePlayback(runLanguageRound));
$('#llm-auto').addEventListener('click',()=>{
  if(languageController){stopLanguage();renderStages();return;}
  startLanguagePlayback(async signal=>{while(!signal.aborted){if(!await runLanguageRound(signal))return;if(!await waitForMotion(300,signal))return;}});
});
$('#llm-fast').addEventListener('click', () => {
  stopLanguage();updateSequence=null;
  for (let i = 0; i < 40; i++) { const next = i === 0 && !batch.optimized ? batch : prepareBatch(language, method, config); optimizeBatch(language, next); batch = next; }
  phase = 4; viewedStage = 4; graphEpoch = batch.epochDetails.length - 1;previewGradient=false; renderLanguage(); renderChart(); toast(`完成 ${method.toUpperCase()} 40 轮更新。`);
});

function syncParameters() {
  $$('[data-method]').forEach(el => { el.classList.toggle('selected', el.dataset.method === method); el.setAttribute('aria-pressed', String(el.dataset.method === method)); });
  const online = method === 'grpo' || method === 'ppo';
  $('#param-group').disabled = !online; $('#param-group').parentElement.classList.toggle('parameter-disabled', !online);
  $('#clip-control').hidden = !online; $('#beta-control').hidden = !online; $('#dpo-beta-control').hidden = method !== 'dpo';
  for (const key of ['group','lr','clip','beta','dpoBeta']) {
    $(`#param-${key}`).value = config[key];
    $(`#out-${key}`).textContent = key === 'group' ? config[key] : f(config[key], key === 'dpoBeta' ? 1 : 2);
  }
  const picker=$('.method-picker'),selected=$(`[data-method="${method}"]`);
  picker.style.setProperty('--method-left',selected.offsetLeft+'px');picker.style.setProperty('--method-width',selected.offsetWidth+'px');
}
for (const key of ['group','lr','clip','beta','dpoBeta']) $(`#param-${key}`).addEventListener('input', event => {
  stopLanguage();
  config[key] = event.target.valueAsNumber;
  $(`#out-${key}`).textContent = key === 'group' ? config[key] : f(config[key], key === 'dpoBeta' ? 1 : 2);
  collectBatch(); phase = 0; viewedStage = 0;
  renderLanguage(); if (key === 'clip') renderClip();
});

function renderStages() {
  if(!$('#stage-track').children.length)$('#stage-track').innerHTML=STAGES[method].map((_,i)=>`<button class="stage" data-stage="${i}"><span class="stage-index"></span><span class="stage-label"><span></span><small></small></span></button>`).join('');
  STAGES[method].forEach(([label,english],i)=>{const node=$(`#stage-track [data-stage="${i}"]`);node.classList.toggle('active',i===viewedStage);node.classList.toggle('complete',i<phase);node.classList.toggle('running',Boolean(languageController)&&i===phase);node.setAttribute('aria-label',`查看第 ${i+1} 阶段：${label}`);node.setAttribute('aria-pressed',String(i===viewedStage));node.querySelector('.stage-index').textContent=i<phase?'✓':String(i+1).padStart(2,'0');node.querySelector('.stage-label>span').textContent=label;node.querySelector('small').textContent=english;});
  $('.model-badge').classList.toggle('is-playing',Boolean(languageController));
  $('.model-badge').innerHTML=phase===3?`epoch ${updateSequence?updateSequence.epoch:batch.epochDetails.length} / ${method==='ppo'||method==='grpo'?batch.config.epochs:1}`:`四分类策略 ${typeset(String.raw`\pi_\theta`)}`;
  $('#llm-round').innerHTML=`${icon('play')} ${phase===3&&!batch.optimized?'继续本轮':'更新一轮'}`;
}

function renderLanguage({animate=true}={}) {
  const task = TASKS[language.taskKey], online = method === 'grpo' || method === 'ppo';
  const counts = task.responses.map((_, i) => batch.ids.filter(id => id === i).length);
  $('#task-prompt').textContent = task.prompt;
  $('#experiment-intro').textContent = {grpo:'对同一问题采样多条回答，用组内相对奖励估计优势，再更新策略。',ppo:'固定采样时的策略与价值估计，用裁剪目标更新 Actor，并拟合 Critic。',dpo:'给定同一问题的偏好对，优化它们相对参考策略的对数概率差。',sft:'给定示范回答，最小化它在当前策略下的负对数似然。'}[method];
  $('#rollout-heading').textContent = online ? '采样的回答' : method === 'dpo' ? '一对偏好回答' : '给定的示范回答';
  $('#batch-badge').textContent = online ? `G = ${batch.ids.length}` : method === 'dpo' ? 'chosen / rejected' : 'demonstration';
  $('#rollout-description').textContent = online ? '点击回答看细节；角标是本轮采样次数。' : method === 'dpo' ? '选择一条偏好回答和一条拒绝回答。' : '选择希望策略学习模仿的那条回答。';
  $('#response-list').innerHTML = task.responses.map((response, i) => {
    const editable = language.taskKey === 'preference' && online;
    return `<article class="response ${i === selectedResponse ? 'selected' : ''}" role="button" tabindex="0" data-response="${i}" aria-label="查看回答 ${responseName(i)}：${escapeHTML(textOf(response))}" aria-pressed="${i === selectedResponse}"><span class="response-id">${responseName(i)}</span><span class="response-text">${escapeHTML(textOf(response))}<small class="response-note">${response.note}</small></span>${online ? `<span class="response-score ${language.scores[i] < .5 ? 'bad' : ''}" title="${editable ? '修改人工分数' : '原始验证奖励'}">${editable ? `<input type="number" min="0" max="1" step="0.1" value="${language.scores[i]}" data-score="${i}" aria-label="回答 ${responseName(i)} 的人工评分">` : f(language.scores[i], 2)}</span>${counts[i] ? `<span class="response-count" title="采样 ${counts[i]} 次">${counts[i]}</span>` : ''}` : `<span class="response-choose"><button data-winner="${i}" class="${language.winner === i ? 'is-winner' : ''}" aria-pressed="${language.winner === i}">${method === 'sft' ? '示范' : '偏好'}</button>${method === 'dpo' ? `<button data-loser="${i}" class="${language.loser === i ? 'is-loser' : ''}" aria-pressed="${language.loser === i}">拒绝</button>` : ''}</span>`}</article>`;
  }).join('');
  $('#batch-strip').innerHTML = online ? `<span>本轮序列</span>${batch.ids.map((id, k) => `<span class="sample-chip ${batch.rewards[k] < .5 ? 'negative' : ''}" title="第 ${k + 1} 次采样，奖励 ${batch.rewards[k]}">${responseName(id)}</span>`).join('')}` : method === 'dpo' ? `<span>固定偏好对：</span><span class="sample-chip">${responseName(batch.winner)} · chosen</span><span class="sample-chip negative">${responseName(batch.loser)} · rejected</span><span>不在线采样</span>` : `<span>固定示范：</span><span class="sample-chip">${responseName(batch.winner)}</span><span>不使用奖励</span>`;
  renderStages(); renderCalculation(); renderPolicy(); renderUpdateDetail(animate); renderGraph({animate});
  const status = ['数据已准备，策略尚未更新','已检查奖励 / 对数概率','已构造学习信号，等待参数更新','本轮参数更新已执行','本轮完成；下一步准备新数据'][phase];
  $('#llm-status').textContent = `第 ${language.rounds + (batch.optimized ? 0 : 1)} 批 · ${phase===3&&updateSequence?`epoch ${updateSequence.epoch} · 计算 ${updateSequence.cursor+1} / ${updateSequence.steps.length}`:status}`;
}

function calcValue(label, value, negative = false) { return `<div class="calc-value"><span>${label}</span><strong class="${negative ? 'negative' : ''}">${value}</strong></div>`; }
function renderCalculation() {
  const id=selectedResponse,name=responseName(id),task=TASKS[language.taskKey],p=softmax(language.theta);
  const old=batch.old[id],reward=language.scores[id],count=batch.ids.filter(value=>value===id).length;
  const online=method==='grpo'||method==='ppo',frame=updateSequence?currentUpdateFrame(updateSequence):null;
  let html='',heading=STAGES[method][viewedStage][0];
  if(viewedStage===0){
    heading=online?'采样分布':method==='dpo'?'固定偏好对':'示范回答';
    html=`<p class="formula-caption">${online?'按策略概率有放回采样':'从训练数据读取回答'}</p>${displayEquation(EQ.softmax)}<div class="calculation-values">${calcValue(`采样策略 ${typeset(String.raw`\pi_{\rm old}`)}`,percent(old))}${calcValue(`当前策略 ${typeset(String.raw`\pi_\theta`)}`,percent(p[id]))}${calcValue(`参考策略 ${typeset(String.raw`\pi_{\rm ref}`)}`,percent(language.reference[id]))}${calcValue(online?'回答 '+name+' 的采样次数':'数据角色',online?`${count} / ${batch.ids.length}`:id===batch.winner?'示范 / 偏好':id===batch.loser&&method==='dpo'?'拒绝':'未使用')}</div><p class="calc-note">${online?'每次采样独立进行，同一回答可多次出现。采样策略在本批的多个 epoch 中保持固定。':'在候选回答中选择示范，或选择偏好与拒绝回答。'}</p>`;
  }else if(viewedStage===1){
    if(online){
      heading=method==='ppo'?'奖励与 KL 整形':'验证器与奖励';
      const penalty=batch.config.beta*Math.log(old/language.reference[id]);
      html=`<p class="formula-caption">回答 ${name} · ${task.kind}</p>${displayEquation(method==='ppo'?EQ.rewardShaping:language.taskKey==='code'?EQ.testReward:language.taskKey==='math'?EQ.mathReward:EQ.humanReward)}<div class="calculation-values">${calcValue(`原始奖励 ${typeset('R_i')}`,f(reward))}${calcValue(`组内均值 ${typeset(String.raw`\overline R`)}`,f(batch.mean))}${calcValue(method==='ppo'?'KL 整形项':'组内样本数',method==='ppo'?signed(penalty):batch.ids.length)}${calcValue(method==='ppo'?`整形奖励 ${typeset(String.raw`\widetilde R_i`)}`:'本回答采样次数',method==='ppo'?f(reward-penalty):count)}</div><p class="calc-note">${task.explanation}</p><button class="subtle-link" data-dialog="reward">查看评分过程 ↗</button>`;
    }else{
      heading='序列对数概率';
      html=`<p class="formula-caption">回答 ${name} · 使用自然对数</p>${displayEquation(EQ.sequenceLogprob,true)}<div class="calculation-values">${calcValue(typeset(String.raw`\log\pi_\theta(y\mid x)`),f(Math.log(p[id])))}${calcValue(typeset(String.raw`\log\pi_{\rm ref}(y\mid x)`),f(Math.log(language.reference[id])))}${calcValue('相对对数概率差',signed(Math.log(p[id]/language.reference[id])))}${calcValue('回答概率',percent(p[id]))}</div><p class="calc-note">每个 token 的条件对数概率相加，得到整条回答的对数概率。点击“Token 概率链”查看各项。</p>`;
    }
  }else if(viewedStage===2){
    if(method==='grpo'){
      heading='组内相对优势';
      const advantage=batch.sd===0?0:(reward-batch.mean)/batch.sd;
      html=`<p class="formula-caption">同题回答，用组内统计作基线</p>${displayEquation(EQ.groupAdvantage)}<div class="calculation-values">${calcValue(`奖励 ${typeset('R_i')}`,f(reward))}${calcValue(`组均值 ${typeset(String.raw`\overline R`)}`,f(batch.mean))}${calcValue(`标准差 ${typeset(String.raw`\sigma_R`)}`,f(batch.sd))}${calcValue(`优势 ${typeset(String.raw`\hat A_i`)}`,signed(advantage),advantage<0)}</div><div class="number-walkthrough">${typeset(batch.sd===0?String.raw`\sigma_R=0\quad\Longrightarrow\quad\hat A_i=0`:String.raw`\hat A_{${name}}=\frac{${f(reward)}-${f(batch.mean)}}{${f(batch.sd)}}\approx${signed(advantage)}`,true)}</div><p class="calc-note">${count?`回答 ${name} 出现 ${count} 次，每次都计入目标平均。`:`回答 ${name} 本批采样次数为零。`} ${advantage>0?'高于组均值，优势为正。':advantage<0?'低于组均值，优势为负。':'奖励相同，组内优势为零。'}</p>`;
    }else if(method==='ppo'){
      heading='价值基线与优势';
      const shaped=reward-batch.config.beta*Math.log(old/language.reference[id]),advantage=shaped-batch.oldValue;
      html=`<p class="formula-caption">整形后的单步回报减去旧价值预测</p>${displayEquation(EQ.ppoAdvantage)}<div class="calculation-values">${calcValue(`整形奖励 ${typeset(String.raw`\widetilde R_i`)}`,f(shaped))}${calcValue(`旧价值 ${typeset(String.raw`V_{\rm old}`)}`,f(batch.oldValue))}${calcValue(`优势 ${typeset(String.raw`\hat A_i`)}`,signed(advantage),advantage<0)}${calcValue('当前 Critic',f(language.value))}</div><div class="number-walkthrough">${typeset(String.raw`\hat A_{${name}}=${f(shaped)}-${f(batch.oldValue)}\approx${signed(advantage)}`,true)}</div><p class="calc-note">旧价值在本批更新中固定，Critic 向这批整形回报的平均值拟合。多步任务通过下方的 GAE 递推计算优势。</p>`;
    }else if(method==='dpo'){
      heading='相对偏好差';
      const d=languageObjective(language.theta,language.reference,batch),w=batch.winner,l=batch.loser;
      html=`<p class="formula-caption">偏好 ${responseName(w)} · 拒绝 ${responseName(l)}</p>${displayEquation(EQ.preferenceMargin)}<div class="calculation-values">${calcValue('偏好回答的对数比',signed(Math.log(p[w]/language.reference[w])))}${calcValue('拒绝回答的对数比',signed(Math.log(p[l]/language.reference[l])))}${calcValue(`偏好差 ${typeset('m')}`,signed(d.z))}${calcValue(`偏好概率 ${typeset(String.raw`\sigma(m)`)}`,percent(d.preference))}</div><div class="number-walkthrough">${typeset(String.raw`\mathcal L=-\log\sigma(${f(d.z)})\approx${f(-d.objective)}`,true)}</div><p class="calc-note">偏好差增大，模型更倾向于这对回答中的偏好项，损失随之下降。</p>`;
    }else{
      heading='示范的负对数似然';
      const w=batch.winner;
      html=`<p class="formula-caption">示范回答 ${responseName(w)}</p>${displayEquation(EQ.sftLoss)}<div class="calculation-values">${calcValue(`示范概率 ${typeset('p^*')}`,percent(p[w]))}${calcValue('负对数似然',f(-Math.log(p[w])))}${calcValue(`学习率 ${typeset(String.raw`\eta`)}`,f(config.lr,2))}${calcValue('示范数量','1')}</div><div class="number-walkthrough">${typeset(String.raw`\mathcal L=-\log(${f(p[w],4)})\approx${f(-Math.log(p[w]))}`,true)}</div><p class="calc-note">示范越容易被生成，负对数似然越低。对数概率的梯度会提高示范坐标，并按当前概率降低其他坐标。</p>`;
    }
  }else if(viewedStage===3){
    heading='梯度与参数增量';
    const epoch=batch.epochDetails[graphEpoch],theta=updateSequence?updateSequence.thetaBefore:epoch?epoch.thetaBefore:language.theta;
    const d=updateSequence?updateSequence.objective:languageObjective(theta,language.reference,batch),input=softmax(theta);
    const gradient=frame?frame.gradient[id]:d.gradient[id],thetaAfter=frame?frame.theta[id]:epoch?epoch.thetaAfter[id]:theta[id];
    html=`<p class="formula-caption">${frame?frame.title:online?'固定样本与优势，每个 epoch 重算梯度':'固定训练数据，计算目标梯度'}</p>${displayEquation(online?method==='grpo'?EQ.grpoObjective:EQ.ppoObjective:method==='dpo'?EQ.dpoGradient:EQ.softmaxGradient,online)}<div class="calculation-values">${calcValue(online?`概率比 ${typeset(String.raw`\rho_i`)}`:'当前损失',online?f(input[id]/old):f(-d.objective))}${calcValue(online?'进入裁剪区的样本':'示范 / 偏好回答',online?`${d.clipped} / ${batch.ids.length}`:responseName(batch.winner))}${calcValue(`参数 ${name} 的${frame&&frame.kind==='sample'?'累积':''}梯度`,signed(gradient),gradient<0)}${calcValue('已计算的参数增量',signed(thetaAfter-theta[id]))}</div><div class="number-walkthrough">${typeset(String.raw`\Delta\theta_{${name}}=\eta g_{${name}},\qquad\theta^{\rm new}_{${name}}=\theta_{${name}}+\Delta\theta_{${name}}`,true)}</div><p class="calc-note">${frame?frame.note:'单步逐条累加梯度，逐项计算新参数，再经 softmax 得到新概率。'} ${online?'本批样本与优势在各 epoch 之间保持固定。':''}</p>`;
  }else{
    heading='更新前后';
    const now=languageMetrics(language),before=batch.before;
    html=`<p class="formula-caption">本批更新前 → 当前策略</p>${displayEquation(EQ.expectedReward)}<div class="calculation-values">${calcValue('更新前期望奖励',f(before.reward))}${calcValue('当前期望奖励',f(now.reward))}${calcValue('当前 KL · nat',f(now.kl))}${calcValue('当前熵 · nat',f(now.entropy))}</div><div class="number-walkthrough">${typeset(String.raw`\begin{aligned}\Delta\mathbb E[R]&=${signed(now.reward-before.reward)}\\p_{${name}}&:\;${f(before.p[id])}\;\longrightarrow\;${f(now.p[id])}\end{aligned}`,true)}</div><p class="calc-note">${online?'下一轮使用新策略重新采样，刷新旧策略快照；参考策略保持固定。':'损失衡量对训练数据的拟合程度；期望奖励按验证分数加权。换一条示范或偏好，可比较两个指标怎样变化。'}</p>`;
  }
  $('#calculation-heading').textContent=heading;$('#calculation-content').innerHTML=html;
  $('#inspector-step').textContent=String(viewedStage+1).padStart(2,'0');
}

function renderPolicy() {
  const metrics = languageMetrics(language);
  $('#metric-reward').textContent = f(metrics.reward); $('#metric-kl').textContent = f(metrics.kl);
  $('#graph-round').textContent = `第 ${language.rounds} 轮`;
  $('#policy-insight').innerHTML = typeset(String.raw`H(\pi_\theta)=${f(metrics.entropy)}\;\mathrm{nat}`);
}

function renderUpdateDetail(animate){
  const frame=updateSequence?currentUpdateFrame(updateSequence):null,panel=$('#update-detail');
  panel.dataset.updateKind=frame?frame.kind:'ready';
  panel.dataset.updateCursor=frame?String(updateSequence.cursor):'-1';
  panel.dataset.updateEpoch=updateSequence?String(updateSequence.epoch):'0';
  $('#update-title').textContent=frame?frame.title:'逐步拆解一次更新';
  $('#update-note').textContent=frame?frame.note:'样本梯度 → 参数增量 → 指数权重 → 概率归一化';
  $('#update-position').textContent=frame?`epoch ${updateSequence.epoch} · ${updateSequence.cursor+1} / ${updateSequence.steps.length}`:'单步推进 · 可随时暂停';
  const tex=frame?frame.tex:EQ.parameterUpdate,formula=$('#update-equation');
  if(formula.dataset.tex!==tex){
    formula.dataset.tex=tex;formula.innerHTML=typeset(tex,true);
    formula.getAnimations().forEach(animation=>animation.cancel());
    if(animate&&!motionPreference.matches)formula.animate([{opacity:.2,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'ease-out'});
  }
  $('#update-progress').innerHTML=frame?updateSequence.steps.map((step,i)=>`<i class="${i<updateSequence.cursor?'done':i===updateSequence.cursor?'current':''}" title="${step.title}"></i>`).join(''):'';
}
function renderGraph({animate=true}={}) {
  $('#graph-epoch-wrap').hidden = batch.epochDetails.length < 2||Boolean(updateSequence);
  $$('#graph-epoch option').forEach((option,i)=>option.hidden=i>=batch.epochDetails.length);
  $('#graph-epoch').value = graphEpoch;
  const pending=graphMotion.render(renderComputationGraph(language,batch,selectedResponse,graphEpoch,{preview:previewGradient,operation:updateSequence}),{animate});
  const frame=updateSequence?currentUpdateFrame(updateSequence):null,svg=$('#computation-graph svg');
  svg.dataset.updateKind=frame?frame.kind:'ready';
  svg.querySelectorAll('[data-atom-active]').forEach(node=>node.removeAttribute('data-atom-active'));
  let selectors=[];
  if(frame?.kind==='sample')selectors=[`.g-sample[data-sample="${frame.sample}"]`];
  if(frame?.kind==='regularize')selectors=['[data-motion-key="objective-reduction"]'];
  if(frame?.kind==='critic')selectors=['[data-motion-key="statistics"]'];
  if(frame?.kind==='parameter')selectors=[`[data-motion-key="gradient:${frame.parameter}"]`,`[data-motion-key="theta:${frame.parameter}"]`];
  if(frame?.kind==='weight'||frame?.kind==='probability')selectors=[`[data-motion-key="probability:${frame.parameter}"]`];
  for(const selector of selectors)svg.querySelectorAll(selector).forEach(node=>node.setAttribute('data-atom-active',''));
  return pending;
}
$('#graph-epoch').addEventListener('change',event=>{stopLanguage();graphEpoch=Number(event.target.value);previewGradient=false;updateSequence=null;renderUpdateDetail(false);renderGraph();});
$('#graph-fit').addEventListener('click',()=>$('#computation-viewport').classList.add('fit'));
$('#graph-expand').addEventListener('click',()=>$('#computation-viewport').classList.remove('fit'));
$('#computation-graph').addEventListener('pointerover',event=>{
  const sample=event.target.closest('[data-sample]');if(!sample){$$('#computation-graph .trace-muted').forEach(el=>el.classList.remove('trace-muted'));return;}
  const k=sample.dataset.sample;
  $$('#computation-graph [data-sample]').forEach(el=>el.classList.toggle('trace-muted',el.dataset.sample!==k));
  $$('#computation-graph [data-graph-response]').forEach(el=>el.classList.toggle('trace-muted',Number(el.dataset.graphResponse)!==batch.ids[Number(k)]));
});
$('#computation-graph').addEventListener('pointerleave',()=>$$('#computation-graph .trace-muted').forEach(el=>el.classList.remove('trace-muted')));

function showGradient(j) {
  const theta=updateSequence?updateSequence.thetaBefore:batch.epochDetails.length&&!previewGradient?batch.epochDetails[graphEpoch].thetaBefore:language.theta;
  const p=softmax(theta),d=languageObjective(theta,language.reference,batch),K=kl(p,language.reference),name=responseName(j);
  const contribution=method==='grpo'?-batch.config.beta*p[j]*(Math.log(p[j]/language.reference[j])-K):0;
  const rows=d.samples.map((sample,k)=>{
    const coefficient=sample.clipped?0:sample.ratio*sample.advantage,logGradient=Number(sample.id===j)-p[j];
    return `<tr><td>${k+1} · ${responseName(sample.id)}</td><td>${signed(sample.advantage)}</td><td>${f(sample.ratio)}</td><td>${sample.clipped?'0 · 裁剪':f(coefficient)}</td><td>${signed(logGradient)}</td><td>${signed(coefficient*logGradient/batch.ids.length,5)}</td></tr>`;
  }).join('');
  const online=method==='grpo'||method==='ppo';
  const body=online?`<p>每条样本给出一个方向，乘以概率比和优势，再按组大小取平均。以下数字来自同一 epoch 的更新前参数。</p>${displayEquation(EQ.gradientParts,true)}<p>${typeset(String.raw`c_i=\rho_i\hat A_i`)}；进入裁剪平坦区时取零。当前 ${typeset(String.raw`p_{${name}}=${f(p[j],5)}`)}。</p><table class="dialog-table"><thead><tr><th>样本</th><th>${typeset(String.raw`\hat A_i`)}</th><th>${typeset(String.raw`\rho_i`)}</th><th>${typeset('c_i')}</th><th>${typeset(String.raw`\partial\log p_{y_i}/\partial\theta_j`)}</th><th>平均梯度贡献</th></tr></thead><tbody>${rows}</tbody></table>`:`<p>${method==='dpo'?'偏好对的归一化项相消，梯度只沿偏好和拒绝坐标移动。':'对示范的对数概率求导：示范位置取一，再减去当前概率。'}</p>${displayEquation(method==='dpo'?EQ.dpoGradient:EQ.softmaxGradient)}`;
  showDialog(`回答 ${name} 对应参数的梯度`,`${body}<div class="number-walkthrough">${typeset(String.raw`\begin{aligned}g_{${name}}&=${f(d.gradient[j]-contribution,6)}+(${f(contribution,6)})=${f(d.gradient[j],6)}\\\Delta\theta_{${name}}&=${f(batch.config.lr,2)}\times(${f(d.gradient[j],6)})=${f(batch.config.lr*d.gradient[j],6)}\end{aligned}`,true)}</div><p>${method==='grpo'?'第二项为 KL 的梯度贡献。':method==='ppo'?'KL 已计入采样时的奖励整形，Actor 在此优化裁剪目标。':'学习率缩放梯度，得到参数增量。'}</p>`,'GRADIENT DECOMPOSITION');
}

function showTokenDetail() {
  const task=TASKS[language.taskKey],p=softmax(language.theta),trace=tokenTrace(task,selectedResponse,p);
  const sum=trace.reduce((value,token)=>value+token.logProbability,0);
  showDialog(`回答 ${responseName(selectedResponse)} 的概率链`,`<p>按候选回答的共同前缀展开路径。每个分支的条件概率，等于该分支的概率质量除以前缀的总概率质量；只有一条可选路径时，条件概率为一。</p><div class="token-chain">${trace.map(token=>`<span class="token"><strong>${escapeHTML(token.token)}</strong><small>${typeset(`p=${f(token.probability,4)}`)}</small><em>${typeset(String.raw`\log p=${f(token.logProbability,4)}`)}</em></span>`).join('')}</div>${displayEquation(EQ.sequenceProduct)}${displayEquation(EQ.sequenceLogprob)}<div class="number-walkthrough">${typeset(String.raw`\begin{aligned}\sum_t\log p_t&=${f(sum,6)}=\log\pi_\theta(y\mid x)\\\prod_t p_t&=${f(Math.exp(sum),6)}=\pi_\theta(y\mid x)\end{aligned}`,true)}</div>`,'TOKEN PROBABILITY');
}
$('#token-detail').addEventListener('click', showTokenDetail);

function showRewards() {
  const task = TASKS[language.taskKey];
  let evidence;
  if (language.taskKey === 'math') evidence = `<table class="dialog-table"><thead><tr><th>回答</th><th>提取的答案</th><th>等于 17×3？</th><th>奖励</th></tr></thead><tbody>${task.responses.map((r, i) => `<tr><td>${responseName(i)}</td><td>${r.answer}</td><td>${r.answer === 17 * 3 ? '通过' : '未通过'}</td><td>${language.scores[i]}</td></tr>`).join('')}</tbody></table>`;
  else if (language.taskKey === 'code') evidence = `<table class="dialog-table"><thead><tr><th>回答</th>${[-3,0,2,5].map(n => `<th>n=${n}</th>`).join('')}<th>奖励</th></tr></thead><tbody>${task.responses.map((r, i) => `<tr><td>${responseName(i)}</td>${[-3,0,2,5].map(n => `<td>${r.fn(n)} ${r.fn(n) === n*n ? '✓' : '×'}</td>`).join('')}<td>${language.scores[i]}</td></tr>`).join('')}</tbody></table>`;
  else evidence = `<table class="dialog-table"><thead><tr><th>回答</th><th>人工评分</th><th>说明</th></tr></thead><tbody>${task.responses.map((r, i) => `<tr><td>${responseName(i)}</td><td>${language.scores[i]}</td><td>${r.note}</td></tr>`).join('')}</tbody></table>`;
  showDialog('奖励的来源，可逐项检查', `<p>${task.explanation}</p>${evidence}`);
}

function stopGrid() {
  if (gridTimer !== null) clearInterval(gridTimer);
  gridTimer = null; $('#grid-auto').innerHTML = `${icon('play')} 开始探索`; $('#grid-auto').classList.remove('running');
}
function resetGrid() { stopGrid(); grid = createGrid(); selectedCell = grid.start; renderGrid(); renderChart(); }
$('#grid-reset').addEventListener('click', resetGrid);
$$('[data-grid-method]').forEach(el => el.addEventListener('click', () => { if (gridConfig.method !== el.dataset.gridMethod) { gridConfig.method = el.dataset.gridMethod; resetGrid(); } }));
function advanceGrid() { gridStep(grid, gridConfig); selectedCell = grid.last.s; renderGrid(); renderChart(); }
$('#grid-step').addEventListener('click', () => { stopGrid(); advanceGrid(); });
$('#grid-auto').addEventListener('click', () => {
  if (gridTimer !== null) { stopGrid(); return; }
  gridTimer = setInterval(advanceGrid, 150);
  $('#grid-auto').innerHTML = `${icon('pause')} 暂停探索`; $('#grid-auto').classList.add('running');
});
$('#grid-fast').addEventListener('click', () => {
  stopGrid(); const goal = grid.episodes + 100;
  while (grid.episodes < goal) gridStep(grid, gridConfig);
  selectedCell = grid.last.s; renderGrid(); renderChart(); toast('已完成 100 个真实环境交互回合。');
});
$('#show-values').addEventListener('change', renderWorld);
for (const key of ['epsilon','alpha','gamma','stepReward']) $(`#grid-${key}`).addEventListener('input', event => {
  gridConfig[key] = event.target.valueAsNumber; $(`#out-${key}`).textContent = f(gridConfig[key], 2); renderGrid();
});
function renderGrid() {
  if (activeLab === 'grid') $('#experiment-intro').innerHTML = `智能体通过探索或利用选择方向，用一次转移的 TD 目标更新 ${typeset('Q(s,a)')}。`;
  $$('[data-grid-method]').forEach(el => { el.classList.toggle('selected', el.dataset.gridMethod === gridConfig.method); el.setAttribute('aria-pressed', String(el.dataset.gridMethod === gridConfig.method)); });
  renderWorld(); renderGridUpdate();
  $('#grid-episodes').textContent = grid.episodes; $('#grid-steps').textContent = grid.totalSteps;
  $('#grid-recent').innerHTML = `<p class="recent-heading">最近完成的回合</p>${grid.history.length ? grid.history.slice(-5).reverse().map(h => `<div class="episode-row"><span>#${h.episode}</span><span>${h.steps} steps</span><span class="${h.success ? 'success' : ''}">${signed(h.reward,2)} ${h.success ? '✓' : '·'}</span></div>`).join('') : '<p class="episode-empty">回合结束后，<br>这里会记录真实回报。</p>'}`;
  $('#grid-explanation').textContent = gridConfig.method === 'q' ? 'Q-learning 用下一状态最大的 Q 作为目标，与实际采样的下一动作分开。即使行为在探索，也在学习贪心策略。' : 'SARSA 把实际选中的下一动作带入更新。ε 所带来的探索风险也进入价值估计。';
  $('#grid-status').textContent = grid.last ? `s${grid.last.s} → ${ACTIONS[grid.last.action].name} → s${grid.last.next} · 奖励 ${signed(grid.last.reward,2)} · ${grid.last.terminal ? '真正终止，bootstrap=0' : grid.last.truncated ? '80 步截断，保留 bootstrap' : '回合进行中'}` : '起点在左下角，终点在右上角。每个 Q 值从 0 开始。';
}
function renderWorld() {
  const show = $('#show-values').checked;
  $('#world').innerHTML = grid.q.map((values, i) => {
    const agent = i === grid.position, wall = grid.walls.has(i), goal = i === grid.goal, trap = grid.traps.has(i), max = Math.max(...values);
    const best = values.map((x, j) => x === max ? j : -1).filter(j => j !== -1);
    return `<button class="world-cell ${wall ? 'wall' : goal ? 'goal' : trap ? 'trap' : ''} ${agent ? 'agent' : ''} ${selectedCell === i ? 'selected' : ''} ${grid.path.includes(i) ? 'in-path' : ''}" data-cell="${i}" ${wall ? 'disabled' : ''} aria-label="状态 ${i}${agent ? '，智能体当前位置' : ''}${goal ? '，终点' : ''}${trap ? '，陷阱' : ''}，最大 Q ${f(max)}"><small>${i}</small>${agent ? icon('agent') : wall ? '' : goal ? icon('goal') : trap ? icon('trap') : `<strong>${best.length === 1 ? ACTIONS[best[0]].symbol : '·'}</strong>`}${show && !wall ? `<em>${f(max,2)}</em>` : ''}</button>`;
  }).join('');
}
function renderGridUpdate() {
  const last=grid.last;
  if(last){
    const c=last.config;
    $('#grid-update').innerHTML=`<div class="transition-row"><span>${typeset(`s_{${last.s}}`)}</span><b>→ ${ACTIONS[last.action].symbol} →</b><span>${typeset(`r=${signed(last.reward,2)}`)}</span><b>→</b><span>${typeset(`s_{${last.next}}`)}</span></div>${displayEquation(c.method==='q'?EQ.qUpdate:EQ.sarsaUpdate,true)}<div class="calculation-values">${calcValue('更新前 Q',f(last.old))}${calcValue('TD 目标',f(last.target))}${calcValue(`TD 误差 ${typeset(String.raw`\delta`)}`,signed(last.delta),last.delta<0)}${calcValue('更新后 Q',f(last.updated))}</div><div class="number-walkthrough">${typeset(String.raw`\begin{aligned}y&=${f(last.reward)}+${f(c.gamma,2)}\times${f(last.bootstrap)}=${f(last.target)}\\Q^{\rm new}&=${f(last.old)}+${f(c.alpha,2)}\times(${f(last.delta)})\\&=${f(last.updated)}\end{aligned}`,true)}</div><p class="calc-note">${last.terminal?'真正终止，后续回报为零。':c.method==='q'?`取下一状态的最大动作价值 ${typeset(String.raw`\max_{a'}Q(s',a')=${f(last.bootstrap)}`)}。`:`下一动作选中 ${ACTIONS[last.nextAction].symbol}，使用 ${typeset(String.raw`Q(s',a')=${f(last.bootstrap)}`)}。`}</p>${last.truncated?'<span class="terminal-note">到达采样步数上限，保留下一状态估值。</span>':''}`;
  }else $('#grid-update').innerHTML=`<p class="inline-empty">点击“走一步”，查看一次环境转移与 TD 更新。<br>初始 ${typeset('Q(s,a)=0')}。</p>${displayEquation(gridConfig.method==='q'?EQ.qTarget:EQ.sarsaTarget,true)}<p class="calc-note">奖励与下一状态估值共同构成目标。重复更新后，终点附近的价值逐步传回起点。</p>`;
  $('#selected-state').textContent=`s${selectedCell}`;
  const terminal=selectedCell===grid.goal||grid.traps.has(selectedCell);
  $('#selected-state').nextElementSibling.innerHTML=terminal?'终止状态':`行为策略 ${typeset(String.raw`\pi(a\mid s)`)}`;
  const probabilities=epsilonPolicy(grid.q[selectedCell],gridConfig.epsilon),positions=['1 / 2','2 / 3','3 / 2','2 / 1'];
  $('#q-compass').innerHTML=ACTIONS.map((action,i)=>`<div class="q-action" style="grid-area:${positions[i]}"><span>${action.symbol} ${f(grid.q[selectedCell][i])}</span><small>${terminal?'终止':percent(probabilities[i])}</small></div>`).join('')+`<span class="q-center" style="grid-area:2 / 2">s${selectedCell}</span>`;
}

function linePath(points, x, y, key) { return points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(2)},${y(p[key]).toFixed(2)}`).join(' '); }
function renderChart({animate=true}={}) {
  const width = 1100, height = 146, left = 38, right = 42, top = 12, bottom = 24, plotW = width-left-right, plotH = height-top-bottom;
  let points, leftMin, leftMax, rightMax, key, secondaryKey, leftLabel, rightLabel, caption;
  if (activeLab === 'language') {
    const initial = { round:0, reward:language.reference.reduce((s,p,i)=>s+p*language.scores[i],0), kl:0 };
    points = [initial, ...language.history]; leftMin=0; leftMax=1; rightMax=Math.max(.2,Math.ceil(Math.max(...points.map(p=>p.kl))*5)/5);
    key='reward'; secondaryKey='kl'; leftLabel='期望奖励'; rightLabel='KL · nat';
    caption='左轴：期望奖励；右轴：相对固定参考策略的 KL。每个点对应一批完成的更新。';
    $('#chart-title').textContent='训练记录'; $('#chart-counter').textContent=`${language.rounds} rounds · ${language.updates} updates`;
  } else {
    points=grid.history.map((h,i)=>({...h,successRate:mean(grid.history.slice(Math.max(0,i-19),i+1).map(v=>Number(v.success)))}));
    leftMin=Math.min(-1,...points.map(p=>p.reward)); leftMax=Math.max(1,...points.map(p=>p.reward)); rightMax=1;
    key='reward'; secondaryKey='successRate'; leftLabel='回合回报'; rightLabel='近20局成功率';
    caption='左轴：每局累计原始奖励；右轴：最近至多 20 个已完成回合的成功比例。每局最多 80 步。';
    $('#chart-title').textContent='回合回报与成功率'; $('#chart-counter').textContent=`${grid.episodes} episodes · ${grid.totalSteps} steps`;
  }
  const x=i=>left+(points.length<=1?0:i/(points.length-1))*plotW;
  const y=v=>top+(1-(v-leftMin)/(leftMax-leftMin))*plotH;
  const y2=v=>top+(1-v/rightMax)*plotH;
  let svg = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${leftLabel}和${rightLabel}训练曲线">`;
  for(let i=0;i<3;i++){const yy=top+i*plotH/2;svg+=`<line class="chart-grid" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text class="chart-axis" x="${left-9}" y="${yy+3}" text-anchor="end">${f(leftMax-(leftMax-leftMin)*i/2,1)}</text><text class="chart-axis" x="${width-right+9}" y="${yy+3}">${f(rightMax*(1-i/2),1)}</text>`;}
  if(points.length){
    svg+=`<path data-motion-key="chart-primary" class="chart-line" d="${linePath(points,x,y,key)}"/><path data-motion-key="chart-secondary" class="chart-line orange" d="${linePath(points,x,y2,secondaryKey)}"/>`;
    const last=points.length-1;
    svg+=`<circle data-motion-key="chart-primary-end" cx="${x(last)}" cy="${y(points[last][key])}" r="3" fill="#53876e"/><circle data-motion-key="chart-secondary-end" cx="${x(last)}" cy="${y2(points[last][secondaryKey])}" r="3" fill="#c66c43"/>`;
    for(let i=0;i<=Math.min(4,last);i++){const ix=last===0?0:Math.round(i*last/Math.min(4,last));svg+=`<text class="chart-axis" x="${x(ix)}" y="${height-5}" text-anchor="middle">${activeLab==='language'?points[ix].round:points[ix].episode}</text>`;}
    if(points.length>1) svg+=points.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p[key])}" r="5" fill="transparent"><title>${activeLab==='language'?'轮':'回合'} ${activeLab==='language'?p.round:p.episode}：${leftLabel} ${f(p[key])}；${rightLabel} ${f(p[secondaryKey])}</title></circle>`).join('');
  }
  if(points.length<2) svg+=`<text class="chart-empty" x="${width/2}" y="${height/2}">${activeLab==='language'?'开始训练，观察每轮实际计算的奖励与 KL。':'完成一回合后，真实训练轨迹将出现在这里。'}</text>`;
  chartMotion.render(svg+'</svg>',{animate});
  $('#chart-caption').textContent=caption;
  $('#chart-legend').innerHTML=`<span><i></i>${leftLabel} · 左轴</span><span><i class="orange"></i>${rightLabel} · 右轴</span>`;
}

function renderClip() {
  const a=$('#clip-advantage').valueAsNumber,r=$('#clip-ratio').valueAsNumber,epsilon=config.clip;
  const width=480,height=140,left=32,right=16,top=9,bottom=22,plotW=width-left-right,plotH=height-top-bottom;
  const ymax=Math.max(.3,Math.abs(a)*1.7), ymin=-ymax;
  const x=v=>left+(v-.4)/1.2*plotW,y=v=>top+(ymax-v)/(ymax-ymin)*plotH;
  const values=Array.from({length:61},(_,i)=>.4+i*.02),term=clippedSurrogate(r,a,epsilon);
  const path=fn=>values.map((v,i)=>`${i?'L':'M'}${x(v)},${y(fn(v))}`).join(' ');
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="PPO 概率比与裁剪代理目标，优势 ${a}"><rect x="${x(1-epsilon)}" y="${top}" width="${x(1+epsilon)-x(1-epsilon)}" height="${plotH}" fill="#eef4e5"/><path d="M${left} ${y(0)}H${width-right}" stroke="#d9e2ca"/><path d="${path(v=>v*a)}" stroke="#bbc5ab" stroke-width="1.5" stroke-dasharray="4 4" fill="none"/><path d="${path(v=>clippedSurrogate(v,a,epsilon).value)}" stroke="#c66c43" stroke-width="2.2" fill="none"/><circle cx="${x(r)}" cy="${y(term.value)}" r="4" fill="#c66c43" stroke="#fffef9" stroke-width="2"/>`;
  for(const v of [.4,1-epsilon,1,1+epsilon,1.6]) svg+=`<text class="chart-axis" x="${x(v)}" y="${height-5}" text-anchor="middle">${f(v,1)}</text>`;
  svg+=`<text class="chart-axis" x="${left}" y="${top+8}">代理项</text><text class="chart-axis" x="${width-right}" y="${top+8}" text-anchor="end">裁剪半宽 ${f(epsilon,2)}</text></svg>`;
  $('#clip-chart').innerHTML=svg;$('#clip-advantage-out').textContent=signed(a,1);$('#clip-ratio-out').textContent=f(r,2);
  $('#clip-explanation').innerHTML=`${typeset(String.raw`\ell=\min\!\left(${f(r*a,2)},${f(term.clippedRatio*a,2)}\right)=${f(term.value,2)}`,true)}<p>${a===0?'优势为零，该样本贡献为零。':term.active?'进入平坦区，这条样本的代理梯度为零。':'当前点的代理梯度仍在推动更新。'} 虚线为原始项，实线为裁剪目标。</p>`;
}
$('#clip-advantage').addEventListener('input',renderClip);$('#clip-ratio').addEventListener('input',renderClip);
function renderGAE(){
  const gamma=$('#gae-gamma').valueAsNumber,lambda=$('#gae-lambda').valueAsNumber,terminated=$('#gae-terminal').checked;
  const rewards=[0,0,1],values=[.2,.35,.6],result=gae(rewards,values,gamma,lambda,terminated,.6);
  $('#gae-gamma-out').textContent=f(gamma,2);$('#gae-lambda-out').textContent=f(lambda,2);
  $('#gae-trace').innerHTML=result.advantages.map((advantage,t)=>`<div class="gae-cell"><span>${typeset(`t=${t}`)}</span><strong>${typeset(String.raw`\hat A_${t}=${f(advantage)}`)}</strong><small>${typeset(String.raw`\delta_${t}=${f(result.deltas[t])}`)}</small><em>${typeset(String.raw`r_{${t+1}}=${rewards[t]},\ V_${t}=${values[t]}`)}</em></div>`).join('');
  $('#gae-explanation').innerHTML=`${typeset(String.raw`\begin{aligned}\hat A_0&=\delta_0+\gamma\lambda\delta_1+(\gamma\lambda)^2\delta_2\\&\approx${f(result.deltas[0])}+${f(gamma*lambda)}\times${f(result.deltas[1])}+${f((gamma*lambda)**2)}\times${f(result.deltas[2])}\\&\approx${f(result.advantages[0])}\end{aligned}`,true)}<p>${terminated?`真正终止，${typeset('V(s_3)=0')}。`:`时间截断，使用 ${typeset('V(s_3)=0.60')} 补足后续价值。`}</p>`;

}
$('#gae-gamma').addEventListener('input',renderGAE);$('#gae-lambda').addEventListener('input',renderGAE);$('#gae-terminal').addEventListener('change',renderGAE);

function renderAtlas(){
  const algorithms=ALGORITHMS.filter(a=>atlasFamily==='all'||a.family===atlasFamily);
  if(!algorithms.some(a=>a.id===atlasId))atlasId=algorithms[0].id;
  $$('[data-family]').forEach(el=>{el.classList.toggle('selected',el.dataset.family===atlasFamily);el.setAttribute('aria-pressed',String(el.dataset.family===atlasFamily));});
  $('#algorithm-atlas').innerHTML=algorithms.map(a=>`<button class="algorithm-node ${a.id===atlasId?'active':''}" data-algorithm="${a.id}" aria-pressed="${a.id===atlasId}"><span>${a.year}</span><strong>${a.name}</strong><small>${a.label}</small></button>`).join('');
  const a=algorithms.find(a=>a.id===atlasId);
  $('#algorithm-detail').innerHTML=`<div><span class="algorithm-tag">${a.family==='value'?'VALUE-BASED':a.family==='policy'?'POLICY OPTIMIZATION':'POST-TRAINING'}</span><h3>${a.name}<small>${a.year}</small></h3><p>${a.intro}</p><div class="algorithm-facts">${a.tags.map(t=>`<span>${t}</span>`).join('')}</div><div class="algorithm-links">${a.demo?`<button class="subtle-link" data-demo="${a.id}">在实验室运行 <span>↗</span></button>`:''}</div></div><div>${displayEquation(a.formula)}<div class="algorithm-pipeline">${a.pipeline.map((p,i)=>`${i?'<b>→</b>':''}<span>${p}</span>`).join('')}</div><p>${a.detail}</p></div>`;
}
$('#concept-guide').innerHTML=CONCEPTS.map(([q,a],i)=>`<article id="concept-${i}"><h3>${q}</h3><div class="concept-body">${a}</div></article>`).join('');

document.addEventListener('click',async event=>{
  const target=event.target.closest('[data-dialog],[data-term],[data-stage],[data-response],[data-winner],[data-loser],[data-cell],[data-family],[data-algorithm],[data-demo],[data-route],[data-gradient]');
  if(!target)return;
  if(target.dataset.dialog){showNamedDialog(target.dataset.dialog);return;}
  if(target.dataset.term){showDialog(...TERMS[target.dataset.term]);return;}
  if(target.dataset.gradient!==undefined){stopLanguage();showGradient(Number(target.dataset.gradient));return;}
  if(target.dataset.stage!==undefined){
    stopLanguage();
    if(target.dataset.sampleIndex!==undefined)selectedResponse=batch.ids[Number(target.dataset.sampleIndex)];
    if(target.dataset.response!==undefined)selectedResponse=Number(target.dataset.response);
    const inGraph=target.closest('#computation-graph');
    viewedStage=Number(target.dataset.stage);renderStages();renderCalculation();renderGraph();
    if(inGraph)showDialog($('#calculation-heading').textContent,`${$('#calculation-content').innerHTML}${viewedStage===0?'<p><button class="subtle-link" data-dialog="tokens">展开 Token 概率链 ↗</button></p>':''}`,STAGES[method][viewedStage][1]);
    return;
  }
  if(target.dataset.winner!==undefined||target.dataset.loser!==undefined){
    event.stopPropagation();stopLanguage();const id=Number(target.dataset.winner??target.dataset.loser);
    if(target.dataset.winner!==undefined){if(id===language.loser&&method==='dpo')language.loser=language.winner;language.winner=id;}
    else{if(id===language.winner)language.winner=language.loser;language.loser=id;}
    collectBatch();phase=0;viewedStage=2;renderLanguage();return;
  }
  if(target.dataset.response!==undefined){if(event.target.matches('input'))return;selectedResponse=Number(target.dataset.response);renderLanguage();return;}
  if(target.dataset.cell!==undefined){selectedCell=Number(target.dataset.cell);renderWorld();renderGridUpdate();return;}
  if(target.dataset.family){atlasFamily=target.dataset.family;renderAtlas();return;}
  if(target.dataset.algorithm){atlasId=target.dataset.algorithm;renderAtlas();return;}
  if(target.dataset.demo||target.dataset.route){
    const id=target.dataset.demo??target.dataset.route;
    if(id==='q'||id==='sarsa'){await setLab('grid');if(requestedLab!=='grid')return;if(gridConfig.method!==id){gridConfig.method=id;resetGrid();}}
    else{await setLab('language');if(requestedLab!=='language')return;setMethod(id);}
    $('#laboratory').scrollIntoView({behavior:'smooth'});return;
  }
});
document.addEventListener('keydown',event=>{if(event.target.matches('.response')&&(event.key==='Enter'||event.key===' ')){event.preventDefault();selectedResponse=Number(event.target.dataset.response);renderLanguage();$(`[data-response="${selectedResponse}"]`).focus();}});
$('#computation-graph').addEventListener('keydown',event=>{if(event.target.matches('.g-interactive')&&(event.key==='Enter'||event.key===' ')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
document.addEventListener('change',event=>{
  if(!event.target.matches('[data-score]'))return;
  const input=event.target;if(!input.checkValidity()){input.reportValidity();return;}
  const scores=[...language.scores];scores[Number(input.dataset.score)]=input.valueAsNumber;
  resetLanguage(language.taskKey,scores);toast('人工评分已改变。实验从相同初始策略重新开始。');
});
$('#export-trace').addEventListener('click',()=>{
  const data=activeLab==='language'?{experiment:'finite-response-post-training',method,task:language.taskKey,seed:42,config,reference:language.reference,logits:language.theta,probabilities:softmax(language.theta),scores:language.scores,winner:language.winner,loser:language.loser,critic:language.value,rounds:language.rounds,history:language.history}:{experiment:'gridworld-td',seed:42,config:gridConfig,q:grid.q,history:grid.history,last:grid.last};
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=`rl-explainer-${activeLab==='language'?method:gridConfig.method}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopLanguage();stopGrid();}});
const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(visible)$$('.site-header nav a').forEach(a=>a.classList.toggle('nav-active',a.getAttribute('href')===`#${visible.target.id}`));},{rootMargin:'-15% 0px -55% 0px',threshold:0});
['laboratory','post-training','algorithms','learn'].forEach(id=>observer.observe(document.getElementById(id)));
document.querySelectorAll('[data-math]').forEach(node=>node.innerHTML=typeset(node.dataset.math));
document.querySelector('.core-formula').innerHTML=typeset(EQ.objective,true);
syncParameters();renderLanguage({animate:false});renderGrid();renderChart({animate:false});renderClip();renderGAE();renderAtlas();syncLabIndicator();
