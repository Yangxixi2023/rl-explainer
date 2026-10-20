import { TASKS, softmax, languageObjective, clippedSurrogate } from './engine.mjs';
import { typeset } from './math-render.mjs';
import { currentUpdateFrame } from './update-steps.mjs';

const color = { text:'#343434', muted:'#858585', line:'#d5d5d5', trajectory:'#d27860', positive:'#39988d', negative:'#9674b4', policy:'#6289b6' };
const num = (value, precision = 3) => Number(value.toFixed(precision)).toFixed(precision).replace('-', '−');
const safe = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
const letter = i => String.fromCharCode(65+i);
const text = (x,y,value,attrs='') => {
  const numeric=String(value).match(/^([+−-]?\d+(?:\.(\d+))?)(%)?$/);
  const numberAttrs=numeric?`data-number="${numeric[1].replace('−','-')}" data-precision="${numeric[2]?.length??0}" data-plus="${numeric[1].startsWith('+')}" data-suffix="${numeric[3]??''}"`:'';
  return `<text x="${x}" y="${y}" ${attrs} ${numberAttrs}>${safe(value)}</text>`;
};
const flow=(stage,key,order=0,extra='')=>`data-motion-key="flow:${key}" data-flow-stage="${stage}" data-flow-order="${order}" ${extra}`;
function math(x,y,tex,{size=18,width=230,align='center'}={}) {
  const left=align==='left'?x:x-width/2;
  return `<foreignObject x="${left}" y="${y-size*1.25}" width="${width}" height="${size*2.5}" class="graph-equation"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:${size}px;text-align:${align};line-height:2.3">${typeset(tex)}</div></foreignObject>`;
}
const rect = (x,y,w,h,fill='#fff',stroke=color.line,attrs='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill}" stroke="${stroke}" ${attrs}/>`;
function curve(x1,y1,x2,y2,stroke=color.line,width=1.2,attrs='') {
  const bend = (x2-x1)*.48;
  return `<path d="M${x1},${y1} C${x1+bend},${y1} ${x2-bend},${y2} ${x2},${y2}" fill="none" stroke="${stroke}" stroke-width="${width}" ${attrs}/>`;
}
function group(stage,body,attrs='',key='') { return `<g class="g-interactive" role="button" tabindex="0" data-stage="${stage}" ${key?`data-motion-key="${key}"`:''} ${attrs}>${body}</g>`; }
function heading(x,number,title,subtitle,stage,isMath=false) {
  return group(stage,rect(x-5,7,190,78,'transparent','none','class="g-heading-hit"')+text(x,29,number,`fill="${color.muted}" class="g-step"`)+text(x+25,29,title,'class="g-heading"')+(isMath?math(x+25,53,subtitle,{size:15,width:260,align:'left'}):text(x+25,50,subtitle,'class="g-subtitle"'))+text(x+25,79,'展开 ↗','class="g-expand"'),'',`heading:${number}`);
}
function cell(x,y,w,value,c,stage,attrs='',strength=0,key='') {
  const alpha=Math.round(14+68*Math.min(1,strength)).toString(16).padStart(2,'0');
  return group(stage,rect(x,y-14,w,28,c+alpha,c+'75')+text(x+w/2,y+5,value,`class="g-number" text-anchor="middle" fill="${c}"`),attrs,key);
}
function wrap(value,capacity=32) {
  const lines=[];let line='',units=0;
  for(const character of value){const size=/[\u0000-\u00ff]/.test(character)?1:2;if(units+size>capacity){lines.push(line);line='';units=0;}line+=character;units+=size;}
  if(line)lines.push(line);return lines;
}
function multiline(x,y,lines,attrs='class="g-answer"') {return `<text x="${x}" y="${y}" ${attrs}>${lines.map((line,i)=>`<tspan x="${x}" dy="${i?16:0}">${safe(line)}</tspan>`).join('')}</text>`;}
function begin(width,height,label){return `<svg viewBox="0 0 ${width} ${height}" class="computation-svg" role="group" aria-label="${label}"><rect width="${width}" height="${height}" fill="white"/>`;}

function computationFrame(state,batch,epochIndex,options) {
  const operation=options.operation,atomic=operation?currentUpdateFrame(operation):null;
  const epoch=!options.preview&&!operation&&batch.epochDetails.length?batch.epochDetails[epochIndex]:null;
  const thetaIn=operation?operation.thetaBefore:epoch?epoch.thetaBefore:state.theta;
  const thetaOut=atomic?atomic.theta:epoch?epoch.thetaAfter:state.theta;
  const objective=operation?operation.objective:languageObjective(thetaIn,state.reference,batch);
  return {thetaIn,thetaOut,pIn:softmax(thetaIn),objective,epoch,atomic};
}

export function renderComputationGraph(state,batch,selectedResponse,epochIndex=0,options={}) {
  const {preview=false}=options;
  const frame=computationFrame(state,batch,epochIndex,options);
  if(batch.method==='dpo')return preferenceGraph(state,batch,frame);
  if(batch.method==='sft')return supervisedGraph(state,batch,frame);
  const method=batch.method,task=TASKS[state.taskKey],n=batch.ids.length;
  const {pIn,objective,epoch}=frame;
  const rowY=k=>122+k*36;
  const responseLines=task.responses.map(r=>wrap(r.tokens.filter(t=>t!=='<eos>').join('')));
  const responseHeight=Math.max(...responseLines.map(lines=>lines.length*16+32));
  const responseY=i=>139+i*(responseHeight+22);
  const responseBottom=responseY(3)-14+responseHeight;
  const statsY=Math.max(454,rowY(n-1)+48);
  const height=Math.max(554,statsY+(method==='ppo'?120:100),responseBottom+95);
  const sumY=(rowY(0)+rowY(n-1))/2;
  const advantageScale=Math.max(...batch.advantages.map(Math.abs));
  const epochLabel=frame.atomic?`逐项计算 · epoch ${options.operation.epoch}`:epoch?`epoch ${epochIndex+1} / ${batch.config.epochs}`:preview?`下一步梯度 · epoch ${batch.epochDetails.length+1}`:'更新前的梯度预览';
  let svg=begin(1450,height,`${method.toUpperCase()} 逐样本计算图`);
  svg+=heading(34,'01','从策略采样',String.raw`\pi_{\rm old}(y\mid x)`,0,true)+heading(448,'02',method==='ppo'?'奖励与 KL 整形':'验证奖励','R(y)',1,true)+heading(650,'03',method==='ppo'?'减去价值预测':'组内标准化',String.raw`\hat A_i`,2,true)+heading(843,'04','计算代理目标',String.raw`\min(\rho_i\hat A_i,\operatorname{clip}(\rho_i)\hat A_i)`,3,true)+heading(1130,'05','更新参数',String.raw`\theta\leftarrow\theta+\eta\nabla_\theta L`,3,true);
  const promptLines=wrap(task.prompt,16);
  svg+=text(24,115,'prompt x','class="g-label"')+rect(15,128,120,Math.max(73,promptLines.length*16+22),'#fafafa','#dedede')+multiline(28,151,promptLines,'class="g-prompt"');
  svg+=math(74,263,String.raw`\pi_{\rm old}`,{size:28,width:110})+text(74,292,'采样策略','text-anchor="middle" class="g-subtitle"');
  task.responses.forEach((response,i)=>{
    const y=responseY(i),selected=i===selectedResponse;
    const repeats=batch.ids.filter(id=>id===i).length;
    svg+=`<g data-graph-response="${i}" class="g-response ${selected?'g-selected':''}">`+curve(92,296,194,y+10,color.trajectory,1+batch.old[i]*13,flow(0,`policy:${i}`,0,`data-flow-repeat="${repeats}"`))+group(0,rect(194,y-14,205,responseHeight,selected?'#fff4ef':'#fffaf8',selected?color.trajectory:'#e8c1b6')+text(204,y+2,`${letter(i)} · 采样概率 ${num(batch.old[i]*100,1)}%`,'class="g-label"')+multiline(204,y+23,responseLines[i])+`<title>${safe(response.tokens.join(' '))}</title>`,`data-response="${i}"`,`answer:${i}`)+`<rect data-motion-key="answer-bar:${i}" x="195" y="${y-12+responseHeight}" width="${203*batch.old[i]}" height="3" fill="${color.trajectory}" opacity=".7"/></g>`;
  });
  svg+=text(199,responseBottom+34,`有放回采样 ${n} 次`,'class="g-label"')+text(199,responseBottom+55,'线宽表示采样概率','class="g-subtitle"');
  batch.ids.forEach((id,k)=>{
    const y=rowY(k),sourceY=responseY(id)+10,A=batch.advantages[k],signal=A>=0?color.positive:color.negative;
    const ratio=pIn[id]/batch.old[id],term=clippedSurrogate(ratio,A,batch.config.clip),attrs=`data-sample-index="${k}" data-sample="${k}"`;
    svg+=`<g data-sample="${k}" class="g-sample">`+curve(399,sourceY,486,y,color.trajectory,1.1,flow(1,`sample:${k}:reward`))+text(466,y+4,letter(id),'class="g-label" text-anchor="end"')+cell(486,y,50,num(batch.rewards[k],2),color.positive,1,attrs,0,`sample:${k}:reward`);
    if(method==='ppo')svg+=curve(536,y,576,y,'#bfcbd5',1.2,flow(2,`sample:${k}:advantage`))+cell(576,y,58,num(batch.shaped[k]),color.policy,1,attrs,0,`sample:${k}:shaped`)+curve(634,y,688,y,signal,1.2,flow(2,`sample:${k}:shaped`,1));
    else svg+=curve(536,y,688,y,signal,1.3,flow(2,`sample:${k}:advantage`));
    svg+=cell(688,y,61,(A>0?'+':'')+num(A),signal,2,attrs,advantageScale===0?0:Math.abs(A)/advantageScale,`sample:${k}:advantage`)+curve(749,y,814,y,signal,1.5,flow(3,`sample:${k}:ratio`,0,`data-gradient-edge="true" data-clipped="${term.active}"`))+cell(814,y,59,num(ratio),color.policy,3,attrs,Math.abs(ratio-1)/batch.config.clip,`sample:${k}:ratio`)+curve(873,y,941,y,signal,1.3,flow(3,`sample:${k}:objective`,1,`data-gradient-edge="true" data-clipped="${term.active}"`))+cell(941,y,62,num(term.value),term.active?'#999999':signal,3,attrs,0,`sample:${k}:objective`);
    if(term.active)svg+=text(915,y+5,'⊣','class="g-clip"');
    svg+=curve(1003,y,1076,sumY,term.active?'#d8d8d8':signal,1.2,flow(3,`sample:${k}:reduce`,2,`data-gradient-edge="true" data-clipped="${term.active}"`))+'</g>';
  });
  svg+=math(511,94,'R_i',{size:14,width:55})+math(718,94,String.raw`\hat A_i`,{size:14,width:55})+math(844,94,String.raw`\rho_i`,{size:14,width:55})+math(972,94,String.raw`\ell_i`,{size:14,width:55});
  if(method==='ppo') svg+=math(605,94,String.raw`\widetilde R_i`,{size:14,width:55})+group(2,rect(552,statsY,196,82,'#f4f7fb','#c1cfdf')+math(565,statsY+19,String.raw`V_{\rm old}=${num(batch.oldValue)}`,{size:15,width:175,align:'left'})+math(565,statsY+41,String.raw`\hat A_i=\widetilde R_i-V_{\rm old}`,{size:15,width:175,align:'left'})+math(565,statsY+66,String.raw`V(x)=${num(state.value)}`,{size:15,width:175,align:'left'}),'','statistics');
  else svg+=group(2,rect(468,statsY,278,55,'#f5f9f7','#b7d6cf')+math(482,statsY+18,String.raw`\overline R=${num(batch.mean)},\quad\sigma_R=${num(batch.sd)}`,{size:15,width:255,align:'left'})+math(482,statsY+42,batch.sd===0?String.raw`\sigma_R=0\;\Longrightarrow\;\hat A_i=0`:String.raw`\hat A_i=(R_i-\overline R)/\sigma_R`,{size:15,width:255,align:'left'}),'','statistics');
  svg+=group(3,`<circle cx="1084" cy="${sumY}" r="17" fill="#fafafa" stroke="#a7a7a7"/>`+math(1084,sumY+3,String.raw`\Sigma`,{size:20,width:32})+text(1084,sumY+37,`÷ ${n}`,'text-anchor="middle" class="g-label"')+math(1084,sumY+63,method==='grpo'?String.raw`-\beta D_{\rm KL}`:String.raw`L^{\rm CLIP}`,{size:12,width:80}),'','objective-reduction');
  svg+=curve(1101,sumY,1114,269,color.policy,1.4,flow(3,'loss-gradient',3))+`<path d="M1125 175H1114V362H1125" fill="none" stroke="${color.policy}"/>`;
  svg+=updateVectors(state,batch,frame);
  svg+=text(1140,423,epochLabel,'class="g-subtitle"')+math(1140,446,String.raw`\eta=${num(batch.config.lr,2)}`,{size:15,width:210,align:'left'})+math(1140,469,String.raw`L=${num(objective.objective,4)}`,{size:16,width:210,align:'left'})+text(1140,493,`${objective.clipped} / ${n} 样本进入裁剪区`,'class="g-subtitle"');
  svg+=text(814,statsY+22,'⊣ 该样本的代理梯度为 0','class="g-subtitle"');
  svg+=text(25,height-14,frame.atomic?'由同一梯度计算各参数增量，再统一归一化。灰色裁剪项的样本梯度为零。':epoch?'概率比、损失和梯度来自所选 epoch 更新前；最右侧是该 epoch 更新后的策略。':'单步查看梯度累加、参数增量与概率归一化；更新一轮会依次播放四个 epoch。','class="g-footnote"');
  return svg+'</svg>';
}

function preferenceGraph(state,batch,frame){
  const p=frame.pIn,w=batch.winner,l=batch.loser,d=frame.objective;
  const ratios=[Math.log(p[w]/state.reference[w]),Math.log(p[l]/state.reference[l])];
  let svg=begin(1450,554,'DPO 偏好对计算图');
  svg+=heading(25,'01','读取偏好对','固定的 chosen / rejected',0)+heading(335,'02','对数概率',String.raw`\log p-\log q`,1,true)+heading(625,'03','偏好差值',String.raw`m=\beta(\log\frac{p_w}{q_w}-\log\frac{p_l}{q_l})`,2,true)+heading(870,'04','偏好损失',String.raw`\mathcal L=-\log\sigma(m)`,3,true)+heading(1130,'05','更新参数',String.raw`\theta\leftarrow\theta-\eta\nabla\mathcal L`,3,true);
  [w,l].forEach((id,k)=>{
    const y=164+k*127,c=k===0?color.positive:color.negative;
    const response=TASKS[state.taskKey].responses[id].tokens.filter(t=>t!=='<eos>').join('');
    const lines=wrap(response,39),h=Math.max(77,lines.length*16+46);
    svg+=group(0,rect(25,y-33,255,h,c+'0b',c+'80')+text(40,y-12,`${k===0?'chosen':'rejected'} · ${letter(id)}`,'class="g-label"')+multiline(40,y+12,lines),`data-response="${id}"`,`answer:${id}`)+curve(280,y,350,y,c,1.4,flow(1,`pair:${id}:logprob`));
    svg+=group(1,rect(350,y-37,208,82,'#f4f7fb','#b8c9dc')+math(366,y-13,String.raw`\log p_${k===0?'w':'l'}=${num(Math.log(p[id]))}`,{size:16,width:182,align:'left'})+math(366,y+11,String.raw`\log q_${k===0?'w':'l'}=${num(Math.log(state.reference[id]))}`,{size:16,width:182,align:'left'})+math(366,y+33,String.raw`\log(p/q)=${num(ratios[k])}`,{size:14,width:182,align:'left'}),'',`logprob:${id}`)+curve(558,y,644,226,c,2,flow(2,`pair:${id}:margin`));
  });
  svg+=group(2,rect(644,187,203,83,'#fbf8f1','#d7c5a3')+math(745,217,String.raw`\beta\,(${num(ratios[0])}-${num(ratios[1])})`,{size:16,width:193})+math(745,247,`m=${num(d.z)}`,{size:18,width:190}),'','learning-signal')+curve(847,226,909,226,'#c5b392',2,flow(3,'margin-loss',0,'data-gradient-edge="true"'));
  svg+=group(3,rect(909,190,172,75,'#f4f7fb','#b8c9dc')+math(995,216,String.raw`\sigma(m)=${num(d.preference)}`,{size:17,width:160})+math(995,246,String.raw`\mathcal L=${num(-d.objective)}`,{size:18,width:160}),'','objective-reduction');
  for(let j=0;j<4;j++)svg+=curve(1081,226,1132,189+j*53,color.policy,1.2,flow(3,`gradient-input:${j}`,1));
  svg+=updateVectors(state,batch,frame);
  svg+=group(3,rect(638,384,445,67,'#f7f9fc','#c8d4e1')+math(861,411,String.raw`\nabla_\theta(-\mathcal L)=\beta\sigma(-m)(e_w-e_l)`,{size:18,width:420})+text(861,438,'偏好对决定梯度方向；其他 logit 的梯度为 0。','text-anchor="middle" class="g-subtitle"'),'','update-formula');
  svg+=text(25,540,'同题偏好对 → 相对对数概率差 → 偏好损失 → 参数更新。','class="g-footnote"');
  return svg+'</svg>';
}

function supervisedGraph(state,batch,frame){
  const p=frame.pIn,w=batch.winner,d=frame.objective;
  let svg=begin(1450,554,'SFT 示范学习计算图');
  svg+=heading(25,'01','给定示范',String.raw`(x,y^*)`,0,true)+heading(355,'02','当前回答概率',String.raw`p^*=\pi_\theta(y^*\mid x)`,1,true)+heading(695,'03','负对数似然',String.raw`\mathcal L=-\log p^*`,2,true)+heading(1130,'04','更新参数',String.raw`\theta\leftarrow\theta-\eta\nabla\mathcal L`,3,true);
  const lines=wrap(TASKS[state.taskKey].responses[w].tokens.filter(t=>t!=='<eos>').join(''),39);
  svg+=group(0,rect(25,176,260,Math.max(82,lines.length*16+50),'#fff6f1','#dfb7a9')+text(42,203,`示范回答 ${letter(w)}`,'class="g-label"')+multiline(42,234,lines),`data-response="${w}"`,`answer:${w}`)+curve(285,218,356,218,color.trajectory,3,flow(1,`pair:${w}:logprob`));
  svg+=group(1,rect(356,176,237,82,'#f3f7fc','#b7cadd')+math(475,211,String.raw`\operatorname{softmax}(\theta)`,{size:23,width:220})+math(475,241,String.raw`p^*=${num(p[w],5)}`,{size:17,width:210}),'',`logprob:${w}`)+curve(593,218,696,218,color.policy,3,flow(2,'demonstration-loss'));
  svg+=group(2,rect(696,176,223,82,'#f6f9f2','#bcd1ab')+math(808,212,String.raw`-\log(${num(p[w],4)})`,{size:22,width:210})+math(808,241,String.raw`\mathcal L=${num(-d.objective,4)}`,{size:18,width:210}),'','objective-reduction');
  for(let j=0;j<4;j++)svg+=curve(919,218,1132,189+j*53,color.policy,1.2,flow(3,`gradient-input:${j}`));
  svg+=updateVectors(state,batch,frame);
  svg+=group(3,rect(638,384,445,67,'#f7f9fc','#c8d4e1')+math(861,414,String.raw`\nabla_\theta\log p^*=e_{y^*}-p`,{size:20,width:420})+text(861,438,'softmax 的归一化使四个 logit 都参与更新。','text-anchor="middle" class="g-subtitle"'),'','update-formula');
  svg+=text(25,540,'示范回答 → 当前概率 → 负对数似然 → 提高示范的生成概率。','class="g-footnote"');
  return svg+'</svg>';
}

function updateVectors(state,batch,frame){
  const {atomic,epoch,objective}=frame;
  const theta=frame.thetaOut,gradient=atomic?atomic.gradient:objective.gradient;
  const showWeights=atomic&&(atomic.kind==='weight'||atomic.kind==='sum');
  const computingTheta=atomic&&['parameter','weight','sum','probability','critic'].includes(atomic.kind);
  const computingProbability=atomic&&['probability','critic'].includes(atomic.kind);
  const probabilities=atomic?showWeights?atomic.weights:atomic.probabilities:softmax(theta);
  const scale=Math.max(...gradient.map(Math.abs));
  let svg=math(1161,151,String.raw`g_j`,{size:16,width:75})+math(1246,151,computingTheta?String.raw`\theta_j^{\rm new}`:String.raw`\theta_j`,{size:16,width:75})+math(1361,151,showWeights?'z_j':computingProbability?String.raw`p_j^{\rm new}`:String.raw`\pi_\theta`,{size:16,width:75});
  svg+=text(1161,126,atomic&&(atomic.kind==='sample'||atomic.kind==='regularize')?'累积梯度':'目标梯度','text-anchor="middle" class="g-subtitle"');
  svg+=text(1361,126,showWeights?'指数权重':computingProbability?'逐项归一化':epoch?'更新后概率':'更新前概率','text-anchor="middle" class="g-subtitle"');
  for(let j=0;j<theta.length;j++){
    const y=189+j*53,g=gradient[j],probability=probabilities[j];
    const probabilityText=probability===null?'…':showWeights?num(probability,4):num(probability*100,1)+'%';
    svg+=text(1122,y-19,letter(j),'class="g-label"')+cell(1132,y,64,(g>0?'+':'')+num(g),g>=0?color.positive:color.negative,3,`data-gradient="${j}"`,scale===0?0:Math.abs(g)/scale,`gradient:${j}`)+curve(1196,y,1217,y,color.policy,1.2,flow(4,`parameter:${j}`))+cell(1217,y,64,computingTheta&&!atomic.thetaReady[j]?'…':num(theta[j]),color.policy,3,`data-response="${j}"`,0,`theta:${j}`)+curve(1281,y,1321,y,color.policy,1.2,flow(4,`probability:${j}`,1))+cell(1321,y,80,probabilityText,color.policy,4,`data-response="${j}"`,showWeights||probability===null?0:probability,`probability:${j}`);
    if(!showWeights&&probability!==null)svg+=`<rect data-motion-key="probability-bar:${j}" x="1322" y="${y+17}" height="4" width="${78*probability}" fill="${color.policy}" opacity=".7"/>`;
  }
  if(batch.method==='dpo'||batch.method==='sft')svg+=text(1140,423,atomic?'逐项计算新策略':epoch?'本轮更新后':'更新前的梯度','class="g-subtitle"')+math(1140,446,String.raw`\eta=${num(batch.config.lr,2)}`,{size:15,width:150,align:'left'});
  return svg;
}
