import katex from './vendor/katex/katex.mjs';

export const EQ = {
  softmax: String.raw`\pi_\theta(y_i\mid x)=\frac{e^{\theta_i}}{\sum_{j=1}^{4} e^{\theta_j}}`,
  rewardShaping: String.raw`\widetilde R_i=R_i-\beta\log\frac{\pi_{\rm old}(y_i\mid x)}{\pi_{\rm ref}(y_i\mid x)}`,
  testReward: String.raw`R_i=\frac{N_{\rm passed}}{N_{\rm tests}}`,
  mathReward: String.raw`R_i=\mathbf 1\!\left[\operatorname{answer}(y_i)=17\times3\right]`,
  humanReward: String.raw`R_i=\operatorname{score}_{\rm human}(x,y_i)`,
  sequenceLogprob: String.raw`\log\pi_\theta(y\mid x)=\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t})`,
  groupAdvantage: String.raw`\begin{aligned}\hat A_i&=\frac{R_i-\overline R}{\sigma_R}\\\overline R&=\frac1G\sum_{i=1}^{G}R_i,\quad\sigma_R=\sqrt{\frac1G\sum_{i=1}^{G}(R_i-\overline R)^2}\end{aligned}`,
  ppoAdvantage: String.raw`\hat A_i=\widetilde R_i-V_{\rm old}(x)`,
  preferenceMargin: String.raw`m=\beta\left[\log\frac{p_w}{q_w}-\log\frac{p_l}{q_l}\right]`,
  sftLoss: String.raw`\mathcal L_{\rm SFT}=-\log\pi_\theta(y^*\mid x)`,
  clipped: String.raw`\ell_i=\min\!\left(\rho_i\hat A_i,\;\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i\right)`,
  grpoObjective: String.raw`\begin{aligned}\ell_i&=\min\!\left(\rho_i\hat A_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i\right)\\L&=\frac1G\sum_i\ell_i-\beta D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})\end{aligned}`,
  ppoObjective: String.raw`L^{\rm CLIP}=\frac1G\sum_{i=1}^{G}\min\!\left(\rho_i\hat A_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i\right)`,
  dpoUpdate: String.raw`\theta\leftarrow\theta+\eta\beta\sigma(-m)(e_w-e_l)`,
  sftUpdate: String.raw`\theta\leftarrow\theta+\eta(e_{y^*}-p)`,
  expectedReward: String.raw`\mathbb E[R]=\sum_{i=1}^{4}\pi_\theta(y_i\mid x)R_i`,
  gradientParts: String.raw`\begin{aligned}z_{ij}&=\mathbf1[y_i=j]-p_j\\g_{{\rm data},j}&=\frac1G\sum_i c_i z_{ij}\\\frac{\partial L}{\partial\theta_j}&=g_{{\rm data},j}-\beta\frac{\partial D_{\rm KL}}{\partial\theta_j}\end{aligned}`,
  sequenceProduct: String.raw`\pi_\theta(y\mid x)=\prod_{t=1}^{|y|}\pi_\theta(y_t\mid x,y_{<t})`,
  qUpdate: String.raw`\begin{aligned}y&=r+\gamma\max_{a'}Q(s',a')\\\delta&=y-Q(s,a)\\Q(s,a)&\leftarrow Q(s,a)+\alpha\delta\end{aligned}`,
  sarsaUpdate: String.raw`\begin{aligned}y&=r+\gamma Q(s',a')\\\delta&=y-Q(s,a)\\Q(s,a)&\leftarrow Q(s,a)+\alpha\delta\end{aligned}`,
  qTarget: String.raw`y=r+\gamma\max_{a'}Q(s',a')`,
  sarsaTarget: String.raw`y=r+\gamma Q(s',a')`,
  returnAdvantage: String.raw`\begin{aligned}G_t&=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}\\A^\pi(s,a)&=Q^\pi(s,a)-V^\pi(s)\end{aligned}`,
  gae: String.raw`\begin{aligned}\delta_t&=r_{t+1}+\gamma V(s_{t+1})-V(s_t)\\\hat A_t^{\rm GAE}&=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}\end{aligned}`,
  klGradient: String.raw`\begin{aligned}K&=\sum_j p_j\log\frac{p_j}{q_j}\\\frac{\partial K}{\partial\theta_j}&=p_j\!\left(\log\frac{p_j}{q_j}-K\right)\end{aligned}`,
  softmaxGradient: String.raw`p_i=\frac{e^{\theta_i}}{\sum_j e^{\theta_j}},\qquad\nabla_\theta\log p_i=e_i-p`,
  policyGradient: String.raw`\nabla_\theta J\approx\mathbb E_{\tau\sim\pi_\theta}\!\left[\sum_t\gamma^t\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]`,
  values: String.raw`V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],\qquad Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]`,
  importance: String.raw`\rho_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\rm old}(y_i\mid x)},\qquad\left.\rho_i\right|_{\theta=\theta_{\rm old}}=1`, 
  clippedGradient: String.raw`c_i=\begin{cases}0,&\hat A_i>0,\ \rho_i>1+\epsilon\\0,&\hat A_i<0,\ \rho_i<1-\epsilon\\\rho_i\hat A_i,&\text{otherwise}\end{cases}\qquad\nabla_\theta\ell_i=c_i(e_{y_i}-p)`,
  parameterUpdate: String.raw`g=\nabla_\theta L,\qquad\Delta\theta=\eta g,\qquad\theta^{\rm new}=\theta+\Delta\theta`,
  dpoGradient: String.raw`\nabla_\theta\log\sigma(m)=\beta\sigma(-m)(e_w-e_l)`,
  rewardModel: String.raw`\begin{aligned}P(y_w\succ y_l\mid x)&=\sigma\!\left(r_\psi(x,y_w)-r_\psi(x,y_l)\right)\\\mathcal L_{\rm RM}&=-\log P(y_w\succ y_l\mid x)\end{aligned}`,
  dpoReward: String.raw`r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)}+C(x)`,
  criticLoss: String.raw`\mathcal L_V=\frac{1}{2G}\sum_i\left(V_\phi(x)-\widetilde R_i\right)^2`,
  kl: String.raw`D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})=\sum_i p_i\log\frac{p_i}{q_i}`,
  objective: String.raw`\begin{aligned}J(\theta)&=\mathbb E_{\tau\sim\pi_\theta}[G_0]\\G_0&=\sum_{t=0}^{T-1}\gamma^t r_{t+1}\end{aligned}`,
  algorithms: {
    q: String.raw`\begin{aligned}y&=r+\gamma\max_{a'}Q(s',a')\\Q(s,a)&\leftarrow Q(s,a)+\alpha[y-Q(s,a)]\end{aligned}`,
    sarsa: String.raw`\begin{aligned}y&=r+\gamma Q(s',a')\\Q(s,a)&\leftarrow Q(s,a)+\alpha[y-Q(s,a)]\end{aligned}`,
    reinforce: String.raw`\nabla_\theta J\approx\sum_t\gamma^tG_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)`,
    dqn: String.raw`\begin{aligned}\mathcal L(\theta)&=\mathbb E\!\left[(Q_\theta(s,a)-y)^2\right]\\y&=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')\end{aligned}`,
    'actor-critic': String.raw`\begin{aligned}\delta_t&=r_{t+1}+\gamma V(s_{t+1})-V(s_t)\\\theta&\leftarrow\theta+\eta\delta_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\end{aligned}`,
    ppo: String.raw`\begin{aligned}\rho_t&=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)}\\L^{\rm CLIP}&=\mathbb E_t\!\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right]\end{aligned}`,
    sft: String.raw`\mathcal L_{\rm SFT}=-\mathbb E_{(x,y)\sim\mathcal D}\!\left[\sum_t\log\pi_\theta(y_t\mid x,y_{<t})\right]`,
    dpo: String.raw`\begin{aligned}m&=\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}\right]\\\mathcal L_{\rm DPO}&=-\log\sigma(m)\end{aligned}`,
    grpo: String.raw`\begin{aligned}\hat A_i&=\frac{R_i-\overline R}{\sigma_R}\\L&=\frac1G\sum_i\min\!\left(\rho_i\hat A_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i\right)\\&\quad-\beta D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})\end{aligned}`
  }
};

export function typeset(tex,displayMode=false) {
  return katex.renderToString(tex,{displayMode,throwOnError:true,strict:'error',output:'htmlAndMathml'});
}
export function displayEquation(tex,compact=false) {
  return `<div class="formula-box${compact?' compact-equation':''}">${typeset(tex,true)}</div>`;
}
