import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const project=new URL('../',import.meta.url);
const read=relative=>readFile(new URL(relative,project),'utf8');
const source=await read('dist/index.html');
const [style,motionStyle,katexStyle,katexScript,favicon]=await Promise.all([
  read('dist/style.css'),read('dist/motion.css'),read('dist/vendor/katex/katex.min.css'),
  read('node_modules/katex/dist/katex.min.js'),read('dist/favicon.svg')
]);
const fontPaths=[...new Set([...katexStyle.matchAll(/url\((fonts\/[^)]+)\)/g)].map(match=>match[1]))];
const embeddedFonts=new Map(await Promise.all(fontPaths.map(async path=>{
  const bytes=await readFile(new URL('dist/vendor/katex/'+path,project));
  const extension=path.split('.').at(-1),mime={woff2:'font/woff2',woff:'font/woff',ttf:'font/ttf'}[extension];
  return[path,`data:${mime};base64,${bytes.toString('base64')}`];
})));
const css=katexStyle.replace(/url\((fonts\/[^)]+)\)/g,(_,path)=>`url(${embeddedFonts.get(path)})`)+ '\n'+style+'\n'+motionStyle;
const modules=await Promise.all(['engine.mjs','math-render.mjs','content.mjs','update-steps.mjs','graph.mjs','motion.mjs','app.mjs'].map(name=>read('dist/'+name)));
const code='const katex=window.katex;\n'+modules.map(module=>module.replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'')).join('\n');
const safeScript=value=>value.replace(/<\/script/gi,'<\\/script');
const html=source
  .replace('href="./favicon.svg"',`href="data:image/svg+xml;base64,${Buffer.from(favicon).toString('base64')}"`)
  .replace('  <link rel="stylesheet" href="./vendor/katex/katex.min.css">','')
  .replace('  <link rel="stylesheet" href="./motion.css">','')
  .replace('  <link rel="stylesheet" href="./style.css">',()=>`  <style>${css}</style>`)
  .replace('  <script type="module" src="./app.mjs"></script>',()=>`  <script>${safeScript(katexScript)}</script>\n  <script type="module">${safeScript(code)}</script>`);
await mkdir(new URL('output/',project),{recursive:true});
const target=new URL('output/RL-Explainer.html',project);
await writeFile(target,html);
console.log(JSON.stringify({path:fileURLToPath(target),bytes:Buffer.byteLength(html),embeddedFonts:fontPaths.length}));
