/**
 * Standalone reproducibility reference: explicit-feedback ALS-WR (JavaScript).
 * This implementation is NOT Apache Spark MLlib, and its results MUST NOT be
 * reported as Spark execution results.
 *
 * Usage: node reference_benchmark.js data/ml-latest-small/ratings.csv
 * Dataset: GroupLens https://grouplens.org/datasets/movielens/latest/
 * Fixed split: xorshift32 seed 20261112, train 80%, val 10%, test 10%.
 */
'use strict';
const fs=require('fs');
const path=process.argv[2]||'data/ml-latest-small/ratings.csv';
if(!fs.existsSync(path)){console.error(`Missing ${path}; run python download_data.py first`);process.exit(1)}
const raw=fs.readFileSync(path,'utf8').trim().split(/\r?\n/).slice(1).map(s=>{const a=s.split(',');return [+a[0],+a[1],+a[2]]});
const users=[...new Set(raw.map(x=>x[0]))],items=[...new Set(raw.map(x=>x[1]))];
const um=new Map(users.map((v,i)=>[v,i])),im=new Map(items.map((v,i)=>[v,i]));
let seed=20261112;
function rand(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return (seed>>>0)/4294967296}
const data=raw.map(x=>({u:um.get(x[0]),i:im.get(x[1]),r:x[2],z:rand()}));
const seedAfterSplit=seed;
const tr=data.filter(x=>x.z<.8),va=data.filter(x=>x.z>=.8&&x.z<.9),te=data.filter(x=>x.z>=.9);
const ur=Array.from({length:users.length},()=>[]),ir=Array.from({length:items.length},()=>[]);
const us=new Float64Array(users.length),is=new Float64Array(items.length);
let sum=0;
for(const x of tr){ur[x.u].push(x);ir[x.i].push(x);us[x.u]+=x.r;is[x.i]+=x.r;sum+=x.r}
const mu=sum/tr.length;
function baseline(x){const nu=ur[x.u].length,ni=ir[x.i].length;return mu+(nu?(us[x.u]/nu-mu)*nu/(nu+10):0)+(ni?(is[x.i]/ni-mu)*ni/(ni+10):0)}
function clamp(x){return Math.max(.5,Math.min(5,x))}
function evalBaseline(ds,kind){let s=0,ae=0,n=0;for(const x of ds){if(!ur[x.u].length||!ir[x.i].length)continue;
 let p=mu;if(kind==='movie')p=(is[x.i]+10*mu)/(ir[x.i].length+10);
 if(kind==='bias')p=baseline(x);p=clamp(p);let e=x.r-p;s+=e*e;ae+=Math.abs(e);n++}
 return {rmse:Math.sqrt(s/n),mae:ae/n,count:n,coverage:n/ds.length}}
function solve(A,b,K){const L=new Float64Array(K*K),v=new Float64Array(K),x=new Float64Array(K);
 for(let i=0;i<K;i++)for(let j=0;j<=i;j++){let t=A[i*K+j];for(let h=0;h<j;h++)t-=L[i*K+h]*L[j*K+h];L[i*K+j]=i===j?Math.sqrt(Math.max(t,1e-10)):t/L[j*K+j]}
 for(let i=0;i<K;i++){let t=b[i];for(let j=0;j<i;j++)t-=L[i*K+j]*v[j];v[i]=t/L[i*K+i]}
 for(let i=K-1;i>=0;i--){let t=v[i];for(let j=i+1;j<K;j++)t-=L[j*K+i]*x[j];x[i]=t/L[i*K+i]}return x}
function runALS({K,lambda,iterations,residualMode}){
 seed=seedAfterSplit;  // Reset the initialization stream for independent experiments.
 const userFactors=Array.from({length:ur.length},()=>new Float64Array(K));
 const itemFactors=Array.from({length:ir.length},()=>new Float64Array(K));
 for(let j=0;j<itemFactors.length;j++)for(let k=0;k<K;k++)itemFactors[j][k]=(rand()-.5)*.45;
 function update(target,obs,other,byuser){for(let z=0;z<target.length;z++){
 const a=obs[z];if(!a.length)continue;
 const A=new Float64Array(K*K),b=new Float64Array(K);
 for(let k=0;k<K;k++)A[k*K+k]=lambda*a.length;
 for(const x of a){const f=other[byuser?x.i:x.u],response=residualMode?x.r-baseline(x):x.r;
 for(let k=0;k<K;k++){b[k]+=response*f[k];for(let h=0;h<K;h++)A[k*K+h]+=f[k]*f[h]}}
 target[z]=solve(A,b,K)}}
 function evaluate(ds){let s=0,ae=0,n=0;for(const x of ds){if(!ur[x.u].length||!ir[x.i].length)continue;
 let p=residualMode?baseline(x):0;for(let k=0;k<K;k++)p+=userFactors[x.u][k]*itemFactors[x.i][k];
 p=clamp(p);let e=x.r-p;s+=e*e;ae+=Math.abs(e);n++}
 return {rmse:Math.sqrt(s/n),mae:ae/n,count:n,coverage:n/ds.length}}
 const prog=[];for(let i=1;i<=iterations;i++){update(userFactors,ur,itemFactors,true);update(itemFactors,ir,userFactors,false);prog.push({iter:i,validation:Math.sqrt(evaluate(va).rmse**2)})}
 return {model:residualMode?'bias_aware_als':'raw_als',rank:K,lambda,iterations,validation:evaluate(va),test:evaluate(te),progress:prog}
}
const result={note:'Independent ALS-WR reference implementation; NOT Spark execution',source:'https://grouplens.org/datasets/movielens/latest/',split:{seed:20261112,train:tr.length,valid:va.length,test:te.length},baseline:{global:evalBaseline(te,'global'),movie:evalBaseline(te,'movie'),bias:evalBaseline(te,'bias')},raw:runALS({K:12,lambda:.1,iterations:8,residualMode:false}),residual:runALS({K:8,lambda:.12,iterations:8,residualMode:true})};
console.log(JSON.stringify(result,null,2));
