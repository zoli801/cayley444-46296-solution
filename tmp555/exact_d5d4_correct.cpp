#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
static constexpr int NP=150,NM=30;using State=array<uint8_t,NP>;
array<array<uint8_t,NP>,NM>P;array<array<uint64_t,256>,NP>Z;
uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
uint64_t hs(const State&s){uint64_t h=0;for(int i=0;i<NP;i++)h^=Z[i][s[i]];return h;}
State mv(const State&s,int m){State t;for(int i=0;i<NP;i++)t[i]=s[P[m][i]];return t;}int inv(int m){return m<15?m+15:m-15;}
struct Node{State s;uint32_t code;uint8_t ax,lay,exp;};struct Rec{uint64_t h;uint32_t code;uint16_t a;uint8_t d,pad;};static_assert(sizeof(Rec)==16);
bool allow(const Node&n,int m){int a=(m%15)/5,l=m%5;bool pos=m<15;if(n.ax==3||a!=n.ax)return true;if(l>n.lay)return true;if(l<n.lay)return false;return n.exp==1&&pos;}
void meta(Node&c,const Node&n,int m){int a=(m%15)/5,l=m%5;bool pos=m<15;if(n.ax==a&&n.lay==l){c.ax=a;c.lay=l;c.exp=2;}else{c.ax=a;c.lay=l;c.exp=pos?1:3;}}
vector<int> dec(uint32_t c,int d){vector<int>w(d);for(int i=0;i<d;i++)w[i]=(c>>(5*i))&31;return w;}
State app(State s,const vector<int>&w){for(int m:w)s=mv(s,m);return s;}
vector<int> bridge(uint32_t c1,int d1,uint32_t c2,int d2){auto w=dec(c1,d1),b=dec(c2,d2);for(int i=d2-1;i>=0;i--)w.push_back(inv(b[i]));return w;}
struct Bloom{vector<uint64_t>b;uint64_t mask;Bloom(int lg=29):b(1ULL<<(lg-6)),mask((1ULL<<lg)-1){};void add(uint64_t h){for(int k=0;k<3;k++){uint64_t x=sm(h+0x9e3779b97f4a7c15ULL*k)&mask;b[x>>6]|=1ULL<<(x&63);}}bool has(uint64_t h)const{for(int k=0;k<3;k++){uint64_t x=sm(h+0x9e3779b97f4a7c15ULL*k)&mask;if(!(b[x>>6]>>(x&63)&1))return false;}return true;}};
int main(int ac,char**av){if(ac<6){cerr<<"moves paths targets pid source_limit\n";return 2;}int pid=stoi(av[4]),lim=stoi(av[5]);ifstream mf(av[1]);for(int m=0;m<NM;m++)for(int i=0;i<NP;i++){int x;mf>>x;P[m][i]=x;}for(int i=0;i<NP;i++)for(int c=0;c<256;c++)Z[i][c]=sm(1234567+i*911+c*131);
 ifstream pf(av[2]);string ln;vector<vector<int>>paths;while(getline(pf,ln)){istringstream ss(ln);vector<int>w;int x;while(ss>>x)w.push_back(x);paths.push_back(w);}auto path=paths.at(pid);int L=path.size();ifstream tf(av[3]);vector<State>targets;while(getline(tf,ln)){istringstream ss(ln);State s{};int x;for(int i=0;i<NP;i++){if(!(ss>>x)){cerr<<"bad target line\n";return 3;}s[i]=(uint8_t)x;}targets.push_back(s);}State v0=targets.at(pid);vector<State>A(L+1);A[0]=v0;for(int i=0;i<L;i++)A[i+1]=mv(A[i],path[i]);
 const uint64_t PER=447706;vector<Rec>R;R.reserve(PER*(L+1ULL));auto t0=chrono::steady_clock::now();
 for(int ai=0;ai<=L;ai++){R.push_back({hs(A[ai]),0,(uint16_t)ai,0,0});vector<Node>cur(1);cur[0]={A[ai],0,3,0,0};for(int d=1;d<=4;d++){vector<Node>nx;nx.reserve(cur.size()*25);for(auto &n:cur)for(int m=0;m<NM;m++)if(allow(n,m)){Node c;c.s=mv(n.s,m);c.code=n.code|(uint32_t(m)<<(5*(d-1)));meta(c,n,m);R.push_back({hs(c.s),c.code,(uint16_t)ai,(uint8_t)d,0});nx.push_back(c);}cur.swap(nx);} }
 cerr<<"radius4 generated recs="<<R.size()<<" sec="<<chrono::duration<double>(chrono::steady_clock::now()-t0).count()<<"\n";
 sort(R.begin(),R.end(),[](auto&a,auto&b){return a.h<b.h;});Bloom bloom(29);for(auto&r:R)bloom.add(r.h);cerr<<"sort+bloom sec="<<chrono::duration<double>(chrono::steady_clock::now()-t0).count()<<"\n";
 int sources=(lim<0?L+1:min(L+1,lim));uint64_t tested=0,maybe=0;int bestgain=0,bi=0,bj=0;vector<int>bestw;
 for(int ai=0;ai<sources;ai++){
  vector<Node>cur(1);cur[0]={A[ai],0,3,0,0};for(int d=1;d<=4;d++){vector<Node>nx;nx.reserve(cur.size()*25);for(auto&n:cur)for(int m=0;m<NM;m++)if(allow(n,m)){Node c;c.s=mv(n.s,m);c.code=n.code|(uint32_t(m)<<(5*(d-1)));meta(c,n,m);nx.push_back(c);}cur.swap(nx);} 
  for(auto&n:cur)for(int m=0;m<NM;m++)if(allow(n,m)){
   State x=mv(n.s,m);uint32_t code=n.code|(uint32_t(m)<<20);uint64_t h=hs(x);tested++;if(!bloom.has(h))continue;maybe++;
   auto lo=lower_bound(R.begin(),R.end(),h,[](const Rec&r,uint64_t v){return r.h<v;});for(auto it=lo;it!=R.end()&&it->h==h;++it){int aj=it->a;if(aj==ai)continue;int i=min(ai,aj),j=max(ai,aj);int len=5+it->d;if(len>=j-i)continue;vector<int>w=(ai<aj)?bridge(code,5,it->code,it->d):bridge(it->code,it->d,code,5);if(app(A[i],w)==A[j]){int gain=(j-i)-len;if(gain>bestgain){bestgain=gain;bi=i;bj=j;bestw=w;cerr<<"FOUND gain="<<gain<<" i="<<i<<" j="<<j<<" len="<<len<<" source="<<ai<<"\n";}}}
  }
  if(ai%5==0)cerr<<"source "<<ai<<"/"<<sources<<" tested="<<tested<<" maybe="<<maybe<<" sec="<<chrono::duration<double>(chrono::steady_clock::now()-t0).count()<<"\n";
  if(bestgain>0)break;
 }
 if(bestgain){vector<int>out;out.insert(out.end(),path.begin(),path.begin()+bi);out.insert(out.end(),bestw.begin(),bestw.end());out.insert(out.end(),path.begin()+bj,path.end());if(app(v0,out)!=A[L]){cerr<<"final fail\n";return 4;}cout<<"FOUND pid="<<pid<<" old="<<L<<" new="<<out.size()<<" gain="<<L-out.size()<<" i="<<bi<<" j="<<bj<<"\nPATH";for(int m:out)cout<<' '<<m;cout<<"\n";}else cout<<"NONE pid="<<pid<<" old="<<L<<" sources="<<sources<<" tested="<<tested<<" maybe="<<maybe<<"\n";
}
