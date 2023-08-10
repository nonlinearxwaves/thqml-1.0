%% theory of entanglement in a BS
% after PRA 66 064304
clearvars
syms theta phi0 phi1 ra phia rb phib Y A c2a c2b s2a s2b real 

sha=sinh(ra);
cha=cosh(ra);
shb=sinh(rb);
chb=cosh(rb);
st = sin(theta);
ct = cos(theta);
xa=sha*cha
xb=shb*chb
Sa=cha^2+sha^2
Sb=chb^2+shb^2
Da = 2*phi0-phia
Db = 2*phi1-phib

m11=Sa*ct^2+Sb*st^2+2*xa*ct^2*cos(Da)+2*xb*st^2*cos(Db)
m12=2*xa*ct^2*sin(Da)+2*xb*st^2*sin(Db)
m21=m12
m22=Sa*ct^2+Sb*st^2-2*xa*ct^2*cos(Da)-2*xb*st^2*cos(Db)

nu2=m11*m22-m12*m21
%%
collect(simplify(expand(nu2)), 2*phi1 - 2*phi0 + phia - phib)
%%
a1=simplify(subs(simplify(expand(nu2)), 2*phi1 - 2*phi0 + phia - phib, Y))
%% A= 1-cos(4 theta)
a2= collect(simplify(subs(a1, cos(4*theta),1-A)),A)
%%
a3=subs(a2,cosh(ra)^2,(c2a+1)/2)
%%
a4=subs(a3,cosh(rb)^2,(c2b+1)/2)
%%
a5=subs(a4,sinh(rb)*cosh(rb)*sinh(ra)*cosh(ra),s2a*s2b/4)
%%
collect(simplify(a5),A)