clear all
close all
Kp=4;%0.6;
Kv=0.02;%0.01;
F_step=0.1;
deltaj=1%.5;
deltab=1%.2;
J=1.0632;%0.005*deltaj;
B=3.2264;%0.1*deltab;
tao=0.01;%0.01;%0.15;0.18
ts=0.001;
gnnum=[0 0 1];
gnden=[J B 0];
qnum=[3*tao,1];
qden=[tao*tao*tao,3*tao*tao,3*tao,1];
num1=conv(gnden,qnum);
den1=conv(gnnum,qden);
[qznum,qzden]=c2dm(qnum,qden,ts,'tustin');
[g2num,g2den]=c2dm(num1,den1,ts,'tustin');
sysq=tf(qznum,qzden,ts);
sysg2=tf(g2num,g2den,ts);
%lqnum=[4175.18064782253900,-12402.82692341595400,8044.36592243045110,8533.11238142945510,-12463.91689828522400,4114.08487634883390];
%lqden=[1.00000000000000,-1.99999255896952, 0.99999888907540,0,0,0];
lqnum=100000*[1.32590507338253,-3.97228953832331,2.63827687269558,2.65987538756506,-3.97494857824138,1.32318078295649];
lqden=[ 1,-1.99997186724381,0.99997536501425,0,0,0];