%clear all;
%close all;

load freq.dat;
tSamp=0.001;
f=freq(:,1);

mag=freq(:,2);
ph=freq(:,3);
for i=1:length(ph)
   if ph(i)>0  ph(i)=ph(i)-360;
   end
end


w=f*2*pi;
mag1=10.^(mag/20);
ph1=ph*pi/180;
h=mag1.*cos(ph1)+j*mag1.*sin(ph1);

dly = -0;
h_dly = exp(-j*w*dly*0.001);

hh = h./h_dly;
hhmag = abs(hh);
hhph = angle(hh);
hhmag1=20*log10(hhmag);
hhph1=hhph*180/pi;




na=4; nb=2;
[bb,aa] = invfreqs(hh,w,nb,na);
sysx=tf(bb,aa);
[zs,ps,ks]=zpkdata(sysx,'v');



sysxd=c2d(sysx,tSamp,'zoh');

zpksys=zpk(sysxd)
[z,p,k]=zpkdata(zpksys,'v')
tfsys=tf(sysxd);


h = freqs(bb,aa,w);
sys1mag = abs(h);
sys1ph = angle(h);
sys1mag1=20*log10(sys1mag);
sys1ph1=sys1ph*180/pi;

figure(5);
subplot(2,1,1); semilogx(w,sys1mag1,'r');
hold on; semilogx(w,mag,'b'); grid on;
%hold on; semilogx(w,hhmag1,'g'); grid on;

subplot(2,1,2), semilogx(w,sys1ph1,'r');
hold on; semilogx(w,ph,'b'); grid on;
%hold on; semilogx(w,hhph1,'g'); grid on;

magError=sys1mag1-mag;
phError=sys1ph1-ph;
figure(6);
plot(w/2/pi,phError,'r',w/2/pi,magError,'b'); grid on;
legend('phError','magError')
% 
% zu=z(1);
% za=z(2:3);
% 
% 
% z1=[p;1/zu];
% p1=za;
% 
% 
% systemp=zpk(z1,p1,1,tSamp);
% k1 = dcgain(systemp);
% systemp = series(systemp,1/k1);
% 
% 
% [numf,denf]=tfdata(systemp,'v');
% 
% denf = denf(4:6);
% zpetc_filt = filt(numf,denf);
% 
% figure;
% step(zpetc_filt);
% fid=fopen('zpetci.dat','w');
% for i=1:1:length(numf)
%    fprintf(fid,'%5.14f\n',numf(i));
% end
% fprintf(fid,'%5.14f\n',0);
% for i=2:1:length(denf)
%    fprintf(fid,'%5.14f\n',denf(i));
% end
% fclose(fid);
% format long;
% numf
% denf


