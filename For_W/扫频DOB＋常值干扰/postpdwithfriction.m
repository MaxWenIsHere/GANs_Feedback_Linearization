%close all
figure(1)
plot(t,r,'-',t,theta,'r:')%,t,thetadob,'-.');
grid on;
% figure(2)
% plot(t,r,'-',t,thetam,':',t,thetadob,'-.');
% legend('command','thetam','thetadob')
% xlabel('time(sec.)')
% ylabel('degree')
% title('r/thetam/thetadob')
% figure(3)
% plot(t,-d_est,'-',t,fc,'r:');
% legend('-d^','f')
% xlabel('time(sec.)')
% %ylabel('degree')
% title('-d^/f')
% figure(4)
% plot(dtheta,fc)
% xlabel('dtheta')
% ylabel('f')
% title('dtheta-f')
% legend('r','theta')
% xlabel('time(sec.)')
% ylabel('degree')
% title('r/theta')

fid=fopen('f.dat','w');
fprintf(fid,'%f\n',f(1:100000));
fclose(fid);
fid1=fopen('time.dat','w');
fprintf(fid1,'%f\n',t(1:100000));
fclose(fid1);
fid2=fopen('input.dat','w');
fprintf(fid2,'%f\n',r(1:100000));
fclose(fid2);
fid3=fopen('output.dat','w');
fprintf(fid2,'%f\n',theta(1:100000));
fclose(fid3);