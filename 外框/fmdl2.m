function ff=fmdl2(uu)

dtheta=uu(1);
u=uu(2);

fc_n=-1.95;%-0.18;
fc_p=1.9;%0.15;
fs_n=-2.55;%-0.25;
fs_p=2.5;%0.2;
dv=0.1;

if ((abs(dtheta)<=dv)&((u>=fs_n)&(u<=fs_p)))
   ff=u;%-0.18;
elseif ((abs(dtheta)<=dv)&(u<=fs_n))
   ff=fs_n;%friction_p;%0.15;%
elseif ((abs(dtheta)<=dv)&(u>=fs_p))
   ff=fs_p;%0;
elseif dtheta<-dv
   ff=fc_n;
else
   ff=fc_p;   
end
