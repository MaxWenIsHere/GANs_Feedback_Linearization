%clear all;
%close all;
load f.dat;
load time.dat;
load input.dat;
load output.dat;
tSamp=0.001;
l=1;
m=size(f);
for k=1:1:m-1
    f1=f(k);
    f2=f(k+1);
   if f2==f1
      fout(l)=f1;
  else l=l+1;
  end
end
% result[2]=f;
a11=0; a12=0; a21=0; a22=0; b1=0; b2=0; c1=0; c2=0;
a11_1=0; a12_1=0; a21_1=0; a22_1=0; b1_1=0; b2_1=0;
for m=1:l
    w(m)=2.0*pi*fout(m);
    point=20000;
    for i=10:(point-18000-10)
        
			     a11=a11_1+cos(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
				 a12=a12_1+cos(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
			     a21=a21_1+sin(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
			     a22=a22_1+sin(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
			     b1=b1_1+cos(w(m)*i*tSamp)*input(20000*(l-1)+i);
			     b2=b2_1+sin(w(m)*i*tSamp)*input(20000*(l-1)+i);
                 a11_1=a11;
                 a12_1=a12;
                 a21_1=a21;
                 a22_1=a22;
                 b1_1=b1;
                 b2_1=b2;
                 
     end
        if(abs(a11*a22-a12*a21)>1e-5)
	     
	      c1=(a22*b1-a12*b2)/(a11*a22-a12*a21);
	      c2=(a11*b2-a21*b1)/(a11*a22-a12*a21);
	      Pin=atan2(c1,c2);
	      Ain=sqrt(c1*c1+c2*c2);
          Angle_in(m)=Pin;
          Amp_in(m)=Ain;
          
        else
                Angle_in(m)=0;
                Amp_in(m)=0;
                
       end
end
  
a11=0; a12=0; a21=0; a22=0; b1=0; b2=0; c1=0; c2=0;
a11_1=0; a12_1=0; a21_1=0; a22_1=0; b1_1=0; b2_1=0;

for m=1:l
    w(m)=2.0*pi*fout(m);
    point=20000;
    for i=10:(point-18000-10)
 	   a11=a11_1+cos(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
 	   a12=a12_1+cos(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
 	   a21=a21_1+sin(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
 	   a22=a22_1+sin(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
 	   b1=b1_1+cos(w(m)*i*tSamp)*output(20000*(l-1)+i);
 	   b2=b2_1+sin(w(m)*i*tSamp)*output(20000*(l-1)+i);
       a11_1=a11;
       a12_1=a12;
       a21_1=a21;
       a22_1=a22;
       b1_1=b1;
       b2_1=b2;
     end
     if (abs(a11*a22-a12*a21)>1e-5)
  	      c1=(a22*b1-a12*b2)/(a11*a22-a12*a21);
   	      c2=(a11*b2-a21*b1)/(a11*a22-a12*a21);
   	      Angle_out(m)=atan2(c1,c2);
   	      Amp_out(m)=sqrt(c1*c1+c2*c2);
     else
 	           Angle_out(m)=0.0;
 	           Amp_out(m)=0.0;
     end
 end
for n=1:l
 if (abs(Amp_in)>1e-5)
	    Amp(n)=20*log10(abs(Amp_out(n)/Amp_in(n)));
 else
	       Freq(n)=0.0;
	       Angle(n)=0.0;
	       Amp(n)=0.0;
 end 
Freq(n)=fout(n);
Angle(n)=(Angle_out(n)-Angle_in(n))*180/pi;
end

fidf=fopen('freq.dat','w');
for j=1:l
fprintf(fidf,'%f\t%f\t%f\n',Freq(j),Amp(j),Angle(j));
end
fclose(fid);

