load f.dat;
load time.dat;
load input.dat;
load output.dat;
%float a11,a12,a21,a22,b1,b2,c1,c2,w,tSamp;
%float Ain,Pin,Af,Pf;

tSamp=0.001;
t=time;
%persistent k;
%global T;
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
 m=1
    w(m)=2.0*pi*fout(m);
    point=20000;
    for i=10:(point-10)
        
%			     a11=a11_1+cos(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
                 a11=a11_1+cos(w(m)*t(20000*(l-1)+i))*cos(w(m)*t(20000*(l-1)+i));
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
     