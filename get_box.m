function [ box,x_encoder ] = get_box( x,x_dot,theta,theta_dot )

%box from 0 to 161
%we add one to x_encoder(box+1)=1;
%x_encoder all zero if failed
%%%%%%%%%
x_encoder=zeros(162,1);
one_degree=0.0174532;
six_degree=0.1047192;
twelve_degree=0.2094384;
fifty_degree=0.87266;

box=0;
%%%%%%%%%%%%%%%%%
if(x<-2.4 || x>2.4 || theta<-twelve_degree || theta>twelve_degree)
    box=-1;
    return;
end
%%%%%%%%%%%%%%%%%
if(x<-0.8)
    box=0;
elseif(x<0.8)
    box=1;
else
    box=2;
end
%%%%%%%%%%%%%%%%%
if(x_dot<-0.5)
    box=box;
elseif(x_dot<0.5)
    box=box+3;
else
    box=box+6;
end
%%%%%%%%%%%%%%%%%
if(theta<-six_degree)
    box=box;
elseif(theta<-one_degree)
    box=box+9;
elseif(theta<0)
    box=box+18;
elseif(theta<one_degree)
    box=box+27;
elseif(theta<six_degree)
    box=box+36;
else
    box=box+45;
end
%%%%%%%%%%%%%%%%%%
if(theta_dot<-fifty_degree)
    box=box;
elseif(theta_dot<fifty_degree)
    box=box+54;
else
    box=box+108;
end
%%%%%%%%%%%%%%    

%%%%%%%%%%%%%%

     x_encoder(box+1,1)=1;

end

