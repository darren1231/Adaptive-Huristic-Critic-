function [ new_x,new_x_dot,new_theta,new_theta_dot ] = simulation( action,x,x_dot,theta,theta_dot )

global gravity; 
gravity=9.8;
masscart=1.0;
masspole=0.1;
total_mass=masscart+masspole;
length=0.5;
polemass_length=masspole*length;
tau=0.02;
fourthirds =1.333333333;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(action>0)
    f=10;
else
    f=-10;
end

% f=0;

costheta = cos(theta);
sintheta = sin(theta);
temp = (f + polemass_length* theta_dot*theta_dot*sintheta)/total_mass;
theta_acc=(gravity*sintheta-costheta*temp)/(length*(fourthirds-masspole*costheta*costheta/total_mass));
x_acc= temp-polemass_length*theta_acc*costheta/total_mass;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     update four variables
new_x = x + tau*x_dot;
new_x_dot = x_dot + tau*x_acc;
new_theta= theta + tau*theta_dot;
new_theta_dot=theta_dot+ tau*theta_acc;
end

