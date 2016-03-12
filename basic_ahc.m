%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------------------------------------------
% Project <Adaptive Huristic Critic algorithm>
% Date    : 2015/12/23
% Author  : Kun da Lin
% Comments: Language: Matlab. 
% Source: matlab 
% This is a short example to implement AHC in inverted pendulum
% the key idea is to collect td-error and refresh with ACE also ASE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set parameters
lambdaw = 0.9;
lambdav = 0.8;
alpha = 1000;
beta = 0.5;
gamma =0.95;

failures=0;
failed=0;
step=0;
episode=0;
%% set ACE & ASE weights , eligibilities to 0
w=zeros(162,1);
v=zeros(162,1);
x_bar=zeros(162,1);
e=zeros(162,1);

store=zeros(1,5);
%% initialization
x=0;
x_dot=0;
theta=0;
theta_dot=0;

[ box,x_encoder ] = get_box( x,x_dot,theta,theta_dot );
while(step<1000000)
    step=step+1;
random_number=rand;
%prob=prob_push_rights(w(box+1,1));
% y: 0==>push left or 1==>push right
y = (random_number<prob_push_rights(w(box+1,1)));                      %owing to box and encoder
% get next state
[ new_x,new_x_dot,new_theta,new_theta_dot ] = simulation( y,x,x_dot,theta,theta_dot );
[ next_box,next_x_encoder ] = get_box( new_x,new_x_dot,new_theta,new_theta_dot );

% next_box:-1  mean failed
    if(next_box<0)
        failed=1;
        failures=failures+1;
        reward=-1;
        p=0;
        %reset to initial condition
        x=0;
        x_dot=0;
        theta=0;
        theta_dot=0;

        [ box,x_next_encoder ] = get_box( x,x_dot,theta,theta_dot );
        
        episode=episode+1;
        disp(['epidoe: ' num2str(episode) ' step: ' num2str(step)]);
        %store information
        store(1,episode)=step;
        %reset step
        step=0;          
 % keep going       
    else
        failed=0;
        reward=0;
        p=sum(v.*next_x_encoder);                  %Compute new_p
        box=next_box;
        
        % refresh environment
        x=new_x;
        x_dot=new_x_dot;
        theta=new_theta;
        theta_dot=new_theta_dot;      
        
    end
%% ACE
% x_encoder: previous encoder(state)
x_bar = x_bar+(1-lambdav)*x_encoder;  %Update ACE eligibility:
old_p=sum(v.*x_encoder);              %Compute old_p
%%  r_hat:TD Error   The KEY FUNCTION
% p=sum(v.*next_x_encoder) 
% Not only the reward have to be good but also have to improve in the future
r_hat = reward + gamma*p-old_p;       %Compute r_hat  
% x_bar:collect how many times you visit the state
% if you always visit it and your r_hat is positve then you will strenthen
% your ACE weights(v)
v=v+beta*r_hat*x_bar;                 %Update ACE weights v

%% ASE
e=e+(1-lambdaw)*(y-0.5)*x_encoder;    %Update ASE eligibility
w=w+alpha*r_hat.*e;                   %Update ASE weights w

%% decide action
%y=sum(w.*x_encoder);
x_encoder=next_x_encoder;             % next_x_encoder become old

    if(failed==1)
        x_bar=zeros(162,1);
        e=zeros(162,1);
    else
        e=e*lambdaw;
        x_bar=x_bar*lambdav;
    end

end
store(1,end)=1000000;
disp('step>1000000 success_foreverXD');
plot(store);

