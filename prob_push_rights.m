function [ prob ] = prob_push_rights( s )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
temp=min(s,50.0);
temp=max(-50.0,temp);

prob=1/(1+exp(-temp));

end

