[ new_x(1,1),new_x_dot(1,1),new_theta(1,1),new_theta_dot(1,1) ] = simulation( 1,0,0,-0.1,0 );

for (i=1:1:5000)
 
    
[ new_x(i+1,1),new_x_dot(i+1,1),new_theta(i+1,1),new_theta_dot(i+1,1) ] = simulation( 1,new_x(i,1),new_x_dot(i,1),new_theta(i,1),new_theta_dot(i,1) );

end