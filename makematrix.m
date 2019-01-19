% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

load moviedata1M_movies

num_m = length(unique([train_vec(:,2); probe_vec(:,2)]));  % Number of movies 
num_p = length(unique([train_vec(:,1); probe_vec(:,1)]));  % Number of users 
count = zeros(num_p,num_m,'single'); %for Netflida data, use sparse matrix instead. 

for mm=1:num_m
 ff= find(train_vec(:,2)==mm);
 count(train_vec(ff,1),mm) = train_vec(ff,3);
end 


