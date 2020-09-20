function [gradient, hyp] = stochastic_gradient_descent(X, theta, y, lambda, k, input_layer_size, hidden_layer_size, classes)

    %stochastic grad. descent process (used in first_NN_sgd.m)
    
%reshape certain parts of theta into w2 and w3. put in this to fit the
%order of w1, w2,... properly.

w2 = reshape(theta(1:((input_layer_size+1)*hidden_layer_size)),input_layer_size+1,hidden_layer_size)';
w3 = reshape(theta(((input_layer_size+1)*hidden_layer_size+1):end),hidden_layer_size+1,classes)';
    
%find number of training ex. 
[m,~] = size(X);
    
z2 = w2*[ones(1,size(X',2)); X']; %computing z by multiplying weights and input layer neurons with bias
a2 = sigmoid_func(z2); %activations of hidden layer
    
z3 = w3*[ones(1,size(a2,2)); a2]; %computing z by multi. weights and input layer neurons with bias
a3 = sigmoid_func(z3); %activations of final layer
hyp = a3; %hypothesis equals activation of final layer
    
a2 = [ones(1,size(a2,2)); a2]; %adding bias to activations of hidden layer
    
%partial C wrt act. vals of final layer. 

a_partial3 = (a3' - y);
    
%partial C wrt act. vals of hidden layer, with bias unit sinze input layer isnt connected to bias unit
    
a_partial2 = (a_partial3*w3)'.*(a2.*(1-a2));
    
%removing bias unit from a_partial2
    
a_partial2 = a_partial2(2:end,:);
    
%transposing for right size
    
a_partial3 = a_partial3';
        
%bias to data 

X_temp = [ones(1,size(X',2)); X']';
    
%set size of deltas. size(1) is num of act. vals. in front layer bc thats the amt of connections and size(2) is +1 of prev layer bc of bias
    
delta1 = zeros(hidden_layer_size,input_layer_size+1);
delta2 = zeros(classes,hidden_layer_size+1);
    
%k-th division of num of training ex. ceil() used in case decimal val
    
e = ceil(m/k);

%first lower bound in loop
s = 1;
    
%stochastic GD
    
%runs k times since k divisions
for i = 1:k
    %set quantity to final_term of inner for loop (below)
    final_term = i*e;

    %when i reaches k, then final_term is equal to m regardless of i*e.
    if i == k
        final_term = m;
    end
    
    %bounds are flexible based on the k-value. change to have mini batches.
    for g = s+e*(i-1):final_term
        %adding nudges and desires.
        delta1 = delta1 + (a_partial2(:,g)*X_temp(g,:));
        delta2 = delta2 + (a_partial3(:,g)*a2(:,g)');
    end
end
    
%take avg of all nudges and add regularization
D1 = ((1/m)*(delta1 + lambda*[zeros(1,size(w2,2)); w2(2:end,:)]))';
D2 = ((1/m)*(delta2 + lambda*[zeros(1,size(w3,2)); w3(2:end,:)]))';
    
%unroll for gradient.
gradient = [D1(:);D2(:)];

end