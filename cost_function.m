function [J] = cost_function(training_ex, y, hyp, lambda, theta, input_layer_size, hidden_layer_size, classes)

%finds cost function value (used in first_NN_sgd.m)

%reshape certain parts of theta into w2 and w3. put in this to fit the
%order of w1, w2,... properly.

w2 = reshape(theta(1:((input_layer_size+1)*hidden_layer_size)),input_layer_size+1,hidden_layer_size)';    
w3 = reshape(theta(((input_layer_size+1)*hidden_layer_size+1):end),hidden_layer_size+1,classes)';
    
%cost function 
J = (-1/training_ex)*sum(sum(y.*log(hyp')+(1-y).*log(1-hyp'),2)) + (lambda/(2*training_ex))*(sum((w2(:,2:end)).^2,'all') + sum((w3(:,2:end)).^2,'all'));

end
    
    
    