clc
clear
close

%finds ideal parameters for neural network thru stochastic grad. descent

classes = 10;  %set final layer size 
input_layer_size = 784;  %set input layer size 
hidden_layer_size = 25; %set hidden layer size 
lambda = 3;

X = load('mnist_train.csv'); %load data (training ex. by features)
y = X(:,1); %
X = X(:,2:end);%

k = 10; %how many parts to split training data into for sto. grad. desc.
max_iter = 250; %set max iterations.

[m,~] = size(X); %set size

y_rev = zeros(size(y,1),classes);

%1 in col of y and rest 0
for i = 1:size(y,1)
    y_rev(i,y(i)+1) = 1;
end

%find initiale parameters. check rand_innit. finds -eps<=x<=eps
w2 = rand_init(hidden_layer_size, input_layer_size); %weights connecting 1 -> 2
w3 = rand_init(classes, hidden_layer_size); %weights connecting 2 -> 3

%transpose so that once vectorized, weights are w1, w2,... order
w2 = w2';
w3 = w3';

J = zeros(max_iter+1, 1);  %great J vector to see if J is decreasing.
J(1) = 0; %first is initialized to 0.

%unroll into theta
theta = [w2(:);w3(:)];
%intialize initial gradient to 0 vec for convenience.
gradient = zeros(size(theta,1),1);

for i = 1:max_iter
    %remove gradient from theta
    theta = theta - gradient;
    
    %find hypothesis of that same theta found above.
    %find gradient at that theta found above.
    [gradient, hyp] = stochastic_gradient_descent(X, theta, y_rev, lambda, k, input_layer_size, hidden_layer_size, classes);
    
    %find cost function of that theta found above and store in J.
    J(i + 1) = cost_function(m, y_rev, hyp, lambda, theta, input_layer_size, hidden_layer_size, classes);
    
    %find column-wise in which row is the greatest val for the final act. vals.
    [~,hyp] = max(hyp,[],1);
    
    %transpose to match dims with y.
    hyp = hyp';
    %find acc between hypothesis and y.
    acc = mean(double(hyp == y)) * 100;
end
