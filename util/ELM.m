function [error]=ELM(train_x,train_y,number_neurons)
% ELM: this function allows to train Singe layer feedforward neural Network
% ELM uses Moore-Penrose pseudoinverse of matrx.


input_weights=rand(number_neurons,size(train_x,2))*2-1;
% Generate random input weights
H=radbas(input_weights*train_x');
%Calculate the value of Hidden layer
%%%% 3rd step: calculate the output weights beta
B=pinv(H') * train_y ; 
%Calculate the output weights using Moore-Penrose pseudoinverse

output=(H' * B)' ;
% Calculate the actual output 
error=sqrt(mse(train_y'-output));
%Root mean sequared error
disp(sprintf('-Training error for %d neurons in ELM: %f ', number_neurons, error));
end