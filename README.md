# Deep Autoencoder with ELM
## Deep Autoencoder with Extreme Learning Machines

<div>
The script helps to train your own Deep Autoencoder with Extreme Learning Machines. <br>
It performs a Deep Autoencoder model with with a specified model. 

After that, it utilizes both Neural Networks and Extreme Learning to compare the efficiency of machine learning algorithms.

<img src="http://www.gokhanaltan.com/wp-content/uploads/DeeAEwithELM.png">

Whereas the first encoder weight matrix of the Deep Autoencoder is unfolded as the first weight matrix of the neural network model, <br>
The ELM is fed with the middle autoencoder layer that sustains significant and compressed representation of Deep Autoencoder.

Example training procedure of Deep Autoencoder with Extreme Learning Machines is in DeepAEwithELM_MNIST.m with the instructons how to easily run a training on MNIST dataset. 

It can be easily modified for your own data. The script is modified from [1]. 

## Example of Deep Autoencoder with Extreme Learning Machines on MNIST

The example training is experimented on MNIST database:
```
load mnist_uint8;
%Preprocessng for normalization of image pixels
train_x = double(train_x)/255;
%train_x:  [sample_size x feature_size]
test_x  = double(test_x)/255;
%test_x:  [sample_size x feature_size]
train_y = double(train_y);
%train_y:  [sample_size x classes ]
test_y  = double(test_y);
%test_y:  [sample_size x classes ]

%  Setup and train a Deep AE
rand('state',0)
DeepAE_ELM = saesetup([784 100 40 30 40 100]);
%Create your DeepAE model [feature_size AE(1) AE(2) CompressedAE AE(2) AE(1)]
DeepAE_ELM.name='SAE';
DeepAE_ELM.ae{1}.learningRate              = 0.01;
%Set your learning rate for DeepAE
DeepAE_ELM.ae{1}.inputZeroMaskedFraction   = 0.5;

opts.numepochs =   10;
%Set your epoch number for DeepAE
opts.batchsize = 100;
%Set your batch size for DeepAE (must be divisible by the number of samples)


    %The script enables training  DeepAE_ELM for the both sigmoid and hiperbolic tangent function at AE stage   
    
    DeepAE_ELM.ae{1}.activation_function       = 'sigm';   % 'sigm', 'tanh_opt'
    %Set activation function for DeepAE_ELM    
    DeepAE_ELM = saetrain(DeepAE_ELM, train_x, opts);
    %Train DeepAE_ELM
    visualize(DeepAE_ELM.ae{1}.W{1}(:,2:end)')
    %Visualize the predicted autoencoder output for desired layer
    
    %TRAIN model using Neural Networks 
    
    %The script enables testing the model with both sigmoid and hiperbolic tangent function at supervised stage   
        
        nn = nnsetup([784 100 10]);
        %Create your neural network model [feature_size HiddenLayer1 HiddenLayer2 .... classes ]
        % Use the DeepAE to initialize a Feedforward neural networks model
        nn.name=   strcat('SAE-' ,'sigm');         % 'sigm', 'tanh_opt'
        nn.activation_function              = 'sigm';
        %Set activation function for Neural networks
        nn.learningRate                     = 1;
        %Set your learning rate for neural networks
        nn.W{1} = DeepAE_ELM.ae{1}.W{1};
        %Set Neural network weights using DeepAE pre-trained weights

        % Train stage of the Feedforward Neural Networks
        opts.numepochs =   100;
        %Set your epoch number for neural network
        opts.batchsize = 100;
        %Set your batch size for Neural network (must be divisible by the number of samples)
        nn = nntrain(nn, train_x, train_y, opts);
        %Train Neural Networks model
        [error_NN, bad] = nntest(nn, test_x, test_y);
        %Test Neural Networks model with your separate testing data
        

    
    %TRAIN the compressed layer of DeepAE using Extreme Learning Machines with Moore Penrose
    %ELM has a single hidden layer 
    ELM_neuronsize_min=10;
    ELM_neuronsize_max=200;
    ELM_neuronsize_increment=10;
    %Set neuron number of hidden layer in ELM
    
    
    %Get the compression layer of DeepELM as input to ELM
    train_x_ELM=[];
    for i=1:(size(DeepAE_ELM.ae,2)+1)/2
        if i==1
             train_x_ELM=train_x*DeepAE_ELM.ae{i}.W{1}(:,1:end-1)';
        else
             train_x_ELM=train_x_ELM*DeepAE_ELM.ae{i}.W{1}(:,1:end-1)';
        end        
    end
    
    %TRAIN Extreme Learning Machine for various neuron numbers in the hidden layer
    for ELM_neuronsize=ELM_neuronsize_min:ELM_neuronsize_increment:ELM_neuronsize_max
        [error_ELM(ELM_neuronsize/ELM_neuronsize_increment)]=ELM(train_x_ELM, train_y,ELM_neuronsize);
         %Start training of ExtremeLearnin Macines with Moore Penrose
    end  
    
    
```


Please cite this article : <b>Gokhan ALTAN, SecureDeepNet-IoT: A Deep Learning application for Invasion Detection in IIoT sensing systems </b>
*The paper is in review process, the details and optimum weight will be added after publication.

[1] R. B. Palm, Prediction as a candidate for learning deep hierarchical models of data, 2012, Master Thesis, https://github.com/rasmusbergpalm/DeepLearnToolbox

</div>
