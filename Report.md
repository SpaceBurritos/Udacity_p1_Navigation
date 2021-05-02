The arquitecture used for the DNN consisted of 1 input layer and  3 deep layers (including the output layer) \n
The number of nodes from the input layer to the output layer were: \n
  37 (state_size) -> 256 \n
  256 -> 512 \n
  512 -> 512 \n
  512 -> action_size \n
  
For all the layers, excluding the output layer, a ReLU activaiton function was used, and for getting the probability distribution of the actions a softmax function was used \n

The rest of the hyperparameters used for the project were:\n
  epsilon -> 1 \n
  epsilon decay -> 0.995 \n
  gamma -> 0.99 \n 
  Tau -> 1e-3 \n 
  Learning rate -> 5e-4 \n
  Batch size -> 64 \n
  Buffer Size -> 1e5 \n
