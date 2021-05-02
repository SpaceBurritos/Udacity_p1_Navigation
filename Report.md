The arquitecture used for the DNN consisted of 1 input layer and  3 deep layers (including the output layer)
The number of nodes from the input layer to the output layer were:
  37 (state_size) -> 256
  256 -> 512
  512 -> 512
  512 -> action_size
  
For all the layers, excluding the output layer, a ReLU activaiton function was used, and for getting the probability distribution of the actions a softmax function was used

The rest of the hyperparameters used for the project were:
  epsilon -> 1
  epsilon decay -> 0.995
  gamma -> 0.99
  Tau -> 1e-3
  Learning rate -> 5e-4
  Batch size -> 64
  Buffer Size -> 1e5
