The arquitecture used for the DNN consisted of 1 input layer and  3 deep layers (including the output layer).  
The number of nodes from the input layer to the output layer were:  
  37 (state_size) -> 64</br> 
  64 -> 128</br>
  128 -> 128</br>
  128 -> action_size</br>
  
For all the layers, excluding the output layer, a ReLU activaiton function was used, and for getting the probability distribution of the actions, a softmax function was used. 

The rest of the hyperparameters used for the project were:  
  epsilon -> 1</br>
  epsilon decay -> 0.995</br>
  gamma -> 0.99</br>
  Tau -> 1e-3</br>
  Learning rate -> 5e-4</br>
  Batch size -> 64</br>
  Buffer Size -> 1e5</br>

The model got an average of over 13 points after 1400 iterations:
![GitHub Logo](/resources/learningGraph.png)
