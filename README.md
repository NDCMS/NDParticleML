# NDParticleML
This a reposity with modifications made to the simple neural network found [here](https://colab.research.google.com/drive/1wpLKRUaBxlfWDmyL9czZ8WP-_4W60Vxh?usp=sharing). This model is tries to learn the function f(x)=x for a set of 3 inputs. 

### To-Do List
- [ ] Get one notebook with the following implemented in GH [Sean]
-  - [x] Run the training example(s) on the GPU 
-  - [x] Make plots that show how well the network is reproducing the outputs that we expect using a number of randomly generated test points
-  - [x] Make loss curves for training (training and testing)
-  - [ ] Find the number of hidden nodes and number of epochs to train to achieve either Δ = 0.01 and/or 1% relative accuracy
-  [ ] Add some code to handle standardization of inputs and outputs (and then test it on examples where the inputs have different distributions) [Carter]
-  [ ] Make a plot of the weights in the model [Evan]
-  [ ] Make another notebook that does some other function (like f(x,y) = x*y, etc.) [Sirak]
-  [ ] Repeat study of number of nodes and epochs for other input distributions (e.g. uniform, exponential)
