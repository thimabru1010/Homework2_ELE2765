# Homework2_ELE2765
Homework 2 of Deep Learning subject on Electrical Engineering Master's at PUC-Rio

This homework has the objective to do a transfer learning in Resnet50 using three different approaches.

In transfer learning we usually upadte with a low learning rate the last layers of the network, therefore, in The first approach I just unfrozen the last block of convolution in Resnet50 (conv_5_block3). I used a small amount of dropout (0.25) just before softmax. 

In the second approach 
