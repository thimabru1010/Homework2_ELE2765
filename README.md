# Homework 2 ELE2765
Homework 2 of Deep Learning subject on Electrical Engineering Master's at PUC-Rio

<p align="center">
  <img src="feature_levels.png" alt="Sublime's custom image"/>
</p>

This homework has the objective to do a transfer learning in Resnet50 using three different approaches. The dataset used was flowers recognition dataset.

In The first approach, question 5, I just unfrozen the last block of convolution in Resnet50 (conv_5_block3). I used a small amount of dropout (0.25) just before softmax. In this approach, Models 1-3 was produced.

In the second approach, question 6, was demanded to unfrozen another convolutional block (conv5_block2), besides the previously one, and add a fully connected layer before softmax layer. In this approach, Models 4-9 was produced.

In the third approach, question 7, was demanded to to use the last three convolutional blocks untrained (conv5_block3, conv5_block2, conv5_5_block1). to do this, I reimplemented these blocks of resnet in function `create_custom_model_q7` inside `train_transfer_learning.py` script. In this approach, Models 10-14 was produced.
