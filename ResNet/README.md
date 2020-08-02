## Resnet

34 Layer residual network as proposed in https://arxiv.org/pdf/1512.03385.pdf. We applied the activation function after batch normalization and skip connection additions, as this seems to be the standard which is different from the paper. 
The paper mentioned two methods for reshaping the size of the skip connections when needed after a maxpooling: Convolutions and zero paddings.
We reduced the dimension and increased the filter size of the skip connection feature maps by using a 1x1 convolution with a stride of 2.

![alt text](https://miro.medium.com/max/772/1*nNZMmh3G6uQDszsB2PQmOw.png)
