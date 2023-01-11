---
layout: default
title: Assignment 12
id: ass12
---


# Assignment 12: Introspection Part 1
**Deadline: January 16th, 9am**


In this assignment, you will implement gradient-based model analysis both for
 creating saliency maps (local) and for feature visualization (global). 
 You can adapt your implementation of the adversarial examples of Assignment 10 
and also take inspiration from the 
[DeepDream tutorial](https://www.tensorflow.org/tutorials/generative/deepdream).
It is recommended that you work on image data as this makes visual inspection of
the results simple and intuitive.

You are welcome to use pre-trained ImageNet models from the `tf.keras.applications`
module. The tutorial linked above uses an Inception model, for example. You can
also train your own models on CIFAR or something similar.

## Gradient-based saliency map (sensitivity analysis)

Run a batch of inputs through the trained model.
Wrap this in a GradientTape where you watch the input batch
(batch size can be 1 if you'd like to just produce a single saliency map).
and compute the gradient for a particular logit or its softmax output _with 
respect to the input_.
This tells us how a change in each input pixel would affect the class output.
This already gives you a batch of gradient-based saliency maps!
Plot the saliency map next to the original image or superimpose it.
Do the saliency maps seem to make sense? How would you interpret them?

- It makes sense to take the sign of the gradient into account when 
interpreting them.
Negative gradients indicate a decrease in output value, positive 
gradients an increase. This means you should use a _diverging_ colormap with 
  separate colors for positive and negative values for plotting.
- Alternatively, maybe using absolute values of the gradient and a _sequential_ colormap might
  make more sense. What do you think?
- You can try smoothing the saliency maps, e.g. with a gaussian filter. This will
generally make them look "better", but also falsifies the actual information somewhat.

## Activation Maximization
Extend the code from the previous part to create an optimal input for a 
particular class.

Start with a _randomly initalized image_, not one from the dataset (although you _could_ also
use a dataset image as a starting point).
Multiply the gradients with a small constant (like a learning rate) and add them
to the input.
Repeat this multiple times, computing new gradients with respect to the input each
time.
Essentially, you are writing a "training loop" for producing an optimal input for
a certain class (do _not_ train the model weights!).  
**Note:** You need to take care that the optimized inputs actually stay valid images
throughout the process, e.g. by clipping to [0, 1] after each gradient step, or by
using a sigmoid function to produce the images.


Does the resulting input look natural?
How do the inputs change when applying many steps of optimization?
How do the optimal inputs differ when initializing the optimization with random 
noise instead of real examples?
Can you see differences between optimizing a logit or a softmax probability?

**Bonus**: Apply regularization strategies to make the optimal input more 
natural-looking.
You can also optimize for _hidden features_ of the network (instead of outputs)
assuming you can "extract" them from the model you built. Distill has 
[an article](https://distill.pub/2017/feature-visualization/) that can provide
some inspiration.
