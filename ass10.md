---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: Adversarial Examples & Training
**Deadline: December 19th, 9am**

In this assignment, we will explore the phenomenon of adversarial examples and
how to defend against them. This is an active research field and somewhat poorly
understood, but we will look at some basic examples.


## Creating Adversarial Examples

Train a model of your choice model on "your favorite dataset". It should be an
image classification task
(e.g. MNIST... although a more complex dataset such as CIFAR should work better!).
Alternatively, see the section below on using pretrained models.
 Next, let's create some adversarial examples:
- Run a batch of inputs through the trained model. Wrap this in a `GradientTape`
where you _watch the input batch_.
- Compute the gradient of the classification loss with respect to the inputs.
This tells us how to change the input such that we maximally increase the loss.
  - The above is an "untargeted" attack -- we simply care about increasing the
  loss as much as possible. You could also try _targeted_ attacks where the goal
  is to make the network misclassify an input in a specific way -- e.g. classify
  every MNIST digit as a 3. Think about how you could achieve this, and feel free
  to try it out.
- Add the gradients to the inputs. This will give you a batch of adversarial
examples!
 You will probably want to multiply the gradients
with a small constant (kind of like a learning rate) to make sure the inputs
aren't changed too much -- that would defeat the purpose! At the same time, the
change needs to be large enough to actually affect the classification. 
  - Instead of multiplying by a fixed constant, a better option rescale the 
  gradients such that the maximum value is equal to some constant, or that the
  length (euclidean norm) of the gradient vector is equal to some constant.
  - Another possibility is the "gradient sign" method, where you take the 
  `tf.math.sign` of the gradient first, _then_ multiply by a small constant.
  - Feel free to come up with other schemes to make use of the gradient information.
- In principle, you can take multiple loss/gradient steps, but one should suffice
for now.
- To be perfectly accurate, you should make sure your adversarial examples are
in the normal data range, e.g. clip images to be in [0, 1] after adding the gradient.
- Run these examples through your model. It should do significantly worse at
classifying them than for the original data! You could create adversarial examples
  for the entire test set and then use `model.evaluate` to compare to the performance
  on the original test set.

Hopefully, you are able to "break" your models somewhat reliably, but you don't
have to expect 100% success rate with your attacks.


## Adversarial Training

Depending on your viewpoint, adversarial examples are either really amazing or
really depressing (or both). Either way, it would be desirable if our models
weren't quite as susceptible to them as they are right now. One such "defense 
method" is called adversarial training -- explicitly train your models to classify
adversarial examples correctly. The procedure is quite simple:
- Integrate the adversarial example procedure in the model's training loop. That
is, at _each step_, run the current batch through the model and use it to create
a batch of adversarial examples. Also run this batch through the model and compute
the loss. You can now update your network on this "double batch" of normal and
adversarial examples (you could even try training using _only_ adversarial examples).
- Test the network on some new adversarial examples, created using the same method
that was used during training. It should now be much less susceptible than without
adversarial training. Ideally, compare accuracies over a large "test set" of
adversarial examples. Also, does adversarial training affect performance on the
original data?
- Bonus: See if the adversarially trained network is also robust to other methods
of attack (if you came up with any), or e.g. using multiple gradient steps to
create the examples as proposed further above.
  
**Note** It's important that, during model training, you do not backpropagate
through the adversarial example creation process. That is, adversarial training
should look like this:

```python
with tf.GradientTape() as adversarial_tape:
    ...  # create adversarial examples

with tf.GradientTape() as training_tape:
    ...  # train model on adversarial and/or true batch
```

NOT like this:

```python
with tf.GradientTape() as training_tape:
    with tf.GradientTape() as adversarial_tape:
        ...  # create adversarial examples
    ...  # train model
```
There _are_ ways to get the second option to work (and it's less wasteful 
computationally), but it's easier to get them wrong.

## Bonus: Pretrained Models

Keras comes with a host of models already trained on ImageNet. This would be a
decent time to try them out. However, given the available time and resources,
you most likely won't be able to download the entire dataset/do adversarial
training.
- You can check [this easy-to-read overview](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json)
of the classes in ImageNet (maybe you find a better one).
- Download some images that belong to those classes (actually, if you download
  any natural image, there's a good chance its class is represented in ImageNet).
- ImageNet models usually expect inputs to be 224x224 pixels, so you will have
to resize the images to that shape!
- Models are available in `tf.keras.applications`. Pick one that you like, e.g.
DenseNet, a ResNet variant, good old VGG...
- Each model comes with a `preprocess_inputs` function. Generally, this needs to
be applied to the raw images (pixel values in [0,255], i.e. don't even normalize them yourself)
  before putting them into the network, else the predictions won't make sense.
  Read the docs for the respective model you are using.
- You can stick a cross-entropy loss on top of the model and create adversarial
examples as before.