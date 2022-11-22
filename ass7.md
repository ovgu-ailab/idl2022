---
layout: default
title: Assignment 7
id: ass7
---

# Assignment 7: Attention-based Neural Machine Translation
**Deadline: November 28th, 9am**

In this task, you will implement a simple NMT with attention for a language pair 
of your choice.  
We will follow the corresponding 
[TF Tutorial on NMT](https://www.tensorflow.org/tutorials/text/nmt_with_attention).

Please do **not** just use the exemplary English-Spanish example to reduce temptation
 of simply copying the tutorial.  
You can find data sets [here](http://www.manythings.org/anki/). We recommend
 picking a language pair where you understand both languages (so if you do speak 
 Spanish... feel free to use it ;)). 
This makes it easier (and more fun) for you to evaluate the results.
However, keep in mind that some language pairs have a very large amount of examples,
whereas some only have very few, which will impact the learning process and the
quality of the trained models.

You may run into issues with the code in two places:
1. The downloading of the data inside the notebook might not work (it crashes
with a 403 Forbidden error). In that case, you can simply download & extract the
data on your local machine and upload the .txt file to your drive, and then mount
it and load the file as you've done before.
2. The `load_data` function might crash. It expects each line to result in
*pairs of sentences*, but there seems to be a third element which talks about
attribution of the example (at least if you download a different dataset from 
   the link above). If this happens, you can use `line.split('\t')[:-1]` to
exclude this in the function.


Tasks:
- Follow the tutorial and train the model on your chosen language pair.
- You might need to adapt the preprocessing depending on the language.
- Implement other attention mechanisms and train models with them (there are Keras
  layers for both):
  - Bahdanau attention (`AdditiveAttention`)
  - Luong's multiplicative attention (`Attention`)
    

Compare the attention weight plots for some examples between the attention 
mechanisms.  
We recommend to add `,vmax=1.0` when creating the plot in 
`ax.matshow(attention, cmap='viridis')` in the `plot_attention` function
so the colors correspond to the same 
attention values in different plots. 
- Do you see qualitative differences in the attention weights between different
 attention mechanisms?  
- Do you think that the model attends to the correct tokens in the input language
 (if you understand both languages)?

Here are a few questions for you to check how well you understood the tutorial.  
Please answer them (briefly) in your solution!  
- Which parts of the sentence are used as a token? Each character, each word, 
or are some words split up?
- Do the same tokens in different language have the same ID?   
  e.g. Would the same token index map to the German word `die` and to the 
  English word `die`?
- Is the decoder attending to all previous positions, including the previous
 decoder predictions?
- Does the encoder output change in different decoding steps?
- Does the context vector change in different decoding steps?
- The decoder uses teacher forcing. Does this mean the time steps can be computed
 in parallel?
- Why is a mask applied to the loss function?
- When translating the same sentence multiple times, do you get the same result?
Why (not)? If not, what changes need to be made to get the same result each time?

Hand in all of your code, i.e. the working tutorial code along with all changes/additions you made. Include outputs which document some of your experiments. Also
remember to answer the questions above! Of course you can also write about other
observations you made.
