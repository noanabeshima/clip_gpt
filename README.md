# CLIP GPT

160M parameter GPT model trained on captions using OpenAI's [CLIP](https://github.com/openai/CLIP) tokenization. Initialized with OpenAI's ViT-B/32 CLIP text encoding model.

Trained on ~12M alt-text captions from [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), using GPU-hours donated by [Hofvarpnir Studios](https://hofvarpnir.ai/).

It was trained to act as a text prior for interpretability, to be used for something like ['learning the prior'/'imitative generalization'](https://www.lesswrong.com/posts/JKj5Krff5oKMb8TjT/imitative-generalisation-aka-learning-the-prior-1).

Suppose you have some function $f: \text{Image} \to \mathbb{R}$ (like e.g. a neuron in a vision model) and you hypothesize that its behavior can be approximately described with text. For example "Diagonal line from bottom-left to top-right of image" or "Dog head facing left".

CLIP has this signature:

$\text{CLIP} : (\text{Image}, \text{Text}) \to \mathbb{R}$

Then, given a fixed piece of text $t \in \text{Text}$, we get a function

$\text{CLIP}_{t} : \text{Image} \to \mathbb{R}$

This function probably monotonically increases as its input $\text{im} \in \text{Image}$ better matches the text, $\text{t} \in \text{Text}$.

Then, if a function $f: \text{Image} \to \mathbb{R}$ monotonically increases as its input better matches *some* text/caption, we might hope that it can be modelled by

$\text{CLIP}_{t} : \text{Image} \to \mathbb{R}$

composed with some learned monotonic function $h_{\theta}$ with signature $h_{\theta}: \mathbb{R} \to \mathbb{R}$.

I.E. we hope that for all $\text{im} \in \text{Image}$,

$f(\text{im}) \approx h_{\theta}(\text{CLIP}_{t} (\text{im}))$

for some $t \in \text{Text}$ and parameters $\theta \in \mathbb{R}^n$.

The idea was that possibly this $t \in \text{Text}$ could be selected using either [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) with CLIP GPT as a prior on text, or using [reinforcement learning with CLIP GPT](https://arxiv.org/abs/2009.01325).

I decided not to pursue the direction because doing gradient descent in CLIP image embedding space + a monotonic function $h: \mathbb{R} \to \mathbb{R}$ along with some other similar techniques didn't do a great job at modelling logits of a small vision model (it did an alright job).

I also tried doing something like [Textual Inversion](https://textual-inversion.github.io/) to come up with learned tokens that combined with CLIP and a monotonic function would model logits. This had similar okay performance. Plugging in the CLIP last hidden states on the textual inversion token into stable diffusion didn't give very class-relevant pictures. I think textual inversion might be different because there's richer model feedback in generating images--trying to predict a single logit doesn't give great feedback.

It still seems plausible to me that something in this direction could work, but less likely than when I began. Feel free to reach out if you're interested in this direction and think I might have useful advice!
