---
title: 'MathJax - Use Math in Hexo, Just Like Tex! (Including Common Issue Solutions)'
tags: Blogging
date: 2018-04-24 14:43:41
---
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

Sometimes you may want to explain some algorithms or principles with beautiful formulae in your blog. How to do this? Edit them in Microsoft Word, take a screenshot, crop it and put it in the blog post? When you finish your article and find out that you missed a symbol in the pictures - oh man, gotta repeat that again? Stop using those images now! A beautiful math display engine - MathJax allows you to code math like a coder.

<div style="font-size: 1.2em">
$$\mathcal{C}\phi \delta e \mathfrak{M}\alpha th \mathit{I}n \mathcal{H}ex\sigma \mathbb{N}o\omega!$$
</div>

<!-- more -->

## 1 Installation
### 1.1 With npm (For those using Hexo like me)
First, install *hexo-math* in your Hexo blog directory.
```bash
$ npm install hexo-math --save
```

Then, add *math* configurations in your *_config.yml* file.
```yaml
math:
  engine: 'mathjax'
```

Finally, also add to your *_config.yml* file in the **theme directory** these configurations below.
```yaml
mathjax:
  enable: true
  per_page: false
  cdn: //cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML
```

### 1.2 Or by inserting a snippet in your HTML code
Maybe you don't have to use math in every blog post. If so, insert the following snippet in your Markdown file also works.
```html
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
```

## 2 Usage
MathJax supports the same grammar that LaTeX does. To learn more about LaTeX, please refer to Chapter 3 of [The Not So Short
Introduction to LATEX](https://tobi.oetiker.ch/lshort/lshort.pdf)(CN version also available [here](http://www.mohu.org/info/lshort-cn.pdf)).

Use a "\\\\\(" and a "\\\\\)" to insert a formula in the line(they decide the boundary of the formula), or two "$$" to insert one that occupy a new line. I'll give a few examples below.


```md
\\(\mathcal{F}(x)=\mathcal{H}(x)-x\\)
```
\\(\mathcal{F}(x)=\mathcal{H}(x)-x\\)
```md
\\(E=mc^2\\)
```
\\(E=mc^2\\)
```md
$$\lim_{n\rightarrow \infty}(1+2^n+3^n)^\frac{1}{x+\sin n}$$
```
$$\lim_{n\rightarrow \infty}(1+2^n+3^n)^\frac{1}{x+\sin n}$$
```md
$$\mathcal{C}\phi \delta e \mathfrak{M}\alpha th \mathit{I}n \mathcal{H}ex\sigma \mathbb{N}o\omega!$$
```
$$\mathcal{C}\phi \delta e \mathfrak{M}\alpha th \mathit{I}n \mathcal{H}ex\sigma \mathbb{N}o\omega!$$

## 3 Problems when using MathJax with Hexo & Solutions
This list will be appended whenever I find any more.
### 3.1 Subscript symbol "_" gets mistaken as Markdown emphasize symbol
This is a tough problem. Hexo renderer would first render the .md file into a .html file, and the MathJax script will only work on the .html file. Therefore, when there are multiple subscript symbols, they might be rendered as &lt;em&gt;&lt;/em&gt; tags. 

For example: when you actually need a full-line formula \\(x_{i+1}+y_j\\), perhaps you'll get a "$$x<em>{i+1}+y</em>j$$" instead. Look into the HTML code and you'll understand why.

My solution for now, is giving up this Markdown emphasize symbol, since both "\_" and "\*" can be used as emphasize tags, and the alternative symbol "\*" will also work if we remove "\_". Using "\\\_" also works, but it would be frequently used(while "\*" isn't), thus turning our math code into mess code.

How do we do this? Bravely look into the *node_modules* directory and find the renderer of the Hexo engine. My renderer is *marked*, which is the default for Hexo. There is a file named *marked.js* inside *node_modules/marked/lib/* directory. You can find two appearances of "em:". Like this: 
```js
var inline = {
  ...
  em: /^\b_((?:[^_]|__)+?)_\b|^\*((?:\*:\*|[\s\S])+?)\*(?!\*)/,
  ...
};
```
and
```js
inline.pedantic = merge({}, inline.normal, {
  ...
  em: /^_(?=\S)([\s\S]*?\S)_(?!_)|^\*(?=\S)([\s\S]*?\S)\*(?!\*)/
});
```

Modify the regular expression after them - remove the one about "\_"s and leave the one about "\*"s. The new version would be:
```js
var inline = {
  ...
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  ...
};
```
and
```js
inline.pedantic = merge({}, inline.normal, {
  ...
  em: /^\*(?=\S)([\s\S]*?\S)\*(?!\*)/
});
```

From now on, you can use "\_" as the subscript in MathJax freely. You don't have to worry about its becoming &lt;em&gt;&lt;/em&gt; tags anymore.

### 3.2 Using "&" for aligning multi-line equations but getting a "Misplaced &"
For example, in my previous post about ResNet, I tried to use the following code to start a new line in an equation while aligning the lines to the equal sign:
```md
$$\frac{\partial{\mathcal{E}}}{\partial{x_l}} & = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\frac{\partial{x_L}}{\partial{x_l}}\\\\
& = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\Big(1+\frac{\partial{}}{\partial{x_l}}\sum_{i=l}^{L-1}\mathcal{F}(x_i,\mathcal{W}_i)\Big)$$
```
The "&" symbols were used to align the lines to a certain point. However, the result was a "Misplaced &" prompt.

By disabling MathJax, I found out that the rendered equation was correct, which means that **the problem isn't with Hexo renderer**. This was when I realized that although 
```md
\begin{equation}
\end{equation}
```
are not necessary, 
```md
\begin{split}
\end{split}
```
shouldn't be removed. Surround the equation with them will work. My code is here:
```md
$$\begin{split}
\frac{\partial{\mathcal{E}}}{\partial{x_l}} & = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\frac{\partial{x_L}}{\partial{x_l}}\\\\
& = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\Big(1+\frac{\partial{}}{\partial{x_l}}\sum_{i=l}^{L-1}\mathcal{F}(x_i,\mathcal{W}_i)\Big)
\end{split}$$
```
And it runs like:
$$\begin{split}
\frac{\partial{\mathcal{E}}}{\partial{x_l}} & = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\frac{\partial{x_L}}{\partial{x_l}}\\\\
& = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\Big(1+\frac{\partial{}}{\partial{x_l}}\sum_{i=l}^{L-1}\mathcal{F}(x_i,\mathcal{W}_i)\Big)
\end{split}$$

### 3.3 To be continued
If you encounter other issues while using MathJax with Hexo(with or without a solution), feel free to leave a comment below!