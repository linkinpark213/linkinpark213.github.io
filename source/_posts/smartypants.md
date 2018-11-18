---
title: Smartypants is NOT SO SMART
date: 2018-03-20 11:00:22
tags: Blogging
---
When blogging with Hexo, every time I type a single quotation mark(also called apostrophe) like this: 
```
'
```
, Hexo would convert it to a symbol like this
```
’
```
You would say that this is also an apotrophe, but it really looks UNBEARABLE in the articles. It's been a problem bothering me for more than a month.(I'm not saying that this is the reason for not updating my blog, but I don't mind if you think so!)

<!-- more -->


![apostrophe](/images/smartypants/apostrophe.png)

Therefore, I Googled about this problem and tried to find other victims. According to their sharings, this problem is caused by *marked* -- the default markdown renderer of Hexo.The *"smatrypants"* function of marked was turned on by default.

Now take a look at the introduction of *smartypants* on the *hexo-renderer-marked* page:
> *smartypants* - Use "smart" typograhic punctuation for things like quotes and dashes.

C'mon, seriously? 

There are a few bloggers who solved this by adding the code below to the _config.yml file in the blog directory. 

```yml
mark:
  smartypants: false
```

This worked for most victims (perhaps all of them), but not for me. I have no idea why those config wasn't working, so if anyone finds out the reason, please contact me by e-mail.

If you're sure *smartypants* is causing the problem, and the solution above didn't work for you either, maybe you can try my solution.

Since *hexo-renderer-marked* is installed in the blog's *node_modules* directory(may also be in your Node.js directory if installed globally), isn't it possible that we change its own configurations? I looked at the *index.js* file in the *node_modules/hexo-renderer-marked/* directory. There you are, smartypants!

```javascript
hexo.config.marked = assign({
  gfm: true,
  pedantic: false,
  sanitize: false,
  tables: true,
  breaks: true,
  smartLists: true,
  smartypants: true,
  modifyAnchors: '',
  autolink: true
}, hexo.config.marked);
```

Now you know what to do.

Aaaaaaaaaaaaaaaand many thanks to Xizi Wu, the artist of my new avatar! I love it!
![Harper Long by Xizi Wu](/images/long_nobg.png)