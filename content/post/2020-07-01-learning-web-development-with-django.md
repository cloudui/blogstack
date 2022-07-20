---
title: Learning Web Development with Django
image: https://blog.echen.io/assets/img/blog/2020-07-01-learning-web-development-with-django/django.jpeg
categories: [Web Development, Django]
tags: [django, aws, guide]
date: 2020-07-01

description: A short guide on how to get started
---

I spent my last couple of months learning how to make websites with the Django framework. I have happened to pick up on a few things from working on it, and I want to give you a few resources on how to get started.

This website (referring to my old site at [blog.echen.io](https://blog.echen.io)) in particular was made using Django, a python web framework. It is not difficult to learn, and it helps if you are at least familiar with the class structure in python. It's good for websites that require things like user log in, posting, and model-related work (versions of the same type of object). I wouldn't recommend it for static websites or rather simple websites, as it might take more time than you want. If you look at any of my other websites, all of them are also written in Django, other than my homepage, which is a static site hosted on Amazon S3 and CloudFront. I'm familiar with Django, but not much else.

## How do I learn Django?

I learned Django by reading ["Django for Beginners"](https://djangoforbeginners.com/introduction/) by Will Vincent. It's really good and I highly recommend it. You should at least know python and a little bit of Linux shell scripting before you get started (and HTML and CSS, of course). Django itself uses python but you need a lot of commands to get stuff set up. After that, you'll start to get a hang of Django development from reading the book.

### Is it useful?

Websites like BitBucket (Git Repository) use Django, and a lot of big companies use it too. Django developers are wanted by tech companies, and it's a good framework to learn regardless. Other powerful frameworks include Express.js, which is based on Node.js, and Angular (Javascript branch). There are a lot, and Django is just one out of a whole collection. Another common python framework is Flask. Both are powerful, but Django has a little more customization and complexity.

## What do I do after I make a project?

Will Vincent will tell you how to deploy Django projects to Heroku, an online service for deploying projects. It's free for basic stuff, but it's not a good long-term solution, mostly if you want your own domain. Learning how to deploy your Django projects will require you to grasp networking, like TCP/IP, working on a cloud server, and learning how to use reverse proxies and HTTP services like Gunicorn. A good tutorial is [here](https://www.google.com/search?q=nginx+gunicorn+django&oq=nginx+gunicorn+django&aqs=chrome..69i57j35i39j0l4j69i60j69i61.2694j0j1&sourceid=chrome&ie=UTF-8). But don't worry, this will come after you finish your projects. I recommend you use Amazon Web Services to deploy your projects, but that is quite complicated, to be frank. You can always search online for the easiest place to get a server for yourself.

Will Vincent's professional Django book will guide you through much more complex stuff, and there are still a lot of APIs (REST frameworks) that you can add to make your projects more complex. Both are a good read but finish the beginner book first. He never really goes into deployment, but you'll know how to develop Django projects pretty extensively.

A lot of your skills will come from messing up. I've spent days debugging issues. I have searched up thousands of questions on Google, but that is all part of the learning process. It will be frustrating, but it will be worth it. 

## Static Hosting Recommendations

You do not need a server or Django if you want a static website. I recommend Jekyll and GitHub Pages. The files sync with your GitHub repository, and the process is simple and streamlined. Furthermore, you don't need to manage a server or anything fancy. You can also try Amazon S3, but that is less friendly to beginners.

Django is a very powerful framework, and you can make almost anything you want. If you're questioning whether it's right for you, I urge you to give it a try. You can then decide what you really want to use. If you don't want to streamline the project bottom up, you can use services like Wix or Squarespace, or WordPress. Those services let you do things like drag and drop, and you don't even need coding to create a very stylish website. They aren't cheap though; you should check your budget to make sure it's the right choice for you. But, it might be worth it, since it makes building the website much faster and easier.