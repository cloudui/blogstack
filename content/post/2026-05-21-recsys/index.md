---
title: RecSys

categories: [Discussion, Politics]
tags: [politics, inequality, socialism, satire]

math: true

author: Eric Chen
image: thumbnail.webp
date: 2026-05-04

description: FlashAttn-v2
---

I worked on the recommendation systems for Facebook Reels for 9 months, and I wanted to talk about the difficulties in making improvements at this level. 

Short-form video took over the world circa 2020, and it's grip on the world has only tightened since. Platforms like TikTok, Instagram Reels, YouTube Shorts, and Facebook reels have billions of monthly users that has driven a massive amount of growth between these companies. It's had a drastic effect on culture all around the world -- people have made careers out of being TikTok stars, breaking news often spills via Instagram Reels, and your dad probably sends the family groupchat too many Shorts while sitting on the toilet. Viral moments cycle through within days, and the last place you probably ate at was probably decided by some video that you watched. 

Hiding behind these platforms are the billions of dollars spent on these recommendation engines that are meant to capture your attention. The "algorithm", as most people call it, decides which videos you get to see: it's the reason you get ten cat videos in a row after you liked one, it bombards everyones feeds with Will Smith slapping Chris Rock after noticing dozens of the same video blowing up across the app. The goal of these models are to learn who you are and what you like based on *how* you interact with the content it gives you. And although I cannot exactly share how the algorithm works under the hood, I can touch on some of the difficulties working within these systems. 

# Model Architecture
Most short-form recommendation systems share a similar underlying architecture. There are an infinite number of ways to narrow down what content you get to see next; however, there is one big limitation that essentially forces these companies into the same underlying pipeline: **responsiveness**. 

The feed is supposed to be an infinite continuous experience. YOu never want a user to wait for the next video -- it should ideally be a seamless experience where the videos just keep coming after every swipe. With milliions or billions of videos being uploaded every day, it would be physically impossible to carefully evaluate whether a user would like every single video item-by-item with one model. If you did, you'd probably have to wait days just to get back one video, or use an extremely dumb model that doesn't understand the user very well.

Instead, these companies have *many different models* stacked on each other, increasing in complexity the more the videos are narrowed down. 


- Responsiveness: the feed is a continuous experience. You want to minimize the amount of time people have to wait for the next videos to load. Ideally, they never even know when it happens.
- 