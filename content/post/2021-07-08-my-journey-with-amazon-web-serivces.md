---
title: My Journey with Amazon Web Services
categories: Cloud
tags: [aws]
image: http://blog.echen.io/assets/img/blog/2021-07-08-my-journey-with-amazon-web-services.md/thumbnail.png
date: 2021-07-08
---

Around a year ago, I was bored out of my mind. School sluggishly churned along while I desperately searched for something to do. I stumbled across this YouTube video about creating a Twitter Bot, and I was intrigued. Having relearned some Python, I decided to implement my own.

The process of creating one wasn't difficult. I essentially just copied the code into my clipboard, created a developer account, and boom, bot completed. It was a rather simple bot-- it scanned tweets that mentioned my profile with a particular hashtag and responded with a corny message. I contemplated making it "naughty," but I ultimately decided against it, since some of my teachers follow me (I know, so famous).

Since the bot essentially just checks for new tweets every now and then, it requires a while loop to continuously poll new tweets with API calls. I wanted to keep it running indefinitely, but I surely wasn't going to use my computer to do that. So I needed a solution that allowed me to run a script continuously. The first place I looked to was [pythonanywhere.com](https://www.pythonanywhere.com/), which allows you to run python scripts on their servers. The interface was easy to use, but the free tier servers clocked out too quickly, only allowing my script to run for maybe a couple minutes tops. I contemplated getting a server, but I was not familiar with how to obtain or use one, so that was out of the picture.

I asked my dad, who directed me to Amazon Web Services (AWS), which is a cloud platform that allows you to do anything you imagine a tech company to do. Machine learning, web servers, AI, databases, you name it. He showed me how to use Lambda, a serverless-compute product, which allows users to run scripts in all sorts of programming languages (Java, Node.js, Python, etc.) without worrying about provisioning a server. How cool is that? I used an S3 bucket (storage service, like Google Drive) to track a file with the ID of the most recent tweet polled. Naively, I stored the credentials in my script that calls the S3 bucket on the behalf of a user in my AWS account with access to S3 (don't do that). Lastly, my dad showed me that I could periodically trigger this Lambda function with a CloudWatch (monitoring service) event, which would invoke it every couple of minutes.

It worked splendidly, and I was happy. *So...what does this have to do with the thing in the title?* Well, I was intrigued by AWS after the completion of this project, and I decided to learn some more. I borrowed my dad's LinuxAcademy account (provided by his employer) to learn some AWS principles. After taking a rather simple and introductory course, I wanted to dive in deeper. I was amazed to learn that a company like Netflix could run all of its operations in the cloud. I eventually stumbled across the Solutions Architect Course, which I saw was like 60 hours long. But the challenge was alluring, and my most-of-the-time-super-lazy ass decided to take it head-on.

Somehow, I completed the entire course within a month. I wanted to take the certification exam immediately, but I still was not completely comfortable with the services yet. I wanted to take a break, and I did not pick it back up until this summer. However, I did end up experimenting with AWS a bit. If you look at the DNS record for this site or any of my others, you can see that it's hosted on Amazon/CloudFront. At least I put my knowledge to some use.

So what now? Well, as I mentioned already, I picked it back up this summer. Since the course I took last year became legacy (there is an updated exam), I decided to take an updated course, with the same goal of taking the certification. I completed it about a week ago, and I feel a lot more confident than I did a year ago, and I'm looking to book a certification exam in the coming weeks. But why would a puny pre-freshman college student need something like this? That's a good question...I don't know, either. Maybe it'll help me get an internship next year, or maybe I'll just end up wasting $150, who knows. Maybe I just want to feel pride over the effort I put into this.

*What did I gain reading this 900-word essay of Eric's stream of consciousness?* Good question. I don't want you to leave without getting something in return. After all, I am thankful you made it this far. Below, I've attached the virtual notebook (using OneNote) containing my AWS notes. This is a culmination of about 100+ hours of video-watching and AWS sandbox experimentation. If you think AWS is interesting, I advise you to read about it and see if it is something you would truly want to learn. I'll attach some resources below.

AWS is a fantastic service. Many start-up companies can develop apps and create their business without ever needing to worry about tricky computer hardware and renting out massive facilities. It's an easy way to enter the tech world where you can focus on the stuff you're motivated to make. Since I don't want to sound like a salesman right now, alternatives to AWS like [Google's Cloud Platform](https://cloud.google.com/) or [Microsoft's Azure](https://azure.microsoft.com/en-us/) are probably just as amazing, and you should check them out too. They all have free trials, where you can experiment with a lot of really cool tech! Best way to learn is just to hop straight in.

> Update: I passed the certification! I am very happy and relieved that the hours I put in meant something. Now on to the next journey...


### Read about AWS

- [https://aws.amazon.com/about](https://aws.amazon.com/)
- [https://www.wikiwand.com/en/Amazon_Web_Services](https://www.wikiwand.com/en/Amazon_Web_Services)
- [https://aws.amazon.com/certification/](https://aws.amazon.com/certification/)

### Extra Links

- Twitter Bot Video: [https://www.youtube.com/watch?v=W0wWwglE1Vc](https://www.youtube.com/watch?v=W0wWwglE1Vc)
- Python Servers: [pythonanywhere.com](https://www.pythonanywhere.com/)
- Google Cloud Platform: [https://cloud.google.com/](https://cloud.google.com/)
- Microsoft Azure: [https://azure.microsoft.com/en-us/](https://azure.microsoft.com/en-us/)

### Beautiful Notes <3:

Actually...this was a lot of work. You have to show you want it 😉. Message me and you will get it.

<br>
> There is no knowledge that is not power
>
> -- <cite>Ralph Waldo Emerson  </cite>