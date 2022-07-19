---
title: Deploying this website [deprecated]
categories: [Web Development, Django]
tags: [django, website]
date: 2020-06-29
---

> Note: "this website" now refers to my previous blog site, coded with Django. You can find it at [blog.echen.io](https://blog.echen.io).

I finally deployed this website! It took a bit longer than I would have liked to write it, but I am happy with the result. I deployed it without using Docker, and it actually saved me a lot of time and frustration compared to my old projects. Take a look at my home website at [echen.io][echen.io] if you would like. It is hosted on Amazon S3 and CloudFront.

Some details about the creation of this site: This website was built with Django. I used the default SQLite as the database; I didn't really need to use something like PostgreSQL considering these models are quite basic. I used Bootstrap4 for the CSS framework and I have some JQuery for the dark mode and Waypoint.js for the infinite scroll on the homepage. I wasn't able to get infinite scroll for the search page since the page URL would get intertwined with the search query URL, but I'll try to add a "next page" button or maybe I'll use AJAX if I can.

The search form logic was created using [django-watson][django-watson]. I wasn't able to get the search function to work for the date, but I will probably fix that soon. The icon pack I use is called ["Feather Icons"][feather-icons], and it looks really clean in my opinion. I'll probably upload this to my GitHub as I have learned how to conceal my secret keys and such, and you guys can take a look at the source code if you would like. About the dark mode: the CSS implementation is just a simple JQuery with an input button. The hard part was creating the toggle and using JQuery's "localStorage" class to store the user's preference and keep consistency.

The animation created on the toggle was very difficult to implement across all of the pages, which is why it's only on the homepage. I think it looks better that way, too. Lastly, you might see some images in future posts. It uses the Media library of Django and Pillow for the model field for user upload. I used LetsEncrypt for the SSL certificate, and I learned how to use NGINX Virtual Host, so I can now fit multiple sites on my small Lightsail server. I might learn how to host multiple websites with Docker, but I don't think I will be using Docker that much for websites anymore, as it doesn't really make things easier. I do think it's cool, and I am definitely not using it to its full potential, but it has caused me more problems than it has solved. I like how I can build everything with just one command, and I don't have to mess with the Daemon for Gunicorn or NGINX.

I can probably deploy websites faster now because of my experience. Crontab is literally the worst in a container though... I'll try to make some more features on this website along with some of my others. You can check them out on my homepage (look to the bottom). One I didn't include is [news.echen.io][news] since it is rather basic and lacks any type of security behind it, but feel free to post stuff on it (I will probably take it down eventually). You can contact me at my email if you have any suggestions or criticism of any of my sites!

<br>
> Do what you can, with what you have, where you are.
>
> -- <cite>Theodore Roosevelt</cite>

[django-watson]: https://github.com/etianen/django-watson
[echen.io]: https://echen.io
[feather-icons]: http://feathericons.com/
[news]: https://news.echen.io