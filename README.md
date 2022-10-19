# Bar Butler
There are thousands upon thousands of whiskies on the market with over ~50 different tasting notes to describe their taste. The Bar Butler, using NLP and a telegram chatbot interface, can listen to your natural language description of a whiskey, and give you a recommendation that you are sure to enjoy.

But what if you're not pretentious enough to know how to describe the taste? No matter, you can also express to the bot using natural language, what movie you're planning to pair the whiskey with. The bot will automatically retrieve the plot of the movie, perform emotion sentiment analysis, find the most semantically similar tasting notes, and recommend a whiskey to you without you even having to understand what tasting notes are.


## Project Structure
```
barbutler/
    bot.py - where the bot state machine is define and the bot starts polling
    bot_states.py - definition of all the states the bot could be in
    constants.py - holds constants and API keys
    handlers.py - bulk of the business logic is held here
    utils.py - holds utility functions for NLP and API querying
```

## Installation and Usage
Install requirements:
```bash
pip3 install -r requirements.txt
```

Get and install API keys:
```bash
echo "#!/bin/bash" >> api_keys.sh
echo TELEGRAM_API_KEY="YOUR KEY HERE" >> api_keys.sh
echo OPENAI_API_KEY="YOUR KEY HERE" >> api_keys.sh
echo TMDB3_API_KEY="YOUR KEY HERE" >> api_keys.sh
echo TMDB4_API_KEY="YOUR KEY HERE" >> api_keys.sh
source api_keys.sh
```

Run the bot:
```bash
python3 barbutler/bot.py
```

You will need a tasting_notes.txt file in the outer barbutler directory. The first time the bot is run, it will create a tasting_notes.pkl file which will contain the embeddings of each tasting note


## API Utilization
#### Whiskey API
Utilizes a fork of the whiskey-api: [https://github.com/ianmkim/whiskey-api](https://github.com/ianmkim/whiskey-api). Alongside the dataset provided by this repository in order to suggest different whiskeys and their tasting notes.

Since this is not a hosted API, we need to self host this one.

####  The Movie Database API
The Movies Database provides a database of movies that can be searched through. [https://developers.themoviedb.org/3/getting-started/introduction](https://developers.themoviedb.org/3/getting-started/introduction)

This API will primarily be used to look up descriptions when given movies in natural language through the telegrams chat.

A subset of the movies will be scraped and saved locally in order to provide recommendations given the whiskey.

#### The GPT-3 API
The smallest GPT-3 Ada model will be used to pick out flavor notes from natural language. Since I'm broke, I will not be fine-tuning or be using a larger model. The Ada model will make inferences few-shot with the example below as the prompt:

```
user: Hey I'm looking for a whiskey that's airy, light, and very fruity, what would you recommend?
tasting notes: airy, light, fruity

user: I'm interested in a dense whiskey that is well balanced and citrusy
tasting notes: dense, balanced, citrusy

user: You got any whiskey with heavy coffee and cigar flavors?
tasting notes: coffee, cigar

user: what sweet whiskey would you recommend?
tasting notes: sweet

user: Can you find me a malty whiskey with floral notes and smells a little bit like chocolate?
tasting notes: malty, floral, chocolate

user: what is a whiskey that's mellow and has butterscotch flavors?
tasting notes: mellow, butterscotch

user: Any idea what whiskey is smooth and apple flavored?
tasting notes: smooth, apple

user: I'm trying to relax for the night, any suggestions for bourbon that's smokey, bitter, but also amber?
tasting notes: smokey, bitter, amber

user: <prompt>
tasting notes:
```

From very unscientific tests, this seems to work relatively well. Much less effort required than finetuning a BERT on this task, which would just as well have sufficed (with much less money), but hey GPT-3 is new and hip.

The extraction of movie names from natural language works in a very similar way.
```
user: What's a good whiskey to pair with The Conjuring series?
movie name: The Conjuring

user: What should I drink when I watch Indiana Jones?
movie name: Indiana Jones

user: I'd like to watch the Hellraiser tonight. What should I get from the bar?
movie name: Hellraiser

user: What would be an interesting whiskey to pair with Star Wars?
movie name: Star Wars

user: Can you find me a whiskey to pair with Top Gun: Maverick?
movie name: Top Gun: Maverick

user: What can I drink with Sleepless in Seattle?
movie name: Sleepless in Seattle

user: What whiskey goes well with Knocked Up?
movie name: Knocked Up

user: I'm trying to relax for the night with Catch Me if You Can
movie name: Catch Me if You Can

user: what should I drink with Schindler's List?
movie name: Schindler's list

user: <prompt>
movie name:
```

## Features
### Fully through natural language.
The Bar Butler will be interacted through entirely natural language through the telegram bot interface.

### Whiskey recommendations
The platform will recommend whiskies based on various moods and tasting note information that you provide through the natural language interface.

For example:
```
Hey bar butler, I'm feeling something dense and fruity today, what would you recommend?
```

The GPT-3 model will extract the tasting notes, and since the extracted tasting notes are not guaranteed to be exactly same as the current tasting notes, it embeds all the tasting notes and runs cosine similarity with the preexisting list of tasting notes. Given this, we simply select the highest matching tasting note that we know exists, and use that as an input to the Whiskey API.

Using this metric, we return the top three whiskies that best matches the criteria.

### Whiskey recommendation based on a movie
The user also has the option of providing the bot with a movie that he/she will watch. The bot will then retrieve a description of this movie from natural language, analyse its emotional content, then find tasting notes that are most semantically similar to the emotional content and provide a whiskey recommendation based on the movie.

For example:
```
I'm planning on watching the Lord of the Rings tonight with my family, what Whiskey would you recommend?
```
