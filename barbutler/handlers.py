import utils

from bot_states import (
    START,
    CHOOSING,
    MOVIE,
    TASTE,
    FOLLOWUP
)

from telegram import ReplyKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import CallbackContext

reply_keyboard = [
    ["tasting notes", "movie"],
    ["Done"],
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)


def help_str() -> str:
    """
    Returns the help string
    """
    return "You can either...\n1) ask for a whiskey recommendation based on tasting notes (type tasting notes), or... \n2) based on a movie (type movie).\n\nEither way you're going to get a recommendation for a whiskey"



def start(update:Update, context:CallbackContext) -> int:
    """
    Entry point for the chatbot. Prints the greeting and the help
    string, then puts the network in the CHOOSING state in which
    the user will be able to enter whether they want recommendations
    based on a movie or a whiskey
    """
    reply_text = "Welcome, my name is BarButler. I give out whiskey recommendations\n\n" + help_str()

    update.message.reply_text(reply_text, reply_markup=markup)
    return CHOOSING



def choosing(update:Update, context:CallbackContext) -> int:
    """
    Handler for the CHOOSING state. If the user input includes the word
    "movie" then it will put the chatbot in the MOVIE mode from which
    the user will be able to enter in a movie they are planning to watch

    If the user enters "tasting notes" then the chatbot will go to the
    TASTE state in which the user will be able to describe what kind of
    tasting notes they want from their whiskey

    If neither phrases were entered, then the bot puts itself back in
    CHOOSING state until it can undersatnd what the user wants
    """
    choice = CHOOSING
    text = update.message.text.lower()

    if "movie" in text:
        update.message.reply_text("Perfect, what movie are you going to watch today?")
        choice = MOVIE
        return choice

    elif "tasting notes" in text:
        update.message.reply_text("Wonderful, what kind of whiskey are you looking for in terms of taste?")
        choice = TASTE
        return choice

    else:
        update.message.reply_text("Sorry didn't understand that\n\n" + help_str())
        return CHOOSING



def help(update:Update, context:CallbackContext):
    """
    If the bot is put in the help state, it will send the help string
    then put the bot back in the START position.
    """
    help_text = help_str()
    update.message.reply_text(help_text, reply_markup=markup)
    return START



def rec_from_movie(update:Update, context:CallbackContext):
    """
    This handler is for when the bot is in the MOVIE state. In this
    state, the user is expected to tell the bot what movie the user
    wants to see.

    Based on this input, the function will extract the name of the
    movie from the input, query the movie API for the description of
    the movie, inference the emotions portrayed by the movie description,
    then find tasting notes that employ similar emotions.

    Finally, with that information, it will query the whiskey API
    to find the whiskies with the most similar tasting notes
    """

    # set the current state so that when the user is asked
    # a followup question, it can know whether to ask for a new movie
    # or new tasting notes
    context.user_data["prev_state"] = MOVIE

    # extract the movie title from natural language
    movie_title = utils.extract_movie_from_str(update.message.text)
    # if no movie could be found, then ask another time
    if(movie_title == ""):
        reply_text = "I didn't find any movie with that title. Could you say that again?"
        update.message.reply_text(reply_text)
        return MOVIE

    # if the movie title could be extracted, query the movie database
    # and search for a movie with that title
    movie_data = utils.retrieve_movie_from_title(movie_title)
    # if the movie is not found in the database, then ask for a new
    # movie title
    if(movie_data["total_results"] == 0):
        reply_text = f"Sorry, couldn't find a movie with the title {movie_title}. What were you going to watch again?"
        update.message.reply_text(reply_text)
        return MOVIE

    # if the movie is found, report to the user that the bot
    # knows about the movie
    movie = movie_data["results"][0]
    reply_text = f"{movie['original_title']} is a great movie. Let me find a whiskey that matches the mood of this movie!"
    update.message.reply_text(reply_text)


    # get the description of the movie then extract the emotion
    # conveyed by that description
    movie_description = movie["overview"]
    emotion = utils.extract_emotion_from_text(movie_description)

    # Then search the embeddings of tasting notes for notes that are
    # most similar to that particular emotion

    # WARN this is where the non-statistically valid thing happens
    # it is not guaranteed that these two embeddings lie on the same
    # latent space that we want. I would further finetune a bert
    # network to do this compression op better if I had more time
    tasting_notes = utils.search_tasting_notes(
            [emotion],
            score_thresh=0.3,
            top_k=5)

    # tasting_notes is a 2d array, so flatten for our purposes
    tasting_notes = utils.flatten_list(tasting_notes)

    reply_text = f"I sense {emotion} from this movie. I'll try to find you a whiskey that is {', '.join(tasting_notes)}"
    update.message.reply_text(reply_text)

    # use the extracted tasting notes to find the right whiskey
    whiskey_recs = utils.retrieve_whiskey_based_on_tags(tasting_notes)

    if whiskey_recs["count"] == 0:
        reply_text = f"Sorry, there were no whiskies that goes well with  {movie_title}"
        update.message.reply_text(reply_text)
        return START

    reply_text = "Alright I got it! I would recommend the "
    # only get the top 1 recommendation
    for rec in whiskey_recs["results"][:1]:
        reply_text += f"{rec['title']} to sip while watching {movie_title}\n\n"
        reply_text += rec["description"] + "\n"
        reply_text += f"The {rec['title']} goes for about {rec['price']} on the market."

    update.message.reply_text(reply_text)
    update.message.reply_text("Would you like another recommendation?")
    return FOLLOWUP



def rec_from_taste(update:Update, context:CallbackContext):
    """
    When the bot is in TASTE state, it waits for the user to give it a
    natural language description of what the whiskey should taste like.

    Using GPT-3, we extract the tasting notes from that natural language
    description. Then we use a sentence transformer to find the most
    similar tasting note from the ~50 tasting notes we have in the
    database. For more explanations, see the utility function:
    search_tasting_notes().
    """
    # set context for followup question
    context.user_data["prev_state"] = TASTE

    notes = utils.extract_tasting_notes_from_str(update.message.text)

    if(len(notes) == 0):
        reply_text = "I didn't get any flavor names from your text. Could you say that again?"
        update.message.reply_text(reply_text)
        return TASTE

    # search tasting notes
    notes_from_db = utils.search_tasting_notes(notes)
    notes_from_db = utils.flatten_list(notes_from_db)

    reply_text = f"Perfect. Looking for whiskies that are {', '.join(notes_from_db)}. Give me a second"
    update.message.reply_text(reply_text)

    # get the whiskey with the tasting notes closest to the notes given
    whiskey_recs = utils.retrieve_whiskey_based_on_tags(notes_from_db)
    if whiskey_recs["count"] == 0:
        reply_text = "Sorry, there were no whiskies with the flavors you were looking for"
        update.message.reply_text(reply_text)
        return START

    reply_text = "Alright I got it! I would recommend the "
    # only get the top 1 recommendations
    for rec in whiskey_recs["results"][:1]:
        reply_text += rec["title"] + ".\n\n"
        reply_text += rec["description"] + "\n"
        reply_text += f"The {rec['title']} goes for about ${rec['price']} on the market."

    update.message.reply_text(reply_text)
    update.message.reply_text("Would you like another recommendation?")
    return FOLLOWUP



def followup(update:Update, context:CallbackContext):
    """
    in the FOLLOWUP state, the bot asks whether the user would want
    another recommendation. If the previous state was TASTE, then
    it asks for another description of the taste of the whiskey the user
    wants. If the previous state was MOVIE, then it asks for a new
    movie to base the recommendation off of.
    """

    # look at yes_or_no_from_text documentation for more info on
    # how this works
    answer = utils.yes_or_no_from_text(update.message.text)
    if answer and context.user_data["prev_state"] == TASTE:
        update.message.reply_text("Perfect, enter in another description of a whiskey you want to drink.")
        return TASTE
    if answer and context.user_data["prev_state"] == MOVIE:
        update.message.reply_text("Perfect, enter in another movie you were going to watch.")
        return MOVIE

    update.message.reply_text("Alright, I'll see you later!")
    return START



def done(update:Update, context:CallbackContext):
    pass
