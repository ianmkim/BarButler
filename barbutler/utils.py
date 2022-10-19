import openai

import requests
import pickle
from os.path import exists

from typing import List

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelWithLMHead
from constants import TMDB3_API_KEY, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

"""
Load all the finetuned models for text/sentence embedding and
predictor for emotion classification
"""
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")


def yes_or_no_from_text(text:str, score_thresh=0.4) -> bool:
    """
    given a text prompt, it measures the cosine distance from the
    statement "yes" and the distance from the statement "no" using
    a sentence embedder all-MiniLM-L6-v2. Whichever one it is closer to
    is taken as a yes or a no.

    Returns a boolean - True: yes, False : no
    """
    approval_emb = embedder.encode(["yes"], convert_to_tensor=True)
    disapproval_emb = embedder.encode(["no"], convert_to_tensor=True)

    emb = embedder.encode([text ], convert_to_tensor=True)

    approval_score = util.cos_sim(emb, approval_emb)[0]
    disapproval_score = util.cos_sim(emb, disapproval_emb)[0]

    return (approval_score[0] > disapproval_score[0]).item()



def extract_emotion_from_text(description:str) -> str:
    """
    Given a discription of a movie, it extracts the emotional content
    of the description. This uses a base t5 model finetuned for emotion
    classification

    Returns an emotion string which is one of the below five:
       - sadness
       - joy
       - love
       - anger
       - fear
       - surprise
    """
    input_ids = tokenizer.encode(description+'</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids,
               max_length=2)

    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0].replace("<pad>", "").strip()

    return label


def retrieve_whiskey_based_on_tags(tags:List[str], price:str=None) -> dict:
    """
    Given a list of tasting notes and an optional price, it queries the
    Whiskey API to find spirits that most resemble the description.

    returns the raw JSON response, which is unchecked.
    """
    url = "https://evening-citadel-85778.herokuapp.com:443/shoot/"
    params = {
        "tags": ",".join([tag.strip() for tag in tags]),
    }
    print(params)
    r = requests.get(url=url, params=params)
    data = r.json()

    return data


def retrieve_movie_from_title(title:str) -> dict:
    """
    Queries the movie API with the title to retrieve metadata about the
    movie. The description of the movie is then used to extract emotional
    content of the description
    """
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB3_API_KEY,
        "query": title,
        "language": "en-US",
        "page": 1,
        "include_adult": True
    }

    r = requests.get(url=url, params=params)
    data = r.json()

    return data


def flatten_list(some_list:List[List[str]]) -> List[str]:
    """
    utility function to flatten a 2d list into a 1d list.
    """
    outlist = []
    for ls in some_list:
        for item in ls:
            outlist.append(item)

    return outlist


def search_tasting_notes(queries:List[str], score_thresh=0.5, top_k=1) -> List[List[str]]:
    """
    Given a list of unsanitized, free form tasting notes, this utility
    function tries to get the closest tasting note to the one given
    based on their semantic similarity from the Whiskey API.

    This is critical because the Whiskey API only accepts a set list of
    tags. For instance, if I searched for "Sophisticated," it will not
    return the correct whiskey, so we have to find the closest descriptor
    to "sophisticated" within the already chosen tags ("complex" in this
    instance).

    The score threshold determines the minimum cosine distance between
    the input term and the tasting notes in order to count as similar.

    top_k determines how many similar tasting notes to add per free form
    input word.
    """
    global embedder

    # first we must retrieve the word embeddings of the tasting notes
    tasting_notes_emb = None
    # check all possible paths
    possible_emb_paths = ["tasting_notes.pkl", "../tasting_notes.pkl"]
    for path in possible_emb_paths:
        # if the path exists
        if(exists(path)):
            # then lead the embeddings pickle file
            with open(path, "rb") as notes_emb_file:
                tasting_notes_emb = pickle.load(notes_emb_file)

    # second, we must retrieve the plain text values of the tasting notes
    tasting_notes = None
    possible_paths = ["tasting_notes.txt", "../tasting_notes.txt"]
    # similar structure as above
    for path in possible_paths:
        if exists(path):
            # open the tasting notes txt file and convert each line to
            # an element in the tasting notes array
            tasting_notes = []
            with open(path, "r") as notes_file:
                for line in notes_file:
                    tasting_notes.append(line.strip())

    # if the tasting notes embeddings do not exist
    if tasting_notes_emb is None:
        # create an embedding from the tasting notes plaintext
        tasting_notes_emb = embedder.encode(tasting_notes, convert_to_tensor=True)
        with open(possible_emb_paths[0], "wb") as file:
            pickle.dump(tasting_notes_emb, file)

    if(tasting_notes is None or tasting_notes_emb is None): return None

    most_sim_notes = []
    # go through each input tasting note
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # find the most similar ones from the database words
        cos_scores = util.cos_sim(query_embedding, tasting_notes_emb)[0]
        # retrieve the score and index of the highest scoring one
        top_results  = torch.topk(cos_scores, top_k)

        top_notes = []
        for score, idx in zip(top_results[0], top_results[1]):
            if(score >= score_thresh):
                top_notes.append(tasting_notes[idx])

        most_sim_notes.append(top_notes)
    return most_sim_notes


def extract_movie_from_str(text:str) -> str:
    """
    Leverages the Open AI GPT-3 model & API in order to extract the name
    of a movie from a free form text.

    Example input: I want to watch the Twilight series tonight
    Example output: Twilight
    """
    prepended_prompt = create_movie_prompt(text)
    response = openai.Completion.create(
          model="text-ada-001",
          prompt=prepended_prompt,
          temperature=0.7,
          max_tokens=256,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0,
          best_of=1
    )
    response_text = response["choices"][0]["text"]
    if(response_text == ""):
        return ""

    # in case many movies are returned by accident
    response_text = response_text.split("\n")[0]
    return response_text.strip()




def extract_tasting_notes_from_str(text:str) -> List[str]:
    """
    Leverages the OpenAI GPT-3 model & API to extract tasting notes
    from free form text.

    Exmple input: Can you recommend a light whiskey that is flowery and complex?
    Example output: light, flowery, complex
    """
    prepended_prompt= create_taste_prompt(text)
    response = openai.Completion.create(
          model="text-ada-001",
          prompt=prepended_prompt,
          temperature=0.7,
          max_tokens=256,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0,
          best_of=1
    )

    response_text = response["choices"][0]["text"]
    if(response_text == ""):
        return []

    response_text = response_text.split("\n")[0]
    notes = [ note.strip() for note in response_text.split(",") ]

    return notes


def create_taste_prompt(user_prompt:str) -> str:
    """
    The model is not finetuned and is doing few-shot inferences
    so we need to provide example entries to prompt the algo
    """

    prompt = """
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

user: """
    prompt += user_prompt
    prompt += "\ntasting notes:"
    return prompt

def create_movie_prompt(user_prompt:str) -> str:
    """
    The model is not finetuned and is doing few-shot inferences
    so we need to provide example entries to prompt the algo
    """
    prompt = """
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

user: """
    prompt += user_prompt
    prompt += "\nmovie name:"

    return prompt



