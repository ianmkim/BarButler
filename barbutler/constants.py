import os

TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB3_API_KEY = os.getenv("TMDB3_API_KEY")
TMDB4_API_KEY = os.getenv("TMDB4_API_KEY")

if __name__ == "__main__":
    print(os.environ)
    print(TELEGRAM_API_KEY)
    print(OPENAI_API_KEY)
    print(TMDB3_API_KEY)
    print(TMDB4_API_KEY)


