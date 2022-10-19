import logging
from typing import Dict

from constants import TELEGRAM_API_KEY
from bot_states import (
    START,
    CHOOSING,
    MOVIE,
    TASTE,
    FOLLOWUP,
)

from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
)

import handlers


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


updater = Updater(TELEGRAM_API_KEY, use_context=True)


def main() -> None:
    updater = Updater(TELEGRAM_API_KEY)
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points = [CommandHandler('start', handlers.start)] ,
        states={
            START: [
                MessageHandler(Filters.text, handlers.start),
            ],
            CHOOSING: [
                MessageHandler(Filters.text, handlers.choosing),
            ],
            MOVIE: [
                MessageHandler(Filters.text, handlers.rec_from_movie),
            ],
            TASTE: [
                MessageHandler(Filters.text, handlers.rec_from_taste),
            ],
            FOLLOWUP: [
                MessageHandler(Filters.text, handlers.followup) ,
            ],
        },
        fallbacks=[MessageHandler(Filters.regex("^Done$"), handlers.done)],
        name="whiskey_conversation",
    )

    dispatcher.add_handler(conv_handler)
    updater.start_polling ()
    updater.idle()


if __name__ == "__main__":
    main()
