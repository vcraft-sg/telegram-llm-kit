from telegram.ext import Application, CommandHandler, MessageHandler, filters

from telegram_llm_kit.bot.handlers import (
    HandlerDependencies,
    message_handler,
    search_handler,
    start_handler,
)


def create_app(token: str, deps: HandlerDependencies) -> Application:
    """Build and configure the Telegram bot application."""
    app = Application.builder().token(token).build()

    # Inject dependencies via bot_data
    app.bot_data["deps"] = deps

    # Register handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("search", search_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    return app
