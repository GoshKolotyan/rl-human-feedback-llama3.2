from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from bot.bot_configs import BotConfigs, Prompt
from bot.load_model import LlamaBot


configs = BotConfigs()
prompt_config = Prompt()
llama_bot = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Hello! I'm your LLaMA assistant bot. Send me a message and I'll respond."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "Just send me any message and I'll respond using LLaMA!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages using LLaMA model."""
    user_message = update.message.text

    await update.message.chat.send_action("typing")

    response = llama_bot.answer(user_message)
    await update.message.reply_text(response)


def main() -> None:
    """Run the bot."""
    global llama_bot

    print("Loading LLaMA model...")
    llama_bot = LlamaBot(model_path=configs.MODEL_PATH)
    print("Model loaded successfully!")

    application = Application.builder().token(configs.BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
