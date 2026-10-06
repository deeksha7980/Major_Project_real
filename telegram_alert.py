import os
from typing import Optional
import telepot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

bot = None
if TELEGRAM_BOT_TOKEN and not TELEGRAM_BOT_TOKEN.startswith("YOUR_"):
    try:
        bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
    except Exception as e:
        print("Telegram init failed:", e)


def send_alert(message: str, image_path: Optional[str] = None) -> bool:
    """Send alert text (+ optional frame image) to Telegram."""
    if bot is None:
        print("[Telegram disabled] alert:", message)
        return False
    try:
        bot.sendMessage(TELEGRAM_CHAT_ID, f"🚨 *ALERT*\n{message}", parse_mode="Markdown")
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                bot.sendPhoto(TELEGRAM_CHAT_ID, f)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False