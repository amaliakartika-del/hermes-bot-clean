# -*- coding: utf-8 -*-
"""
Hermes AI Telegram Bot
Powered by Nous Research via OpenRouter
"""

import os
import logging
from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from openai import AsyncOpenAI

# Load konfigurasi dari environment
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("HERMES_MODEL", "nousresearch/hermes-3-llama-3.1-70b")
BOT_NAME = os.getenv("BOT_NAME", "Hermes")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah Hermes, asisten AI dari Nous Research. "
    "Selalu jawab dalam Bahasa Indonesia kecuali diminta bahasa lain."
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Setup OpenAI client (OpenRouter)
ai_client = AsyncOpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/amaliakartika-del/hermes-bot",
        "X-Title": "Hermes Telegram Bot",
    }
)

# Simpan riwayat percakapan per user
chat_history: dict[int, list[dict]] = {}

# Daftar model yang tersedia
MODELS = {
    "1": ("nousresearch/hermes-3-llama-3.1-70b",  "Hermes 3 70B (Terbaik)"),
    "2": ("nousresearch/hermes-3-llama-3.1-8b",   "Hermes 3 8B (Cepat)"),
    "3": ("nousresearch/deephermes-3-llama-3-8b-preview:free", "DeepHermes 3 (GRATIS)"),
}
user_model: dict[int, str] = {}


# ─── Helper Functions ─────────────────────────────────────

def allowed(user_id: int) -> bool:
    if ALLOWED_USER_ID == 0:
        return True
    return user_id == ALLOWED_USER_ID


def get_history(user_id: int) -> list[dict]:
    if user_id not in chat_history:
        chat_history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return chat_history[user_id]


def get_model(user_id: int) -> str:
    return user_model.get(user_id, MODEL)


async def send_long_message(update: Update, text: str):
    """Kirim pesan panjang dengan memotong per 4000 karakter."""
    max_len = 4000
    if len(text) <= max_len:
        try:
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text)
        return

    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, max_len) or max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip()

    for i, part in enumerate(parts):
        prefix = f"*[{i+1}/{len(parts)}]*\n" if len(parts) > 1 else ""
        try:
            await update.message.reply_text(prefix + part, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(prefix + part)


# ─── Command Handlers ─────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not allowed(user.id):
        await update.message.reply_text("Akses ditolak.")
        return

    chat_history[user.id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    model_name = get_model(user.id)
    label = next((v[1] for v in MODELS.values() if v[0] == model_name), model_name)

    text = (
        f"Halo *{user.first_name}*! Saya *{BOT_NAME}*\n\n"
        f"Model: `{label}`\n"
        f"Provider: Nous Research via OpenRouter\n\n"
        f"Kirim pesan untuk mulai ngobrol!\n\n"
        f"Perintah:\n"
        f"/new - Reset percakapan\n"
        f"/model - Ganti model AI\n"
        f"/status - Info bot\n"
        f"/help - Bantuan"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_new(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    uid = update.effective_user.id
    chat_history[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    await update.message.reply_text("Percakapan direset. Mulai topik baru!")


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    uid = update.effective_user.id
    current = get_model(uid)
    lines = ["*Pilih Model:*\n"]
    for k, (mid, label) in MODELS.items():
        mark = ">" if mid == current else " "
        lines.append(f"`{k}` {mark} {label}")
    lines.append("\nBalas dengan nomor (contoh: `2`)")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    ctx.user_data["choosing_model"] = True


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    uid = update.effective_user.id
    history = get_history(uid)
    mid = get_model(uid)
    label = next((v[1] for v in MODELS.values() if v[0] == mid), mid)
    msg_count = len([m for m in history if m["role"] != "system"])

    text = (
        f"*Status Bot*\n\n"
        f"Model: `{label}`\n"
        f"Pesan sesi ini: `{msg_count}`\n"
        f"User ID: `{uid}`\n"
        f"Provider: OpenRouter"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    text = (
        "*Bantuan Hermes Bot*\n\n"
        "/start - Mulai bot\n"
        "/new - Reset percakapan\n"
        "/model - Ganti model AI\n"
        "/status - Info bot\n"
        "/help - Bantuan ini\n\n"
        "Kirim pesan biasa untuk chat dengan Hermes!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ─── Message Handler ──────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not allowed(user.id):
        await update.message.reply_text("Akses ditolak.")
        return

    text = update.message.text.strip()

    # Handle pemilihan model
    if ctx.user_data.get("choosing_model") and text in MODELS:
        mid, label = MODELS[text]
        user_model[user.id] = mid
        ctx.user_data["choosing_model"] = False
        await update.message.reply_text(f"Model diganti ke: *{label}*", parse_mode="Markdown")
        return
    ctx.user_data["choosing_model"] = False

    # Tambah ke history
    history = get_history(user.id)
    history.append({"role": "user", "content": text})

    # Kirim typing indicator
    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        response = await ai_client.chat.completions.create(
            model=get_model(user.id),
            messages=history,
            max_tokens=2048,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        # Batasi history max 40 pesan
        if len(history) > 41:
            chat_history[user.id] = [history[0]] + history[-40:]

        await send_long_message(update, reply)

    except Exception as e:
        logger.error(f"Error: {e}")
        if history and history[-1]["role"] == "user":
            history.pop()
        await update.message.reply_text(
            f"Terjadi error: `{str(e)[:150]}`\n\nCoba lagi atau /new untuk reset.",
            parse_mode="Markdown"
        )


# ─── Main ─────────────────────────────────────────────────

async def on_start(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start",  "Mulai bot"),
        BotCommand("new",    "Reset percakapan"),
        BotCommand("model",  "Ganti model AI"),
        BotCommand("status", "Info bot"),
        BotCommand("help",   "Bantuan"),
    ])
    logger.info("Bot siap!")


def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN belum diisi!")
        return
    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY belum diisi!")
        return

    logger.info(f"Starting {BOT_NAME} dengan model {MODEL}")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(on_start)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("new",    cmd_new))
    app.add_handler(CommandHandler("model",  cmd_model))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot berjalan...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
