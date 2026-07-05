# -*- coding: utf-8 -*-
"""
Hermes AI Agent Telegram Bot
Powered by Nous Research via OpenRouter

Agent capabilities:
- Web search (DuckDuckGo)
- Wikipedia lookup
- Calculator
- Date & Time
- URL reader
"""

import os
import json
import logging
import datetime
import math
import httpx
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

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID  = int(os.getenv("TELEGRAM_USER_ID", "0"))
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
MODEL            = os.getenv("HERMES_MODEL", "nousresearch/hermes-3-llama-3.1-405b")
BOT_NAME         = os.getenv("BOT_NAME", "Hermes")
SYSTEM_PROMPT    = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah Hermes, AI Agent cerdas dari Nous Research. "
    "Kamu bisa menggunakan tools untuk mencari informasi, menghitung, dan menjawab pertanyaan. "
    "PENTING: Selalu gunakan tool web_search atau news_search ketika ditanya tentang berita, informasi terkini, atau fakta. "
    "WAJIB: Selalu sertakan link/URL sumber dalam jawabanmu jika tersedia dari hasil pencarian. "
    "Format link seperti ini: [Judul Artikel](URL). "
    "Jawab dalam Bahasa Indonesia kecuali diminta bahasa lain."
)

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ai_client = AsyncOpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/amaliakartika-del/hermes-bot-clean",
        "X-Title": "Hermes AI Agent",
    }
)

chat_history: dict[int, list[dict]] = {}

MODELS = {
    "1": ("nousresearch/hermes-3-llama-3.1-405b", "Hermes 3 405B (Terkuat)"),
    "2": ("nousresearch/hermes-3-llama-3.1-70b",  "Hermes 3 70B (Seimbang)"),
    "3": ("nousresearch/hermes-3-llama-3.1-8b",   "Hermes 3 8B (Cepat)"),
    "4": ("nousresearch/deephermes-3-llama-3-8b-preview:free", "DeepHermes 3 (GRATIS)"),
}
user_model: dict[int, str] = {}


# ═══════════════════════════════════════════════
#  TOOLS DEFINITION
# ═══════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": "Cari berita terbaru dari Google News. Selalu gunakan tool ini untuk pertanyaan tentang berita, kejadian terkini, dan informasi hari ini. Mengembalikan judul, ringkasan, tanggal, dan LINK berita.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Kata kunci berita"},
                    "language": {"type": "string", "description": "Kode bahasa: 'id' Indonesia, 'en' Inggris", "default": "id"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Cari informasi umum di internet. Mengembalikan hasil pencarian dengan judul, deskripsi, dan LINK sumber.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Kata kunci pencarian"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Cari ringkasan artikel dari Wikipedia untuk informasi umum, definisi, atau penjelasan topik.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topik yang dicari"},
                    "language": {"type": "string", "description": "Kode bahasa: 'id' atau 'en'", "default": "id"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Hitung ekspresi matematika. Mendukung operasi dasar, trigonometri, logaritma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Ekspresi matematika, contoh: '2**10', 'math.sqrt(144)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Dapatkan tanggal dan waktu saat ini.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Zona waktu, contoh: 'Asia/Jakarta'", "default": "Asia/Jakarta"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Baca konten dari sebuah URL/website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL yang ingin dibaca"}
                },
                "required": ["url"]
            }
        }
    }
]


# ═══════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════

async def tool_news_search(query: str, language: str = "id") -> str:
    """Cari berita via Google News RSS - selalu return link asli."""
    try:
        import xml.etree.ElementTree as ET
        import urllib.parse

        lang_map = {"id": ("id", "ID", "ID:id"), "en": ("en", "US", "en-US:en")}
        hl, gl, ceid = lang_map.get(language, ("id", "ID", "ID:id"))
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={ceid}"

        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(url)
            r.raise_for_status()

        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:6]

        if not items:
            return f"Tidak ada berita ditemukan untuk '{query}'."

        lines = [f"*Berita terbaru: '{query}'*\n"]
        for i, item in enumerate(items, 1):
            title   = item.findtext("title", "Tanpa judul")
            link    = item.findtext("link", "")
            pubdate = item.findtext("pubDate", "")[:16]
            source  = item.findtext("source", "")
            # Bersihkan judul dari nama sumber
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            lines.append(
                f"{i}. *{title}*\n"
                f"   Sumber: {source} | {pubdate}\n"
                f"   Link: {link}\n"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Error pencarian berita: {str(e)}"


async def tool_web_search(query: str) -> str:
    """Cari info umum via Brave Search API (tanpa API key, pakai endpoint publik)."""
    try:
        import urllib.parse
        encoded = urllib.parse.quote(query)
        # Gunakan DuckDuckGo HTML endpoint
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}) as client:
            r = await client.post("https://html.duckduckgo.com/html/", data={"q": query})

        import re
        # Ambil judul dan link dari hasil HTML
        titles = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', r.text)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', r.text)

        if not titles:
            # Fallback: coba news search
            return await tool_news_search(query)

        lines = [f"*Hasil pencarian: '{query}'*\n"]
        for i, ((link, title), snippet) in enumerate(zip(titles[:5], snippets[:5] + [""]*5), 1):
            # Decode URL jika encoded
            if link.startswith("/l/?"):
                uddg = re.search(r'uddg=([^&]+)', link)
                if uddg:
                    import urllib.parse
                    link = urllib.parse.unquote(uddg.group(1))
            lines.append(
                f"{i}. *{title.strip()}*\n"
                f"   {snippet.strip()[:200]}\n"
                f"   Link: {link}\n"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Error pencarian web: {str(e)}"



async def tool_wikipedia(topic: str, language: str = "id") -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Search first
            search_r = await client.get(
                f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{httpx.QueryParams({'title': topic})}",
            )
            # Try direct lookup
            r = await client.get(
                f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            )
            if r.status_code == 200:
                data = r.json()
                title = data.get("title", topic)
                extract = data.get("extract", "Tidak ada konten")
                url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                result = f"**{title}**\n\n{extract[:1500]}"
                if url:
                    result += f"\n\nSelengkapnya: {url}"
                return result
            else:
                # Try English fallback
                r2 = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
                )
                if r2.status_code == 200:
                    data = r2.json()
                    return f"**{data.get('title')}** (Wikipedia EN)\n\n{data.get('extract', '')[:1500]}"
                return f"Artikel '{topic}' tidak ditemukan di Wikipedia."
    except Exception as e:
        return f"Error Wikipedia: {str(e)}"


def tool_calculator(expression: str) -> str:
    try:
        # Allowed safe functions
        safe_dict = {
            "math": math,
            "abs": abs, "round": round,
            "min": min, "max": max,
            "sum": sum, "pow": pow,
            "int": int, "float": float,
        }
        # Block dangerous ops
        blocked = ["import", "exec", "eval", "open", "os", "__"]
        for b in blocked:
            if b in expression:
                return f"Operasi '{b}' tidak diizinkan."
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error kalkulasi: {str(e)}"


def tool_get_datetime(timezone: str = "Asia/Jakarta") -> str:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.datetime.now(tz)
        return (
            f"Waktu saat ini di {timezone}:\n"
            f"Tanggal: {now.strftime('%A, %d %B %Y')}\n"
            f"Waktu: {now.strftime('%H:%M:%S %Z')}"
        )
    except Exception:
        now = datetime.datetime.utcnow()
        return f"Waktu UTC: {now.strftime('%A, %d %B %Y %H:%M:%S UTC')}"


async def tool_read_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            # Basic HTML stripping
            text = r.text
            import re
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:3000] + ("..." if len(text) > 3000 else "")
    except Exception as e:
        return f"Error membaca URL: {str(e)}"


async def execute_tool(name: str, args: dict) -> str:
    if name == "news_search":
        return await tool_news_search(args.get("query", ""), args.get("language", "id"))
    elif name == "web_search":
        return await tool_web_search(args.get("query", ""))
    elif name == "wikipedia_search":
        return await tool_wikipedia(args.get("topic", ""), args.get("language", "id"))
    elif name == "calculator":
        return tool_calculator(args.get("expression", ""))
    elif name == "get_datetime":
        return tool_get_datetime(args.get("timezone", "Asia/Jakarta"))
    elif name == "read_url":
        return await tool_read_url(args.get("url", ""))
    else:
        return f"Tool '{name}' tidak dikenal."


# ═══════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def allowed(user_id: int) -> bool:
    return ALLOWED_USER_ID == 0 or user_id == ALLOWED_USER_ID

def get_history(user_id: int) -> list[dict]:
    if user_id not in chat_history:
        chat_history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return chat_history[user_id]

def get_model(user_id: int) -> str:
    return user_model.get(user_id, MODEL)

async def send_message_safe(update: Update, text: str, parse_mode: str = "Markdown"):
    max_len = 4000
    if len(text) <= max_len:
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
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
            await update.message.reply_text(prefix + part, parse_mode=parse_mode)
        except Exception:
            await update.message.reply_text(prefix + part)


# ═══════════════════════════════════════════════
#  AGENT LOOP
# ═══════════════════════════════════════════════

async def run_agent(user_id: int, user_message: str, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> str:
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    model = get_model(user_id)
    max_iterations = 5  # Batas loop agent

    for iteration in range(max_iterations):
        response = await ai_client.chat.completions.create(
            model=model,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
            temperature=0.7,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Simpan response ke history
        history.append(msg.model_dump(exclude_none=True))

        # Jika agent selesai (tidak mau pakai tool lagi)
        if finish_reason == "stop" or not msg.tool_calls:
            final_answer = msg.content or "Maaf, tidak ada jawaban."
            # Batasi history
            if len(history) > 41:
                chat_history[user_id] = [history[0]] + history[-40:]
            return final_answer

        # Eksekusi tools yang diminta agent
        tool_results = []
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except Exception:
                tool_args = {}

            # Tampilkan ke user tool apa yang dipakai
            await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            tool_status = {
                "news_search":       f"Mencari berita: _{tool_args.get('query', '')}_",
                "web_search":        f"Mencari web: _{tool_args.get('query', '')}_",
                "wikipedia_search":  f"Membaca Wikipedia: _{tool_args.get('topic', '')}_",
                "calculator":        f"Menghitung: `{tool_args.get('expression', '')}`",
                "get_datetime":      "Mengecek waktu...",
                "read_url":          f"Membaca: _{tool_args.get('url', '')}_",
            }
            status_msg = tool_status.get(tool_name, f"Menggunakan tool: `{tool_name}`")
            try:
                await update.message.reply_text(f"_[Agent]_ {status_msg}", parse_mode="Markdown")
            except Exception:
                pass

            # Jalankan tool
            result = await execute_tool(tool_name, tool_args)

            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Masukkan hasil tool ke history
        history.extend(tool_results)

    return "Maaf, agent mencapai batas iterasi. Coba pertanyaan yang lebih spesifik."


# ═══════════════════════════════════════════════
#  COMMAND HANDLERS
# ═══════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not allowed(user.id):
        await update.message.reply_text("Akses ditolak.")
        return
    chat_history[user.id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    model_id = get_model(user.id)
    label = next((v[1] for v in MODELS.values() if v[0] == model_id), model_id)
    text = (
        f"Halo *{user.first_name}*! Saya *{BOT_NAME} Agent*\n\n"
        f"Model: `{label}`\n\n"
        f"Saya adalah AI Agent yang bisa:\n"
        f"- Mencari informasi di internet\n"
        f"- Membaca artikel Wikipedia\n"
        f"- Menghitung matematika\n"
        f"- Mengecek waktu & tanggal\n"
        f"- Membaca konten website\n\n"
        f"Tanya apa saja!\n\n"
        f"/new - Reset percakapan\n"
        f"/model - Ganti model\n"
        f"/tools - Lihat daftar tools\n"
        f"/status - Info agent"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_tools(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    text = (
        "*Tools yang dimiliki Hermes Agent:*\n\n"
        "- *web\\_search* — Cari info terbaru di internet\n"
        "- *wikipedia\\_search* — Baca artikel Wikipedia\n"
        "- *calculator* — Hitung ekspresi matematika\n"
        "- *get\\_datetime* — Cek waktu & tanggal\n"
        "- *read\\_url* — Baca konten website\n\n"
        "Agent akan otomatis memilih tool yang tepat!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_new(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    chat_history[update.effective_user.id] = [{"role": "system", "content": SYSTEM_PROMPT}]
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
    lines.append("\nBalas dengan nomor (contoh: `1`)")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    ctx.user_data["choosing_model"] = True


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    uid = update.effective_user.id
    history = get_history(uid)
    mid = get_model(uid)
    label = next((v[1] for v in MODELS.values() if v[0] == mid), mid)
    msg_count = len([m for m in history if m.get("role") not in ("system", "tool")])
    text = (
        f"*Status Hermes Agent*\n\n"
        f"Model: `{label}`\n"
        f"Pesan sesi ini: `{msg_count}`\n"
        f"Tools tersedia: `5`\n"
        f"User ID: `{uid}`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        return
    text = (
        "*Bantuan Hermes Agent*\n\n"
        "/start - Mulai agent\n"
        "/new - Reset percakapan\n"
        "/model - Ganti model AI\n"
        "/tools - Lihat daftar tools\n"
        "/status - Info agent\n"
        "/help - Bantuan ini\n\n"
        "*Contoh pertanyaan:*\n"
        "- Berita terbaru tentang AI\n"
        "- Siapa presiden Indonesia?\n"
        "- Hitung 15% dari 2.5 juta\n"
        "- Sekarang jam berapa di Tokyo?\n"
        "- Jelaskan apa itu blockchain"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ═══════════════════════════════════════════════
#  MESSAGE HANDLER
# ═══════════════════════════════════════════════

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not allowed(user.id):
        await update.message.reply_text("Akses ditolak.")
        return

    text = update.message.text.strip()

    # Handle pilihan model
    if ctx.user_data.get("choosing_model") and text in MODELS:
        mid, label = MODELS[text]
        user_model[user.id] = mid
        ctx.user_data["choosing_model"] = False
        await update.message.reply_text(f"Model diganti ke: *{label}*", parse_mode="Markdown")
        return
    ctx.user_data["choosing_model"] = False

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply = await run_agent(user.id, text, update, ctx)
        await send_message_safe(update, reply)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        history = get_history(user.id)
        if history and history[-1].get("role") == "user":
            history.pop()
        await update.message.reply_text(
            f"Terjadi error: `{str(e)[:150]}`\n\nCoba lagi atau /new untuk reset.",
            parse_mode="Markdown"
        )


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

async def on_start(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start",  "Mulai agent"),
        BotCommand("new",    "Reset percakapan"),
        BotCommand("model",  "Ganti model AI"),
        BotCommand("tools",  "Lihat daftar tools"),
        BotCommand("status", "Info agent"),
        BotCommand("help",   "Bantuan"),
    ])
    logger.info(f"{BOT_NAME} Agent siap dengan {len(TOOLS)} tools!")


def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN belum diisi!")
        return
    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY belum diisi!")
        return

    logger.info(f"Starting {BOT_NAME} Agent | Model: {MODEL} | Tools: {len(TOOLS)}")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(on_start)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("new",    cmd_new))
    app.add_handler(CommandHandler("model",  cmd_model))
    app.add_handler(CommandHandler("tools",  cmd_tools))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Agent berjalan...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
