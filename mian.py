# Telegram imports
from typing import Final
from contextvars import ContextVar
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# AI imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Json imports
import json
import os
from datetime import datetime, timedelta

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

# Stores the live back-and-forth for the current process.
# This is short-term memory only and is lost if the bot restarts.
user_histories = {}

# The tools do not receive Telegram user IDs directly, so this context variable lets
# tool calls know which user triggered the current request while the agent is running.
active_user_id = ContextVar("active_user_id", default=None)

# This is the stable personality layer that gets injected as a system message on every turn.
# It gives the bot a consistent voice even when the user asks unrelated questions.
STOIC_PERSONA_PROMPT = """You are Future Me, a wise, stoic, blunt assistant.

Core style:
- Speak with calm authority.
- Be concise, direct, and practical.
- No fluff, no hype, no empty encouragement.
- Be inspirational only through discipline, clarity, patience, and high standards.
- Never sound like a motivational speaker.

Behavior:
- Tell the truth plainly, even when it is uncomfortable.
- Challenge weak assumptions directly and explain why they are weak.
- Prefer simple, durable advice over clever advice.
- Give actionable answers.
- Do not ramble.

Response rules:
- Use short paragraphs or short lists when useful.
- If the user is vague, ask the minimum needed to move forward.
- If a plan is weak, say so clearly and propose a stronger one.
- If a request involves reminders or scheduling, be precise about date and time.
"""

# Reminder Manager
class ReminderManager:
    def __init__(self, filename="reminders.json"):
        # Reminders are stored in a JSON file so they survive process restarts.
        self.filename = filename
        self.load()
    
    def load(self):
        try:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
    
    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_reminder(self, date: str, text: str, time: str = "", category: str = "General"):
        """Add reminder: date='2026-03-10', time='10:30', category='Work'"""
        if date not in self.data:
            self.data[date] = []
        
        reminder = {
            "id": f"rem_{datetime.now().timestamp()}",
            "text": text,
            "time": time,
            "category": category,
            "completed": False
        }
        self.data[date].append(reminder)
        self.save()
    
    def get_reminders_for_time(self, date, time, category=None):
        """Get all reminders for a specific date/time, optionally filtered by category"""
        reminders = []
        if date in self.data:
            for reminder in self.data[date]:
                if reminder["time"] == time and not reminder["completed"]:
                    if category is None or reminder.get("category", "General") == category:
                        reminders.append(reminder)
        return reminders
    
    def complete_reminder(self, date, reminder_id):
        """Mark reminder as complete"""
        if date in self.data:
            for reminder in self.data[date]:
                if reminder["id"] == reminder_id:
                    reminder["completed"] = True
                    self.save()


class GoalMemoryManager:
    def __init__(self, filename="goal_memories.json"):
        # This file acts as long-term memory for goals, motives, and identity-level statements.
        self.filename = filename
        self.load()

    def load(self):
        try:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def _get_user_record(self, user_id: int):
        # Each Telegram user gets their own memory bucket so conversations do not mix.
        user_key = str(user_id)
        if user_key not in self.data:
            self.data[user_key] = {
                "summary": "",
                "goals": [],
                "updated_at": ""
            }
        return self.data[user_key]

    def add_goal_memory(self, user_id: int, goal: str, why: str = "", details: str = "") -> None:
        # Normalize user input before comparing or storing it so small formatting changes
        # do not create duplicate goal records.
        user_record = self._get_user_record(user_id)
        normalized_goal = goal.strip()
        normalized_why = why.strip()
        normalized_details = details.strip()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # If the same goal already exists, update the existing record instead of adding noise.
        for item in user_record["goals"]:
            if item["goal"].strip().lower() == normalized_goal.lower():
                if normalized_why:
                    item["why"] = normalized_why
                if normalized_details:
                    # New details are appended so later conversations can enrich the same goal.
                    existing_details = item.get("details", "").strip()
                    if existing_details and normalized_details.lower() not in existing_details.lower():
                        item["details"] = f"{existing_details} | {normalized_details}"
                    elif not existing_details:
                        item["details"] = normalized_details
                item["updated_at"] = timestamp
                user_record["updated_at"] = timestamp
                self.save()
                return

        user_record["goals"].append({
            "goal": normalized_goal,
            "why": normalized_why,
            "details": normalized_details,
            "created_at": timestamp,
            "updated_at": timestamp,
        })
        user_record["updated_at"] = timestamp
        self.save()

    def set_summary(self, user_id: int, summary: str) -> None:
        user_record = self._get_user_record(user_id)
        user_record["summary"] = summary.strip()
        user_record["updated_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save()

    def format_goal_context(self, user_id: int) -> str:
        # This converts stored JSON memory into plain text that can be injected into the
        # system prompt for the model to use during normal conversation.
        user_record = self.data.get(str(user_id))
        if not user_record:
            return ""

        lines = []
        summary = user_record.get("summary", "").strip()
        if summary:
            lines.append(f"Summary: {summary}")

        goals = user_record.get("goals", [])
        if goals:
            lines.append("Goals:")
            for item in goals[-5:]:
                goal_line = f"- Goal: {item['goal']}"
                if item.get("why"):
                    goal_line += f" | Why: {item['why']}"
                if item.get("details"):
                    goal_line += f" | Details: {item['details']}"
                lines.append(goal_line)

        return "\n".join(lines)

    def has_data(self, user_id: int) -> bool:
        return str(user_id) in self.data and bool(self.data[str(user_id)].get("goals") or self.data[str(user_id)].get("summary"))




# Initialize AI Agent
# Main conversational model used for replies.
model = ChatOpenAI(temperature=0.3)

# Separate low-temperature model used for structured extraction.
# Keeping extraction deterministic reduces bad JSON and inconsistent memory saves.
goal_extractor_model = ChatOpenAI(temperature=0)

@tool
def arithmetic(a: float, b: float, operation: str) -> str:
    """Performs arithmetic: addition, subtraction, multiplication, division, exponents, roots, logarithms, modulus and absolute value."""
    print("Arithmetic tool has been called.")
    op = operation.lower()
    if op in ["addition", "add", "plus"]:
        return f"The sum of {a} and {b} is {a + b}"
    elif op in ["subtraction", "subtract", "minus"]:
        return f"The difference between {a} and {b} is {a - b}"
    elif op in ["multiplication", "multiply", "times"]:
        return f"The product of {a} and {b} is {a * b}"
    elif op in ["division", "divide"]:
        if b == 0:
            return "Cannot divide by zero"
        return f"The division of {a} by {b} is {a / b}"
    elif op in ["exponent", "power"]:
        return f"{a} to the power of {b} is {a ** b}"
    elif op in ["modulus", "mod"]:
        return f"The modulus of {a} and {b} is {a % b}"
    elif op in ["absolute value", "abs"]:
        return f"The absolute value of {a} is {abs(a)}"
    else:
        raise ValueError("Operation not supported.")




# Initialize ReminderManager
# Store the JSON files next to this script so deployment stays simple.
reminder_manager = ReminderManager(os.path.join(os.path.dirname(__file__), "reminders.json"))
goal_memory_manager = GoalMemoryManager(os.path.join(os.path.dirname(__file__), "goal_memories.json"))

def parse_relative_date(relative_date: str) -> str:
    """Convert relative dates like 'today', 'tomorrow', 'in 3 days' to actual date format 'YYYY-MM-DD'"""
    # This lets users speak naturally instead of forcing exact dates every time.
    today = datetime.now().date()
    relative_date_lower = relative_date.lower().strip()
    
    if relative_date_lower == "today":
        return today.strftime('%Y-%m-%d')
    elif relative_date_lower == "tomorrow":
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    elif relative_date_lower.startswith("in "):
        try:
            days = int(relative_date_lower.split()[1])
            return (today + timedelta(days=days)).strftime('%Y-%m-%d')
        except:
            return today.strftime('%Y-%m-%d')
    else:
        # If it's already in YYYY-MM-DD format, return it
        return relative_date

@tool
def add_reminder(date: str, text: str, time: str = "", category: str = "General") -> str:
    """Add a reminder with relative dates and categories. Format: date='today' or 'tomorrow' or 'in 3 days' or '2026-03-10', text='Buy milk', time='14:30' (optional), category='Work' or 'Personal' or 'Shopping' (optional)"""
    print("Add reminder tool has been called.")
    actual_date = parse_relative_date(date)
    reminder_manager.add_reminder(actual_date, text, time, category)
    time_str = f" at {time}" if time else ""
    category_str = f" [{category}]" if category != "General" else ""
    return f"✓ Reminder set for {actual_date}{time_str}{category_str}: {text}"

@tool
def list_reminders(date: str, category: str = "") -> str:
    """List all reminders for a specific date, optionally filtered by category. Use relative dates like 'today', 'tomorrow' or exact date '2026-03-10'. Category can be 'Work', 'Personal', 'Shopping', etc."""
    print("List reminders tool has been called.")
    actual_date = parse_relative_date(date)
    reminders = reminder_manager.data.get(actual_date, [])
    
    # Filter by category if specified
    if category:
        reminders = [r for r in reminders if r.get("category", "General") == category]
    
    if not reminders:
        filter_str = f" in category {category}" if category else ""
        return f"No reminders for {actual_date}{filter_str}"
    
    result = []
    for r in reminders:
        time_str = f"{r['time']}: " if r['time'] else ""
        category_str = f" [{r.get('category', 'General')}]" if r.get('category', 'General') != 'General' else ""
        result.append(f"• {time_str}{r['text']}{category_str}")
    return "\n".join(result)

@tool
def complete_reminder(date: str, reminder_id: str) -> str:
    """Mark a reminder as complete. Use relative dates like 'today' or exact date '2026-03-10'. Provide the reminder ID."""
    print("Complete reminder tool has been called.")
    actual_date = parse_relative_date(date)
    reminder_manager.complete_reminder(actual_date, reminder_id)
    return f"✓ Reminder completed"

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    current = datetime.now()
    return f"Current date and time: {current.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def save_goal_memory(goal: str, why: str = "", details: str = "") -> str:
    """Save or update a user's long-term goal, why they want it, and any useful details. Use this when the user talks about what they want from themselves or why it matters."""
    # The active user is supplied through the context variable set in handle_response.
    current_user_id = active_user_id.get()
    if current_user_id is None:
        return "I could not determine which user to save this goal for."

    goal_memory_manager.add_goal_memory(current_user_id, goal, why, details)
    return "Goal memory saved. I will use it in future conversations."


@tool
def get_goal_memory() -> str:
    """Get the saved long-term goals and reasons for the current user."""
    current_user_id = active_user_id.get()
    if current_user_id is None:
        return "I could not determine which user to load goal memory for."

    context = goal_memory_manager.format_goal_context(current_user_id)
    if not context:
        return "No long-term goal memory is saved yet."

    return context

tools = [
    arithmetic,
    add_reminder,
    list_reminders,
    complete_reminder,
    get_current_datetime,
    save_goal_memory,
    get_goal_memory,
]

# LangGraph wraps the model plus tools into a single agent that can decide when to call tools
# and when to answer directly.
agent_executor = create_react_agent(model, tools)




# Telegram Bot Setup
TOKEN: Final = require_env("TELEGRAM_BOT_TOKEN")
BOT_USERNAME: Final = os.getenv("TELEGRAM_BOT_USERNAME", "future_me_assistant_bot").strip() or "future_me_assistant_bot"


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I am Future You. Speak plainly. I will help you think clearly, plan well, and follow through.\n\n"
        "Talk to me about your goals and why they matter. I will remember the important parts."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - begin\n"
        "/list_all - show every reminder\n\n"
        "You can ask for reminders, schedules, or direct advice.\n"
        "You can also talk through your goals, your reasons, and what kind of person you want to become."
    )

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("custom command run")

async def list_all_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lists every reminder in each category, ordered by date, separated by category, in a readable format."""
    all_reminders = reminder_manager.data
    reminders_flat = []
    for date, reminders in all_reminders.items():
        for r in reminders:
            reminders_flat.append((date, r))
    reminders_flat.sort(key=lambda x: (x[0], x[1].get('time', '')))
    from collections import defaultdict
    categories = defaultdict(list)
    for date, r in reminders_flat:
        cat = r.get('category', 'General')
        categories[cat].append((date, r))
    if not categories:
        await update.message.reply_text("You have no reminders.")
        return
    msg = "<b>All Reminders by Category</b>\n"
    for cat in sorted(categories):
        msg += f"\n<b>{cat}</b>\n"
        for date, r in categories[cat]:
            status = "✅" if r.get('completed') else "🕒"
            time_str = f"{r['time']} " if r.get('time') else ""
            msg += f"{status} {date} {time_str}- {r['text']}\n"
    await update.message.reply_text(msg, parse_mode="HTML")












    




# Responses
def build_system_prompt(user_id: int) -> str:
    # Every response starts with the base persona. If we have long-term memory for this user,
    # we append it so the bot can reconnect current conversations to old goals.
    goal_context = goal_memory_manager.format_goal_context(user_id)
    if not goal_context:
        return STOIC_PERSONA_PROMPT

    return (
        f"{STOIC_PERSONA_PROMPT}\n\n"
        "Long-term user context:\n"
        f"{goal_context}\n\n"
        "Use this memory when it is relevant. Bring it up days later when the user discusses the same themes, drifts from their stated reasons, or asks for direction. "
        "Be concrete. Do not turn every reply into a lecture."
    )


def build_message_history(user_id: int):
    """Build a typed message history with a fixed system persona."""
    # The first message is always the system prompt. After that we replay the saved chat turns
    # using the correct message types so the model can distinguish user input from its own replies.
    history = [SystemMessage(content=build_system_prompt(user_id))]

    for entry in user_histories.get(user_id, []):
        role = entry["role"]
        content = entry["content"]
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))

    return history


def extract_goal_memory(text: str):
    # This prompt asks a second model to convert free-form user language into structured JSON.
    # That keeps the memory-saving path separate from the normal conversational reply path.
    prompt = (
        "Extract long-term goal information from the user's message. "
        "Return valid JSON only with this exact schema: "
        '{"save": true or false, "summary": "short summary", "goals": [{"goal": "", "why": "", "details": ""}]}. '
        "If the message does not contain meaningful long-term goals, values, or reasons, return save=false, summary='', goals=[]. "
        "Do not include markdown or explanation.\n\n"
        f"User message: {text}"
    )

    try:
        response = goal_extractor_model.invoke([HumanMessage(content=prompt)])
        # If the model returns invalid JSON, we fail quietly instead of breaking the user reply.
        payload = json.loads(response.content)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def persist_goal_memory(user_id: int, text: str) -> None:
    # This runs on every user message. Most messages will not produce memory updates,
    # but meaningful goal statements and motives will get stored for later use.
    extracted = extract_goal_memory(text)
    if not extracted or not extracted.get("save"):
        return

    summary = extracted.get("summary", "").strip()
    if summary:
        goal_memory_manager.set_summary(user_id, summary)

    for item in extracted.get("goals", []):
        goal = item.get("goal", "").strip()
        why = item.get("why", "").strip()
        details = item.get("details", "").strip()
        if goal:
            goal_memory_manager.add_goal_memory(user_id, goal, why, details)


async def handle_response(text: str, user_id: int) -> str:
    """Send user message and history to AI agent and get response."""
    # Maintain conversation history for each user
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_histories[user_id].append({"role": "user", "content": text})

    # Persist long-term goals before generating the reply so the current message can influence
    # the system prompt immediately if it contains something important.
    persist_goal_memory(user_id, text)

    # Rebuild the prompt and chat history fresh on every turn so the latest saved memory is included.
    messages = build_message_history(user_id)
    response_text = ""

    # Expose the current user ID to any tool the agent may decide to call during this request.
    token = active_user_id.set(user_id)
    try:
        # Stream agent events so tool outputs and the final answer can be collected incrementally.
        for chunk in agent_executor.stream({"messages": messages}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response_text += message.content
        # Add bot response to history
        user_histories[user_id].append({"role": "assistant", "content": response_text})
        return response_text if response_text else "I couldn't process that request."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        active_user_id.reset(token)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Track chats that have spoken to the bot so reminder notifications know where they can send messages.
    Application.chat_ids.add(update.message.chat.id)
    message_type: str = update.message.chat.type
    text: str = update.message.text
    print(f"User({update.message.chat.id}) in {message_type}: '{text}'")

    # In group chats the bot only responds when tagged, which avoids spamming the whole group.
    if message_type == "group":
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, "").strip()
            response: str = await handle_response(new_text, update.message.chat.id)
        else:
            return
    else:
        response: str = await handle_response(text, update.message.chat.id)
    print('Bot:', response)
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")









# Patch Application to store chat_ids
# This adds a lightweight shared set onto the Application object so background tasks can
# find active chats without introducing a separate database layer.
Application.chat_ids = set()

import asyncio

async def reminder_notifier(app: Application):
    while True:
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M')
        # Check today's reminders
        reminders = reminder_manager.data.get(today_str, [])
        for r in reminders:
            if not r.get('completed') and r.get('time') == current_time:
                # Any chat that has interacted with the bot becomes eligible to receive reminder messages.
                for chat_id in app.chat_ids if hasattr(app, 'chat_ids') else []:
                    await app.bot.send_message(chat_id=chat_id, text=f"⏰ Reminder: {r['text']}")
        await asyncio.sleep(60)  # Check every minute

# Post-init callback to start notifier after event loop is running
async def post_init(application: Application):
    # The background reminder loop must start after the Telegram application event loop exists.
    application.create_task(reminder_notifier(application))

if __name__ == "__main__":
    print("Starting bot...")

    # Build the Telegram application, attach handlers, then start long-polling for updates.
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    # commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))
    app.add_handler(CommandHandler("list_all", list_all_reminders))
    # messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    # errors
    app.add_error_handler(error)
    print("Polling...")
    app.run_polling(poll_interval=3)



# Add functionality to send daily Meditations verses
# add something where A