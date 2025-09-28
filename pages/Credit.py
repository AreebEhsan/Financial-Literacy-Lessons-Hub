import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage, SystemMessage

# ---------------- CONFIG ----------------
MODEL_NAME = "gpt-3.5-turbo"
TOTAL_QUESTIONS = 6
MASTERY_THRESHOLD = 3
# Path to your saved transcript file
TRANSCRIPT_PATH = os.path.join("financial_texts", "Credit.txt")
# Optional YouTube video to embed
VIDEO_ID = "2WLZY7aJ7Ak"

# --- UNIQUE KEYS for this page ---
LESSON_KEY      = "credit_lesson"
QUESTION_KEY    = "credit_question_num"
CONVO_KEY       = "credit_conversation"
FINISHED_KEY    = "credit_finished"
SCORE_KEY       = "credit_score"

# -------------- HELPERS -----------------
def load_local_text(path: str) -> str:
    """Read the saved transcript text file once."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Could not load lesson text: {e}")
        return ""

def summarize_lesson(raw_text: str, api_key: str) -> str:
    """Summarize a long lesson into a short quiz-ready version."""
    llm = OpenAI(api_key=api_key, temperature=0, model="gpt-3.5-turbo-instruct")
    max_chars = 3000
    chunks = [raw_text[i:i + max_chars] for i in range(0, len(raw_text), max_chars)]
    bullets = []
    for ch in chunks:
        bullets.append(llm.invoke(
            f"Summarize this financial lesson chunk in 4–5 bullet points:\n\n{ch}"
        ))
    combined = "\n".join(str(b) for b in bullets)
    final_prompt = (
        "Combine these bullet points into a single concise financial lesson "
        "summary (~300 words):\n\n" + combined
    )
    return llm.invoke(final_prompt)

def init_state():
    defaults = {
        QUESTION_KEY: 0,
        CONVO_KEY: [],
        FINISHED_KEY: False,
        SCORE_KEY: 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def ask_question(llm: ChatOpenAI, lesson: str, num: int) -> str:
    asked = [t["content"] for t in st.session_state[CONVO_KEY] if t["role"] == "assistant"]
    msgs = [
        SystemMessage(content="You are a friendly finance tutor."),
        HumanMessage(content=(
            f"Here is the financial lesson:\n{lesson}\n\n"
            f"Previous questions: {asked}\n"
            f"Ask question #{num} of {TOTAL_QUESTIONS} to check understanding. "
            "Do NOT repeat previous questions and do NOT give the answer."
        ))
    ]
    return llm.invoke(msgs).content

def grade_answer(llm: ChatOpenAI, question: str, user_answer: str, lesson: str) -> str:
    trigger_words = ["answer", "give", "tell", "idk", "don't know"]
    if any(w in user_answer.lower() for w in trigger_words):
        msgs = [
            SystemMessage(content="You are a helpful finance tutor."),
            HumanMessage(content=(
                f"Financial lesson:\n{lesson}\n\n"
                f"Provide the correct answer to this question and a short explanation:\n{question}"
            ))
        ]
        return llm.invoke(msgs).content

    msgs = [
        SystemMessage(content="You are a strict but helpful finance tutor."),
        HumanMessage(content=(
            f"Financial lesson:\n{lesson}\n\n"
            f"Question: {question}\n"
            f"Student Answer: {user_answer}\n\n"
            "Say Correct or Incorrect and give a very short explanation or hint."
        ))
    ]
    return llm.invoke(msgs).content

def run_quiz():
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=api_key, temperature=0.6, model=MODEL_NAME)

    for turn in st.session_state[CONVO_KEY]:
        st.chat_message(turn["role"]).write(turn["content"])

    if st.session_state[QUESTION_KEY] <= TOTAL_QUESTIONS:
        prompt = (
            "Type your answer and press Enter"
            if st.session_state[CONVO_KEY]
            else "Press Enter to receive the first question"
        )
        user_msg = st.chat_input(prompt)

        if not st.session_state[CONVO_KEY] and user_msg is None:
            q = ask_question(llm, st.session_state[LESSON_KEY], 1)
            st.session_state[CONVO_KEY].append({"role": "assistant", "content": q})
            st.chat_message("assistant").write(q)
            return

        if user_msg:
            last_q = st.session_state[CONVO_KEY][-1]["content"]
            st.session_state[CONVO_KEY].append({"role": "user", "content": user_msg})
            st.chat_message("user").write(user_msg)

            feedback = grade_answer(llm, last_q, user_msg, st.session_state[LESSON_KEY])
            st.session_state[CONVO_KEY].append({"role": "assistant", "content": feedback})
            st.chat_message("assistant").write(feedback)

            if "correct" in feedback.lower():
                st.session_state[SCORE_KEY] += 1

            st.session_state[QUESTION_KEY] += 1

            if (st.session_state[QUESTION_KEY] >= MASTERY_THRESHOLD
                and st.session_state[SCORE_KEY] >= MASTERY_THRESHOLD
                and not st.session_state[FINISHED_KEY]):
                mastery_msg = (
                    "🎯 Fantastic! You’ve demonstrated mastery of this financial lesson. "
                    "You can continue for more practice or stop here."
                )
                st.session_state[CONVO_KEY].append({"role": "assistant", "content": mastery_msg})
                st.chat_message("assistant").write(mastery_msg)

            if st.session_state[QUESTION_KEY] <= TOTAL_QUESTIONS:
                next_q = ask_question(llm,
                                      st.session_state[LESSON_KEY],
                                      st.session_state[QUESTION_KEY])
                st.session_state[CONVO_KEY].append({"role": "assistant", "content": next_q})
                st.chat_message("assistant").write(next_q)
            else:
                st.session_state[FINISHED_KEY] = True

# ---------------- MAIN APP --------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Finance Lesson Quiz-Bot", page_icon="🎥")
    st.title("🎥 Lesson 2: Understanding Credit")
    st.caption("Read the saved transcript below, then answer questions to test your understanding.")

    init_state()

    # Optional: embed the original video if you still want to show it
    st.video(f"https://www.youtube.com/embed/{VIDEO_ID}")

    if LESSON_KEY not in st.session_state:
        with st.spinner("Preparing for financial quiz session..."):
            raw = load_local_text(TRANSCRIPT_PATH)
            if raw:
                api_key = os.getenv("OPENAI_API_KEY")
                summary = summarize_lesson(raw, api_key)
                st.session_state[LESSON_KEY] = summary
                st.success("Quiz Available! Click *Start Quiz* to begin.")

    if LESSON_KEY in st.session_state and not st.session_state[FINISHED_KEY]:
        if st.button("Start Quiz", disabled=st.session_state[QUESTION_KEY] > 0):
            st.session_state[QUESTION_KEY] = 1
            st.session_state[CONVO_KEY] = []
            st.session_state[FINISHED_KEY] = False
            st.session_state[SCORE_KEY] = 0

        if st.session_state[QUESTION_KEY] > 0:
            run_quiz()

    if st.session_state[FINISHED_KEY]:
        st.success("🎉 Quiz complete! Thanks for playing.")

if __name__ == "__main__":
    main()
