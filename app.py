import streamlit as st
import random
import re
import json
import os
from pathlib import Path
from collections import Counter

# Force model cache into the project folder so the app behaves more like a local setup.
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_PROXY", "1")

cache_root = Path(__file__).resolve().parent / ".cache" / "huggingface"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_root))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))

PROGRESS_FILE = "progress.json"


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


STOPWORDS = {
    "the","and","is","in","to","of","a","an","it","that","this","for","on","with",
    "as","are","was","were","be","by","from","or","at","which","but","have","has","had",
}


def extract_keywords(text: str, max_keywords=8):
    words = re.findall(r"\b[a-zA-Z']{3,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    freq = Counter(words)
    return freq.most_common(max_keywords)


def split_sentences(text: str):
    # naive sentence split
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def create_quizzes_from_text(text: str, n: int):
    sentences = split_sentences(text)
    if not sentences:
        return []
    quizzes = []
    tries = 0
    while len(quizzes) < n and tries < n * 10:
        s = random.choice(sentences)
        words = re.findall(r"\b[a-zA-Z']{3,}\b", s)
        candidates = [w for w in words if w.lower() not in STOPWORDS and len(w) > 3]
        if not candidates:
            tries += 1
            continue
        answer = random.choice(candidates)
        question = re.sub(r"\b" + re.escape(answer) + r"\b", "____", s, flags=re.IGNORECASE)
        quizzes.append({"question": question, "answer": answer})
        tries += 1
    return quizzes


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_progress(scores):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(scores, f)


def extractive_summary_from_text(text: str, length: str = "short") -> str:
    sentences = [s.strip() for s in split_sentences(text) if s.strip()]
    if not sentences:
        return ""

    word_counts = Counter(re.findall(r"[A-Za-z']+", text.lower()))
    for w in list(STOPWORDS):
        word_counts.pop(w, None)

    nmap = {"short": 1, "medium": 2, "detailed": 3}
    target_sentences = nmap.get(length, 2)
    target_sentences = min(target_sentences, len(sentences))

    scored = []
    for idx, sentence in enumerate(sentences):
        words = re.findall(r"[A-Za-z']+", sentence.lower())
        if not words:
            continue
        score = 0
        for word in words:
            if word in STOPWORDS:
                continue
            score += word_counts.get(word, 0)
        score += 0.1 * len(words)
        scored.append((idx, score, sentence))

    if not scored:
        return " ".join(sentences[:target_sentences])

    scored.sort(key=lambda x: x[1], reverse=True)
    selected_indices = {idx for idx, _, _ in scored[:target_sentences]}
    ordered = [sentence for idx, _, sentence in scored if idx in selected_indices]
    ordered.sort(key=lambda s: sentences.index(s))
    return " ".join(ordered)


def summarize_text(text: str, length: str = "short", return_debug: bool = False):
    clean_text = " ".join(split_sentences(text)).strip()
    if not clean_text:
        return ("", None) if return_debug else ""

    error_message = None

    try:
        from transformers import pipeline

        summary_map = {
            "short": {"min_length": 20, "max_length": 80},
            "medium": {"min_length": 35, "max_length": 120},
            "detailed": {"min_length": 50, "max_length": 180},
        }
        summary_cfg = summary_map.get(length, summary_map["medium"])

        summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
            device=-1,
        )

        task_prefix = "summarize: "
        result = summarizer(
            task_prefix + clean_text,
            min_length=summary_cfg["min_length"],
            max_length=summary_cfg["max_length"],
            do_sample=False,
            truncation=True,
        )
        if result and result[0].get("generated_text"):
            summary = result[0]["generated_text"].strip()
            if summary and len(summary.split()) >= 8:
                return (summary, None) if return_debug else summary
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

    fallback_summary = extractive_summary_from_text(clean_text, length)
    if return_debug:
        return (fallback_summary, error_message)
    return fallback_summary


def main():
    st.set_page_config(page_title="AI Study Mate", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f5f5;
        }
        .block-container {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #f0f0f0;
            border-right: 1px solid #d9d9d9;
        }
        .sidebar-section {
            margin-top: 1.2rem;
            padding-top: 0.5rem;
            border-top: 1px solid #d0d0d0;
        }
        .sidebar-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1f1f1f;
            margin-top: 1rem;
        }
        .sidebar-text {
            font-size: 1rem;
            color: #2b2b2b;
            line-height: 1.7;
        }
        .sidebar-item {
            margin: 0.15rem 0;
            color: #2b2b2b;
            font-size: 0.95rem;
        }
        .summary-box {
            background: #dff3ea;
            border: 1px solid #d0e8dd;
            border-radius: 10px;
            padding: 1.15rem 1.2rem;
            margin: 0.5rem 0 1.3rem 0;
            font-size: 1.02rem;
            color: #1f2d2a;
            text-align: center;
            font-weight: 500;
            min-height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .keyword-card {
            background: #fff;
            border: 1px solid #dfe5ea;
            border-radius: 10px;
            padding: 0.9rem 0.6rem 0.8rem 0.6rem;
            text-align: center;
            min-height: 132px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin: 0.2rem 0.2rem 0.9rem 0.2rem;
            box-shadow: 0 0 0 0 rgba(0,0,0,0);
        }
        .keyword-label {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1f1f1f;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .keyword-count {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1c1c1c;
            margin-top: 0.35rem;
            line-height: 1.1;
        }
        .info-dot {
            width: 16px;
            height: 16px;
            border: 1px solid #7c7c7c;
            border-radius: 50%;
            font-size: 0.7rem;
            color: #666;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
            margin-left: 0.2rem;
        }
        .notice-box {
            background: #dfeaf8;
            border: 1px solid #cfe0f8;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            color: #1d2f57;
            font-size: 1.02rem;
            text-align: center;
            font-weight: 500;
        }
        .stButton > button {
            background: #ef5c5c;
            color: white;
            border: none;
            border-radius: 8px;
            width: 100%;
            font-weight: 700;
            padding: 0.8rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='sidebar-title'>Project information</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-text'>AI model: <span style='color:#2a8b5c; font-weight:600;'>google/flan-t5-small</span></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-text'>Cost: No paid API is required</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-text'>Best input: English study passages</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-text'>Recommended length: 100–600 words</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-title'>How it works</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-item'>1. FLAN-T5 creates a summary.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-item'>2. Python counts important words.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-item'>3. Important source sentences become quiz questions.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-item'>4. Session State keeps the score after the app runs.</div>", unsafe_allow_html=True)

    st.title("AI Study Mate")
    st.caption("Streamlit 기반 영어 지문 학습 도우미 — 요약, 단어 추출, 퀴즈, 진행 관리")

    tabs = st.tabs(["Study", "Quiz", "Progress", "About"])

    # ensure session state
    if "current_quizzes" not in st.session_state:
        st.session_state.current_quizzes = []
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "last_score" not in st.session_state:
        st.session_state.last_score = None

    with tabs[0]:  # Study
        st.subheader("Study")
        passage = st.text_area("Enter English passage", height=250)
        summary_length = st.selectbox("Summary length", ["short", "medium", "detailed"])
        num_quizzes = st.slider("Number of quizzes to generate", min_value=1, max_value=10, value=3)

        if st.button("Create study set"):
            if count_words(passage) < 40:
                st.warning("Passage must be at least 40 words. Please enter a longer passage.")
            else:
                with st.spinner("Generating summary and keywords..."):
                    summary, model_error = summarize_text(passage, summary_length, return_debug=True)
                    keyword_counts = extract_keywords(passage, max_keywords=8)
                    quizzes = create_quizzes_from_text(passage, num_quizzes)
                    st.session_state.current_quizzes = quizzes
                    st.session_state.quiz_answers = {}
                    st.session_state.passage = passage
                    st.success("Study set created. Switch to the Quiz tab to take quizzes.")

                    st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.8rem;'>2. AI summary</h3>", unsafe_allow_html=True)
                    if model_error:
                        st.warning(f"AI 모델 로딩에 실패해서 문장 점수 계산 방식으로 요약했습니다. 원인: {model_error}")
                    st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

                    st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.8rem;'>3. Key vocabulary</h3>", unsafe_allow_html=True)
                    if keyword_counts:
                        cols = st.columns(4)
                        for i, (word, count) in enumerate(keyword_counts):
                            with cols[i % 4]:
                                st.markdown(
                                    f"""
                                    <div class='keyword-card'>
                                        <div class='keyword-label'>{word} <span class='info-dot'>i</span></div>
                                        <div class='keyword-count'>{count}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.write("No keywords found.")

                    st.markdown("<div class='notice-box'>Your quiz is ready. Open the Quiz tab.</div>", unsafe_allow_html=True)

    with tabs[1]:  # Quiz
        st.subheader("Quiz")
        if not st.session_state.current_quizzes:
            st.info("No quiz available. Create a study set in the Study tab first.")
        else:
            quizzes = st.session_state.current_quizzes
            for i, q in enumerate(quizzes):
                st.write(f"Q{i+1}. " + q["question"])
                key = f"answer_{i}"
                default = st.session_state.quiz_answers.get(key, "")
                st.session_state.quiz_answers[key] = st.text_input("Your answer", value=default, key=key)

            if st.button("Check answers"):
                correct = 0
                review = []
                for i, q in enumerate(quizzes):
                    ans = st.session_state.quiz_answers.get(f"answer_{i}", "").strip()
                    if ans.lower() == q["answer"].lower():
                        correct += 1
                        review.append({"question": q["question"], "given": ans, "answer": q["answer"], "correct": True})
                    else:
                        review.append({"question": q["question"], "given": ans, "answer": q["answer"], "correct": False})
                score = int(round(100 * correct / len(quizzes)))
                st.session_state.last_score = score
                progress = load_progress()
                progress.append(score)
                save_progress(progress)

                st.success(f"Your score: {score}% ({correct}/{len(quizzes)})")
                st.markdown("**Answer review**")
                for r in review:
                    if r["correct"]:
                        st.write(f"✅ {r['question']} — Your answer: {r['given']}")
                    else:
                        st.write(f"❌ {r['question']} — Your answer: {r['given']} — Correct: {r['answer']}")

            if st.session_state.last_score is not None:
                if st.button("Create another quiz"):
                    # regenerate quizzes from last passage
                    passage = st.session_state.get("passage", "")
                    st.session_state.current_quizzes = create_quizzes_from_text(passage, len(st.session_state.current_quizzes))
                    st.session_state.quiz_answers = {}
                    st.session_state.last_score = None
                    if hasattr(st, "rerun"):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()

    with tabs[2]:  # Progress
        st.subheader("Progress")
        progress = load_progress()
        completed = len(progress)
        recent = progress[-1] if progress else None
        best = max(progress) if progress else None
        st.write(f"Completed quizzes: {completed}")
        st.write(f"Most recent score: {recent if recent is not None else 'N/A'}")
        st.write(f"Best score: {best if best is not None else 'N/A'}")
        if progress:
            import pandas as pd

            df = pd.DataFrame({"score": progress})
            df.index += 1
            st.line_chart(df)

        if st.button("Clear progress"):
            save_progress([])
            st.success("Progress cleared.")

    with tabs[3]:  # About
        st.subheader("About")
        st.markdown("AI Study Mate: 영어 지문을 요약하고, 주요 단어를 추출하고, 빈칸 채우기 퀴즈로 학습 진행을 돕는 도구입니다.")


if __name__ == "__main__":
    main()
