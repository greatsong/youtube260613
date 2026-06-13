import re
import os
import emoji
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="YouTube 댓글 심층 분석기",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #666;
    margin-bottom: 1.2rem;
}
.metric-card {
    padding: 1rem;
    border-radius: 1rem;
    background: #f7f7f9;
    border: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 유틸 함수
# =========================
def get_api_key():
    try:
        return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        return os.getenv("YOUTUBE_API_KEY")


def extract_video_id(url_or_id: str):
    text = url_or_id.strip()

    # 이미 video id만 입력한 경우
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", text):
        return text

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def clean_html(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return re.sub(r"\s+", " ", text).strip()


def contains_korean(text):
    return bool(re.search(r"[가-힣]", str(text)))


def count_emojis(text):
    return sum(1 for ch in str(text) if ch in emoji.EMOJI_DATA)


@st.cache_resource
def get_kiwi():
    return Kiwi()


def tokenize_korean(text):
    kiwi = get_kiwi()
    tokens = []

    for token in kiwi.tokenize(str(text)):
        # N: 명사, V: 동사/형용사, SL: 외국어
        if token.tag.startswith("N") or token.tag.startswith("V") or token.tag == "SL":
            word = token.form.strip()
            if len(word) >= 2:
                tokens.append(word)

    return tokens


def get_korean_font_path():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "NanumGothic.ttf"
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


# 간단 감성 사전
POSITIVE_WORDS = set("""
좋다 좋아 최고 만족 추천 감사 예쁘다 훌륭 대박 재밌 감동 유익 도움 편하다
깔끔 친절 완벽 기대 응원 사랑 멋지다 공감 신기 정확 빠르다 성공
""".split())

NEGATIVE_WORDS = set("""
싫다 별로 최악 실망 문제 오류 불편 짜증 화남 답답 무섭다 아쉽다
비싸다 느리다 부족 실패 거짓 과장 논란 심각 불만 노잼 걱정 위험
""".split())


def sentiment_score(text):
    tokens = tokenize_korean(text)
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS or any(p in t for p in POSITIVE_WORDS))
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS or any(n in t for n in NEGATIVE_WORDS))

    if pos > neg:
        return "긍정"
    elif neg > pos:
        return "부정"
    else:
        return "중립"


def fetch_replies(youtube, parent_id, max_replies=20):
    replies = []
    next_page_token = None

    while len(replies) < max_replies:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=min(100, max_replies - len(replies)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            snip = item["snippet"]
            replies.append({
                "comment_id": item["id"],
                "parent_id": parent_id,
                "author": snip.get("authorDisplayName", ""),
                "text": clean_html(snip.get("textDisplay", "")),
                "like_count": snip.get("likeCount", 0),
                "published_at": snip.get("publishedAt", ""),
                "updated_at": snip.get("updatedAt", ""),
                "is_reply": True
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


@st.cache_data(show_spinner=False)
def fetch_comments(video_id, api_key, max_comments=300, order="relevance", include_replies=False):
    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            order=order,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comment_id = item["snippet"]["topLevelComment"]["id"]
            total_reply_count = item["snippet"].get("totalReplyCount", 0)

            comments.append({
                "comment_id": comment_id,
                "parent_id": "",
                "author": top.get("authorDisplayName", ""),
                "text": clean_html(top.get("textDisplay", "")),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt", ""),
                "updated_at": top.get("updatedAt", ""),
                "reply_count": total_reply_count,
                "is_reply": False
            })

            if include_replies and total_reply_count > 0 and len(comments) < max_comments:
                remain = max_comments - len(comments)
                replies = fetch_replies(youtube, comment_id, max_replies=min(30, remain))
                for r in replies:
                    r["reply_count"] = 0
                comments.extend(replies)

            if len(comments) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments)


def build_wordcloud(texts):
    all_text = " ".join(texts)
    tokens = tokenize_korean(all_text)

    stopwords = set("""
    그냥 진짜 너무 정말 완전 계속 이제 여기 저기 그리고 그래서 하지만
    영상 댓글 사람 생각 내용 우리 저는 제가 이거 그거 수 있다 없다
    합니다 있어요 없어요 되는 같은 대한 보다 이런 저런
    """.split())

    words = [w for w in tokens if w not in stopwords and len(w) >= 2]
    counter = Counter(words)

    font_path = get_korean_font_path()
    if font_path is None:
        raise FileNotFoundError("한글 폰트를 찾지 못했습니다. packages.txt에 fonts-nanum을 추가하세요.")

    wc = WordCloud(
        font_path=font_path,
        width=1400,
        height=800,
        background_color="white",
        max_words=150,
        prefer_horizontal=0.9,
        collocations=False
    ).generate_from_frequencies(counter)

    return wc, counter


def extract_topics(df, n_topics=5):
    texts = df["text"].dropna().astype(str).tolist()

    if len(texts) < n_topics:
        return pd.DataFrame()

    def tokenizer_for_tfidf(text):
        return tokenize_korean(text)

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer_for_tfidf,
        token_pattern=None,
        min_df=2,
        max_df=0.85
    )

    X = vectorizer.fit_transform(texts)

    if X.shape[1] < n_topics:
        return pd.DataFrame()

    k = min(n_topics, len(texts), X.shape[1])
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X)

    terms = np.array(vectorizer.get_feature_names_out())
    rows = []

    for topic_num in range(k):
        center = model.cluster_centers_[topic_num]
        top_indices = center.argsort()[::-1][:8]
        keywords = ", ".join(terms[top_indices])

        sample_indices = np.where(labels == topic_num)[0][:3]
        samples = [texts[i][:80] for i in sample_indices]

        rows.append({
            "토픽": f"토픽 {topic_num + 1}",
            "대표 키워드": keywords,
            "댓글 수": int((labels == topic_num).sum()),
            "대표 댓글 예시": " / ".join(samples)
        })

    return pd.DataFrame(rows)


def make_download_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# 화면 구성
# =========================
st.markdown('<div class="main-title">💬 YouTube 댓글 심층 분석기</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">유튜브 링크를 입력하면 댓글을 수집하고, 감성·토픽·키워드·한글 워드클라우드를 분석합니다.</div>',
    unsafe_allow_html=True
)

api_key = get_api_key()

with st.sidebar:
    st.header("⚙️ 분석 설정")

    video_input = st.text_input(
        "유튜브 링크 또는 영상 ID",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    max_comments = st.slider(
        "최대 수집 댓글 수",
        min_value=50,
        max_value=2000,
        value=300,
        step=50
    )

    order = st.selectbox(
        "댓글 수집 순서",
        ["relevance", "time"],
        format_func=lambda x: "관련도순" if x == "relevance" else "최신순"
    )

    include_replies = st.checkbox(
        "답글도 일부 포함하기",
        value=False,
        help="답글까지 수집하면 더 깊게 분석할 수 있지만 API 사용량과 시간이 늘어납니다."
    )

    analyze_button = st.button("🚀 댓글 분석 시작", use_container_width=True)

    st.divider()
    st.caption("API 키는 Streamlit Cloud Secrets에 YOUTUBE_API_KEY 이름으로 저장하세요.")


if not api_key:
    st.error("""
    YouTube API 키가 설정되어 있지 않습니다.

    Streamlit Cloud의 App settings → Secrets에 아래처럼 입력하세요.

    ```toml
    YOUTUBE_API_KEY = "여기에_내_API_KEY"
    ```
    """)
    st.stop()


if analyze_button:
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error("유효한 유튜브 링크 또는 11자리 영상 ID를 입력해주세요.")
        st.stop()

    try:
        with st.spinner("댓글을 수집하는 중입니다..."):
            df = fetch_comments(
                video_id=video_id,
                api_key=api_key,
                max_comments=max_comments,
                order=order,
                include_replies=include_replies
            )

        if df.empty:
            st.warning("수집된 댓글이 없습니다. 댓글이 비활성화되었거나 접근할 수 없는 영상일 수 있습니다.")
            st.stop()

        # 파생 변수
        df["text_length"] = df["text"].astype(str).str.len()
        df["has_korean"] = df["text"].apply(contains_korean)
        df["emoji_count"] = df["text"].apply(count_emojis)
        df["question"] = df["text"].astype(str).str.contains(r"\?|？|왜|어떻게|무엇|뭐|언제|어디|누구")
        df["sentiment"] = df["text"].apply(sentiment_score)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        st.success(f"댓글 {len(df):,}개를 수집했습니다.")

        # =========================
        # 핵심 지표
        # =========================
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("수집 댓글", f"{len(df):,}개")
        c2.metric("고유 작성자", f"{df['author'].nunique():,}명")
        c3.metric("평균 좋아요", f"{df['like_count'].mean():.1f}")
        c4.metric("한글 댓글 비율", f"{df['has_korean'].mean() * 100:.1f}%")
        c5.metric("질문형 댓글", f"{df['question'].sum():,}개")

        st.divider()

        # =========================
        # 감성 분석
        # =========================
        st.subheader("1. 댓글 감성 분석")

        sentiment_count = (
            df["sentiment"]
            .value_counts()
            .reindex(["긍정", "중립", "부정"])
            .fillna(0)
            .reset_index()
        )
        sentiment_count.columns = ["감성", "댓글 수"]

        fig_sentiment = px.pie(
            sentiment_count,
            names="감성",
            values="댓글 수",
            hole=0.45,
            title="댓글 감성 비율"
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

        with st.expander("감성별 대표 댓글 보기"):
            for senti in ["긍정", "중립", "부정"]:
                st.markdown(f"#### {senti}")
                sample = df[df["sentiment"] == senti].sort_values("like_count", ascending=False).head(5)
                if sample.empty:
                    st.write("해당 댓글이 없습니다.")
                else:
                    for _, row in sample.iterrows():
                        st.markdown(f"- 👍 {row['like_count']} | {row['text']}")

        st.divider()

        # =========================
        # 한글 워드클라우드
        # =========================
        st.subheader("2. 한글 워드클라우드")

        korean_df = df[df["has_korean"]].copy()

        if korean_df.empty:
            st.warning("한글 댓글이 없어 한글 워드클라우드를 만들 수 없습니다.")
        else:
            wc, word_counter = build_wordcloud(korean_df["text"].tolist())

            fig, ax = plt.subplots(figsize=(14, 8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            top_words = pd.DataFrame(
                word_counter.most_common(30),
                columns=["단어", "빈도"]
            )

            fig_words = px.bar(
                top_words,
                x="빈도",
                y="단어",
                orientation="h",
                title="상위 키워드 30개"
            )
            fig_words.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_words, use_container_width=True)

        st.divider()

        # =========================
        # 토픽 분석
        # =========================
        st.subheader("3. 댓글 토픽 분석")

        topic_df = extract_topics(df, n_topics=5)

        if topic_df.empty:
            st.warning("토픽 분석을 수행하기에는 댓글 수 또는 반복 키워드가 부족합니다.")
        else:
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

            fig_topic = px.bar(
                topic_df,
                x="토픽",
                y="댓글 수",
                text="댓글 수",
                title="토픽별 댓글 수"
            )
            st.plotly_chart(fig_topic, use_container_width=True)

        st.divider()

        # =========================
        # 참여도 분석
        # =========================
        st.subheader("4. 댓글 참여도 분석")

        col1, col2 = st.columns(2)

        with col1:
            top_likes = df.sort_values("like_count", ascending=False).head(10)
            fig_likes = px.bar(
                top_likes,
                x="like_count",
                y=top_likes["text"].str.slice(0, 40),
                orientation="h",
                title="좋아요가 많은 댓글 TOP 10",
                labels={"like_count": "좋아요 수", "y": "댓글"}
            )
            fig_likes.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_likes, use_container_width=True)

        with col2:
            fig_len = px.histogram(
                df,
                x="text_length",
                nbins=30,
                title="댓글 길이 분포",
                labels={"text_length": "댓글 글자 수"}
            )
            st.plotly_chart(fig_len, use_container_width=True)

        if df["published_at"].notna().sum() > 0:
            st.subheader("5. 시간 흐름 분석")

            time_df = (
                df.dropna(subset=["published_at"])
                .set_index("published_at")
                .resample("D")
                .size()
                .reset_index(name="댓글 수")
            )

            fig_time = px.line(
                time_df,
                x="published_at",
                y="댓글 수",
                markers=True,
                title="날짜별 댓글 수"
            )
            st.plotly_chart(fig_time, use_container_width=True)

        st.divider()

        # =========================
        # 원본 데이터
        # =========================
        st.subheader("6. 원본 댓글 데이터")

        st.dataframe(
            df[[
                "published_at", "author", "text", "like_count",
                "reply_count", "sentiment", "has_korean",
                "emoji_count", "is_reply"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 댓글 분석 결과 CSV 다운로드",
            data=make_download_csv(df),
            file_name=f"youtube_comments_{video_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

    except HttpError as e:
        st.error("YouTube API 오류가 발생했습니다.")
        st.code(str(e), language="text")
        st.info("""
        자주 발생하는 원인:
        - API 키가 잘못됨
        - YouTube Data API v3가 활성화되지 않음
        - 댓글이 비활성화된 영상
        - API 할당량 초과
        - 비공개 또는 접근 제한 영상
        """)

    except Exception as e:
        st.error("분석 중 오류가 발생했습니다.")
        st.code(str(e), language="text")

else:
    st.info("왼쪽 사이드바에 유튜브 링크를 입력하고 분석을 시작하세요.")

    st.markdown("""
    ### 이 앱에서 가능한 분석

    - 유튜브 댓글 자동 수집
    - 한글 댓글 비율 확인
    - 긍정 / 중립 / 부정 감성 분석
    - 한글 워드클라우드 생성
    - 상위 키워드 추출
    - 댓글 토픽 군집화
    - 좋아요 많은 댓글 확인
    - 댓글 길이, 질문형 댓글, 이모지 사용 분석
    - CSV 다운로드

    ### 수업 활용 아이디어

    이 앱은 단순히 “댓글을 많이 모으는 도구”가 아니라,  
    하나의 영상에 대한 시청자 반응을 데이터로 읽는 활동에 적합합니다.

    예를 들어 학생들은 다음 질문을 탐구할 수 있습니다.

    - 사람들이 영상의 어떤 부분에 가장 많이 반응했는가?
    - 긍정 댓글과 부정 댓글은 어떤 단어를 자주 쓰는가?
    - 좋아요가 많은 댓글은 일반 댓글과 무엇이 다른가?
    - 댓글 데이터만 보고 영상의 주제를 추론할 수 있는가?
    """)
