
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FeedbackLens AI",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background-color: #111c30;
}
.metric-card {
    background: #17243a;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263750;
}
.insight-card {
    background: #17243a;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 10px;
    border-left: 4px solid #38bdf8;
}
h1, h2, h3 {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SAMPLE DATA ----------------
@st.cache_data
def load_data():
    data = {
        "review": [
            "I love the new design, very easy to use",
            "The application crashes every time I open it",
            "Customer support was very helpful",
            "The payment process is extremely slow",
            "Amazing experience, I recommend this app",
            "The latest update made everything worse",
            "Great features and beautiful interface",
            "I cannot log in to my account",
            "Very satisfied with the service",
            "Too many bugs after the update",
            "The app is simple and convenient",
            "Delivery was late and support did not help",
            "Excellent quality and smooth experience",
            "The search feature is not working",
            "I am happy with the overall experience",
            "The app takes too long to load",
            "Fantastic customer service",
            "The new feature is confusing and difficult",
            "Really useful application",
            "I want a refund because the app does not work"
        ],
        "category": [
            "Design", "Performance", "Support", "Payments",
            "Experience", "Performance", "Design", "Login",
            "Experience", "Performance", "Usability", "Delivery",
            "Experience", "Search", "Experience", "Performance",
            "Support", "Usability", "Features", "Payments"
        ],
        "rating": [
            5, 1, 5, 2, 5, 1, 5, 1, 4, 1,
            4, 1, 5, 2, 4, 2, 5, 2, 5, 1
        ]
    }
    return pd.DataFrame(data)


# ---------------- SENTIMENT MODEL ----------------
training_texts = [
    # Positive examples (10)
    "I love this product",
    "Amazing experience",
    "Excellent service",
    "Very happy and satisfied",
    "Great application",
    "Works perfectly",
    "Wonderful features",
    "I recommend this app",
    "Very useful",
    "Beautiful design",

    # Negative examples (10)
    "I hate this product",
    "Terrible experience",
    "Worst service ever",
    "Very disappointed",
    "The app crashes",
    "Full of bugs",
    "Not working properly",
    "Very slow application",
    "I want a refund",
    "Completely useless",

    # Neutral examples (7)
    "The product is okay",
    "It is average",
    "Nothing special",
    "The service is acceptable",
    "It works normally",
    "The experience was fine",
    "Neither good nor bad"
]

training_labels = (
    ["Positive"] * 10 +
    ["Negative"] * 10 +
    ["Neutral"] * 7
)

# Safety check: both lists must have equal lengths
assert len(training_texts) == len(training_labels), (
    "Training data and labels must have equal lengths."
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(training_texts, training_labels)


def predict_sentiment(text):
    return model.predict([str(text)])[0]


# ---------------- HEADER ----------------
st.title("🔍 FeedbackLens AI")
st.caption("CUSTOMER FEEDBACK INTELLIGENCE PLATFORM")
st.markdown(
    "Transform customer feedback into actionable product decisions."
)


# ---------------- SIDEBAR ----------------
st.sidebar.title("FeedbackLens")
st.sidebar.caption("AI-Powered Product Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Sentiment Analysis",
        "Customer Feedback",
        "Product Insights",
        "AI Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Demo project using sample customer feedback.")


# ---------------- DATA UPLOAD ----------------
default_df = load_data()

uploaded_file = st.sidebar.file_uploader(
    "Upload feedback CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if "review" not in uploaded_df.columns:
            st.error("Your CSV must contain a column named 'review'.")
            df = default_df.copy()
        else:
            df = uploaded_df.copy()

            if "category" not in df.columns:
                df["category"] = "General"

            if "rating" not in df.columns:
                df["rating"] = 3

    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        df = default_df.copy()
else:
    df = default_df.copy()


# ---------------- DATA PROCESSING ----------------
df["review"] = df["review"].fillna("").astype(str)
df["category"] = df["category"].fillna("General").astype(str)
df["rating"] = pd.to_numeric(
    df["rating"], errors="coerce"
).fillna(3)

df["sentiment"] = df["review"].apply(predict_sentiment)


# ---------------- FILTERS ----------------
st.sidebar.markdown("### Filters")

categories = ["All"] + sorted(
    df["category"].unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Feedback category",
    categories
)

if selected_category != "All":
    filtered_df = df[
        df["category"] == selected_category
    ].copy()
else:
    filtered_df = df.copy()


# ---------------- EXECUTIVE OVERVIEW ----------------
if page == "Executive Overview":

    st.subheader("Executive Overview")

    total = len(filtered_df)

    positive = (
        filtered_df["sentiment"] == "Positive"
    ).sum()

    negative = (
        filtered_df["sentiment"] == "Negative"
    ).sum()

    neutral = (
        filtered_df["sentiment"] == "Neutral"
    ).sum()

    satisfaction = (
        round(positive / total * 100, 1)
        if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", total)
    c2.metric("Positive Reviews", int(positive))
    c3.metric("Negative Reviews", int(negative))
    c4.metric("Positive Share", f"{satisfaction}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("### Sentiment Distribution")

        sentiment_counts = (
            filtered_df["sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Reviews",
            hole=0.55
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.markdown("### Feedback by Category")

        category_counts = (
            filtered_df.groupby("category")
            .size()
            .reset_index(name="Reviews")
            .sort_values("Reviews", ascending=False)
        )

        fig = px.bar(
            category_counts,
            x="category",
            y="Reviews",
            color="Reviews"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Latest Customer Feedback")

    st.dataframe(
        filtered_df.head(10),
        use_container_width=True
    )


# ---------------- SENTIMENT ANALYSIS ----------------
elif page == "Sentiment Analysis":

    st.subheader("AI Sentiment Analysis")

    st.write(
        "Classify customer feedback using a machine-learning text model."
    )

    user_text = st.text_area(
        "Enter customer feedback",
        placeholder="Example: The application is easy to use and very helpful..."
    )

    if st.button("Analyse Feedback"):
        if user_text.strip():
            result = predict_sentiment(user_text)
            st.success(f"Predicted sentiment: {result}")
        else:
            st.warning("Please enter some feedback first.")

    st.markdown("---")
    st.markdown("### Sentiment Breakdown")

    sentiment_summary = (
        filtered_df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_summary.columns = [
        "Sentiment",
        "Count"
    ]

    st.dataframe(
        sentiment_summary,
        use_container_width=True
    )


# ---------------- CUSTOMER FEEDBACK ----------------
elif page == "Customer Feedback":

    st.subheader("Customer Feedback Explorer")

    search = st.text_input(
        "Search customer feedback"
    )

    display_df = filtered_df.copy()

    if search:
        display_df = display_df[
            display_df["review"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    sentiment_filter = st.multiselect(
        "Filter by sentiment",
        ["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )

    display_df = display_df[
        display_df["sentiment"].isin(sentiment_filter)
    ]

    st.write(f"Showing {len(display_df)} reviews")

    st.dataframe(
        display_df,
        use_container_width=True
    )

    csv = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered feedback",
        csv,
        "feedbacklens_results.csv",
        "text/csv"
    )


# ---------------- PRODUCT INSIGHTS ----------------
elif page == "Product Insights":

    st.subheader("Product Insights")

    negative_df = filtered_df[
        filtered_df["sentiment"] == "Negative"
    ]

    st.markdown("### Categories with negative feedback")

    if len(negative_df) > 0:

        issue_counts = (
            negative_df.groupby("category")
            .size()
            .reset_index(name="Negative Reviews")
            .sort_values("Negative Reviews", ascending=False)
        )

        fig = px.bar(
            issue_counts,
            x="category",
            y="Negative Reviews",
            color="Negative Reviews"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("### Reviews needing attention")

        st.dataframe(
            negative_df[
                ["review", "category", "sentiment"]
            ],
            use_container_width=True
        )

    else:
        st.info(
            "No negative feedback in the current selection."
        )

    st.markdown("### Average customer rating")

    avg_rating = (
        filtered_df["rating"].mean()
        if len(filtered_df) else 0
    )

    st.metric(
        "Average Rating",
        f"{avg_rating:.2f}/5"
    )


# ---------------- AI RECOMMENDATIONS ----------------
elif page == "AI Recommendations":

    st.subheader("AI Product Recommendations")

    negative_df = filtered_df[
        filtered_df["sentiment"] == "Negative"
    ]

    if len(negative_df) > 0:

        top_issue = (
            negative_df["category"]
            .value_counts()
            .idxmax()
        )

        issue_total = int(
            (negative_df["category"] == top_issue).sum()
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <b>Priority 1: Investigate {top_issue}</b><br>
                {issue_total} negative reviews are associated
                with this category. Review customer complaints
                and identify the root cause.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="insight-card">
                <b>Priority 2: Improve customer experience</b><br>
                Review recurring negative feedback and simplify
                the affected user journeys.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="insight-card">
                <b>Priority 3: Measure improvements</b><br>
                Track sentiment and customer ratings after
                releasing product changes.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.success(
            "No negative feedback detected. Continue monitoring customer satisfaction."
        )

    st.markdown("---")

    st.caption(
        "Recommendations are rule-based insights generated from "
        "the feedback dataset. Sentiment labels come from a "
        "model trained on a small demonstration dataset."
    )


# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "FeedbackLens AI | Portfolio Project | Built with Python and Streamlit"
)
