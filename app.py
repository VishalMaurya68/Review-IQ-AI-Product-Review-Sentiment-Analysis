import streamlit as st
import pandas as pd
import joblib
import sqlite3
import json
import plotly.express as px
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ReviewIQ AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- DARK THEME ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080816, #11112b);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #101025;
}
h1, h2, h3 {
    color: #c4b5fd !important;
}
div[data-testid="stMetric"] {
    background: #191936;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #393966;
}
.stButton button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- DATABASE ----------------

def create_database():
    conn = sqlite3.connect("reviews.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT,
            review TEXT,
            sentiment TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_review(product, review, sentiment, confidence):
    conn = sqlite3.connect("reviews.db")
    conn.execute("""
        INSERT INTO review_history
        (product, review, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        product,
        review,
        sentiment,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect("reviews.db")
    df = pd.read_sql_query(
        "SELECT * FROM review_history ORDER BY id DESC",
        conn
    )
    conn.close()
    return df

create_database()

# ---------------- ASPECT ANALYSIS ----------------

def analyze_aspects(review):

    review_lower = review.lower()

    aspect_keywords = {
        "Battery": ["battery", "backup", "charging", "charge"],
        "Price": ["price", "cost", "money", "expensive", "cheap"],
        "Quality": ["quality", "build", "material", "durable"],
        "Performance": ["performance", "speed", "fast", "slow", "processor"],
        "Camera": ["camera", "photo", "picture"],
        "Display": ["display", "screen", "brightness"]
    }

    positive_words = [
        "excellent", "amazing", "good", "great",
        "fast", "beautiful", "happy", "smooth",
        "perfect", "best", "impressive"
    ]

    negative_words = [
        "poor", "bad", "slow", "terrible",
        "disappointing", "worst", "cheap",
        "problem", "drains", "not worth"
    ]

    results = []

    for aspect, keywords in aspect_keywords.items():

        if any(word in review_lower for word in keywords):

            positive_count = sum(
                word in review_lower for word in positive_words
            )

            negative_count = sum(
                word in review_lower for word in negative_words
            )

            if positive_count > negative_count:
                sentiment = "Positive"

            elif negative_count > positive_count:
                sentiment = "Negative"

            else:
                sentiment = "Neutral"

            results.append({
                "Aspect": aspect,
                "Sentiment": sentiment
            })

    if not results:
        results.append({
            "Aspect": "General Review",
            "Sentiment": "Not Detected"
        })

    return results

# ---------------- PREDICTION ----------------

def predict_review(review):
    transformed = vectorizer.transform([review])
    prediction = model.predict(transformed)[0]

    probabilities = model.predict_proba(transformed)[0]
    confidence = max(probabilities) * 100

    return prediction, confidence

# ---------------- HEADER ----------------

st.markdown("# 🤖 ReviewIQ AI")
st.markdown(
    "### Intelligent Product Review Sentiment Analysis"
)
st.caption("Offline Machine Learning | TF-IDF | Logistic Regression")

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚀 ReviewIQ AI")
st.sidebar.caption("Navigation Panel")

page = st.sidebar.radio(
    "Choose a Section",
    [
        "Dashboard",
        "Analyze Review",
        "Batch Analysis",
        "Review History",
        "Model Performance",
        "Product Comparison"
    ]
)

st.sidebar.divider()
st.sidebar.success("Offline ML Model Active")
st.sidebar.caption("Version 2.0 | College Project")

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("📊 Project Dashboard")

    history = get_history()

    total = len(history)

    if total > 0:
        positive = (history["sentiment"] == "Positive").sum()
        negative = (history["sentiment"] == "Negative").sum()
        neutral = (history["sentiment"] == "Neutral").sum()
    else:
        positive = negative = neutral = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", total)
    c2.metric("😊 Positive", int(positive))
    c3.metric("😞 Negative", int(negative))
    c4.metric("😐 Neutral", int(neutral))

    st.divider()

    if total > 0:
        left, right = st.columns(2)

        with left:
            st.subheader("Sentiment Distribution")

            counts = history["sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]

            fig = px.pie(
                counts,
                names="Sentiment",
                values="Count",
                hole=0.55,
                title="Customer Sentiment"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Product-wise Reviews")

            product_counts = (
                history.groupby(["product", "sentiment"])
                .size()
                .reset_index(name="Count")
            )

            fig2 = px.bar(
                product_counts,
                x="product",
                y="Count",
                color="sentiment",
                barmode="group",
                title="Product Sentiment Comparison"
            )
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recent Reviews")
        st.dataframe(
            history.head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No reviews analyzed yet. Go to Analyze Review to begin.")

# ---------------- ANALYZE REVIEW ----------------

elif page == "Analyze Review":

    st.subheader("🔍 Analyze a Customer Review")

    product = st.text_input(
        "Product Name",
        placeholder="Example: Samsung Galaxy Book4"
    )

    review = st.text_area(
        "Enter Customer Review",
        placeholder="Write your product experience here...",
        height=150
    )

    if st.button("✨ Analyze Sentiment", use_container_width=True):

        if not product.strip() or not review.strip():
            st.warning("Please enter both product name and review.")

        else:
            sentiment, confidence = predict_review(review)

            aspects = analyze_aspects(review)

            save_review(
                product,
                review,
                sentiment,
                confidence
            )

            st.divider()
            st.subheader("Analysis Result")

            if sentiment == "Positive":
                st.success("😊 Positive Sentiment")
            elif sentiment == "Negative":
                st.error("😞 Negative Sentiment")
            else:
                st.info("😐 Neutral Sentiment")

            col1, col2 = st.columns(2)

            col1.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            col2.metric(
                "Detected Aspects",
                len(aspects)
            )

            st.progress(int(confidence))

            st.write("**Product Aspects Detected:**")
            aspect_df = pd.DataFrame(aspects)

            st.write("**Aspect-wise Sentiment Analysis:**")

            st.dataframe(
                aspect_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee of prediction correctness."
            )

            st.success("Review saved in database!")

# ---------------- BATCH ANALYSIS ----------------

elif page == "Batch Analysis":

    st.subheader("📁 Upload Multiple Reviews")

    st.write(
        "Upload a CSV file containing a column named 'review'. "
        "An optional 'product' column is also supported."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )
    if uploaded_file is not None:

        try:
            batch = pd.read_csv(uploaded_file)

            if "review" not in batch.columns:
                st.error("CSV must contain a 'review' column.")

            else:
                if "product" not in batch.columns:
                    batch["product"] = "Unknown Product"

                batch = batch.dropna(subset=["review"]).copy()

                if st.button("Analyze All Reviews"):

                    results = []

                    for _, row in batch.iterrows():

                        review_text = str(row["review"])
                        product_name = str(row["product"])

                        sentiment, confidence = predict_review(
                            review_text
                        )

                        aspects = analyze_aspects(review_text)

                        results.append({
                            "product": product_name,
                            "review": review_text,
                            "sentiment": sentiment,
                            "confidence": round(confidence, 2),
                            "aspects": ", ".join(aspects)
                        })

                        save_review(
                            product_name,
                            review_text,
                            sentiment,
                            confidence
                        )

                    result_df = pd.DataFrame(results)

                    st.success(
                        f"{len(result_df)} reviews analyzed!"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Analysis Report",
                        data=csv_data,
                        file_name="sentiment_analysis_report.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"CSV error: {e}")

# ---------------- REVIEW HISTORY ----------------

elif page == "Review History":

    st.subheader("🕒 Saved Review History")

    history = get_history()

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Full History",
            data=csv_data,
            file_name="review_history.csv",
            mime="text/csv"
        )

    else:
        st.info("No saved reviews available yet.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ReviewIQ AI | Built using Python, Streamlit, "
    "Scikit-learn, Pandas, Plotly and SQLite"
)
# ---------------- MODEL PERFORMANCE ----------------

if page == "Model Performance":

    st.subheader("📈 Machine Learning Model Performance")

    try:
        with open("evaluation_metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        report = metrics["report"]

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        col1, col2, col3 = st.columns(3)

        for column, label in zip(
            [col1, col2, col3],
            ["Positive", "Negative", "Neutral"]
        ):
            values = report.get(label, {})

            with column:
                st.metric(
                    f"{label} F1-score",
                    f"{values.get('f1-score', 0) * 100:.2f}%"
                )
                st.metric(
                    f"{label} Precision",
                    f"{values.get('precision', 0) * 100:.2f}%"
                )
                st.metric(
                    f"{label} Recall",
                    f"{values.get('recall', 0) * 100:.2f}%"
                )

        st.divider()
        st.subheader("Confusion Matrix")

        cm = pd.read_csv(
            "confusion_matrix.csv",
            index_col=0
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Purples",
            labels={
                "x": "Predicted Sentiment",
                "y": "Actual Sentiment",
                "color": "Reviews"
            }
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except FileNotFoundError:
        st.warning(
            "Evaluation files not found. "
            "Run python train_model.py first."
        )
# ---------------- PRODUCT COMPARISON ----------------

if page == "Product Comparison":

    st.subheader("📊 Product Sentiment Comparison")

    history = get_history()

    if history.empty:
        st.info("Analyze product reviews first.")

    else:
        products = sorted(
            history["product"].dropna().unique().tolist()
        )

        if len(products) < 2:
            st.warning(
                "Please analyze reviews for at least two different products."
            )

        else:
            col1, col2 = st.columns(2)

            with col1:
                product1 = st.selectbox(
                    "Select First Product",
                    products,
                    index=0
                )

            with col2:
                product2 = st.selectbox(
                    "Select Second Product",
                    products,
                    index=1
                )

            if product1 == product2:
                st.warning("Select two different products.")

            else:
                comparison = history[
                    history["product"].isin(
                        [product1, product2]
                    )
                ]

                summary = (
                    comparison.groupby(
                        ["product", "sentiment"]
                    )
                    .size()
                    .reset_index(name="Reviews")
                )

                st.divider()
                st.subheader("Sentiment Distribution")

                fig = px.bar(
                    summary,
                    x="product",
                    y="Reviews",
                    color="sentiment",
                    barmode="group",
                    text="Reviews",
                    title="Product Sentiment Comparison"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Product Statistics")

                for product in [product1, product2]:

                    product_data = comparison[
                        comparison["product"] == product
                    ]

                    total = len(product_data)

                    positive = (
                        product_data["sentiment"] == "Positive"
                    ).sum()

                    negative = (
                        product_data["sentiment"] == "Negative"
                    ).sum()

                    neutral = (
                        product_data["sentiment"] == "Neutral"
                    ).sum()

                    positive_rate = (
                        positive / total * 100 if total else 0
                    )

                    st.markdown(f"### {product}")

                    a, b, c, d = st.columns(4)

                    a.metric("Total Reviews", total)
                    b.metric("Positive", int(positive))
                    c.metric("Negative", int(negative))
                    d.metric("Positive Rate", f"{positive_rate:.1f}%")