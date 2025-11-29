import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Dataset (extracted from notebook) ---
DATA = {
    "message": [
        # HIGH priority
        "Water is entering our house, we are stuck on the first floor",
        "My grandmother is very sick and we can't reach hospital",
        "The bridge is broken and people are trapped on the other side",
        "We are on the roof, water is rising quickly, please send help",
        "Children and elderly are stuck inside the house with water inside",
        "House is surrounded by flood water, we cannot get out",
        "My father collapsed and there is no way to reach a clinic",
        "We are stuck in a tree after the flood, need urgent rescue",

        # MEDIUM priority
        "We are safe but there is no electricity in our area",
        "Road is flooded, but we are at a friend's house and okay",
        "We need drinking water and dry food, but our house is safe",
        "We are staying at a school as a shelter, need blankets",
        "House is wet and some furniture is damaged, but everyone is safe",
        "Our ground floor is flooded but we moved upstairs and are safe",
        "We need baby milk and medicine but not in immediate danger",

        # LOW priority
        "Can you tell me when schools will reopen after the flood",
        "Where can I donate clothes for flood victims",
        "Is there any volunteer group I can join to help",
        "Where can I get information about relief programs",
        "Can you send me updates about flood news in Colombo",
        "How to apply for government compensation after the flood",
        "Is there a place to give cooked food for people in shelters",
    ],
    "priority": [
        "high","high","high","high","high","high","high","high",
        "medium","medium","medium","medium","medium","medium","medium",
        "low","low","low","low","low","low","low"
    ]
}

# --- Model training (small dataset; quick) ---
@st.cache_data(show_spinner=False)
def load_model():
    df = pd.DataFrame(DATA)
    X = df['message']
    y = df['priority']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return vectorizer, model

vectorizer, model = load_model()


def predict_priority(text: str) -> str:
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Flood Message Priority Helper", page_icon="🌊")
st.title("🌊 Flood Message Priority Helper")
st.write("This simple AI classifies flood-related messages into High, Medium, or Low priority.")
st.markdown("---")

with st.expander("ℹ️ Example messages / dataset"):
    st.write("This app is a small educational prototype trained on example messages:")
    st.dataframe(pd.DataFrame(DATA))

user_message = st.text_area(
    "✉️ Enter a flood-related message:",
    placeholder="Example: Water is rising fast and we are stuck upstairs with no way out...",
    height=120,
)

if st.button("Predict Priority"):
    if not user_message or not user_message.strip():
        st.warning("Please type a message before predicting.")
    else:
        priority = predict_priority(user_message)
        if priority == "high":
            st.error("🚨 Predicted Priority: HIGH — Possible urgent / life-threatening situation.")
        elif priority == "medium":
            st.warning("⚠️ Predicted Priority: MEDIUM — Needs support, but not immediately life-threatening.")
        else:
            st.info("ℹ️ Predicted Priority: LOW — General information / non-urgent request.")

st.markdown("---")
with st.expander("ℹ️ How this works"):
    st.write(
        "- Trained on a small set of example messages labeled as high/medium/low.\n"
        "- Uses TF–IDF to vectorize text and Logistic Regression to classify.\n"
        "- This is a prototype for demonstration only; do not use for real emergencies."
    )

st.caption("Built by Dinithi — educational prototype")
