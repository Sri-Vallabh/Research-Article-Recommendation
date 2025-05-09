import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity

# Load models and data
with open("journal_main.pkl", "rb") as f:
    journal_main = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("journal_tfidf_matrix.pkl", "rb") as f:
    journal_tfidf_matrix = pickle.load(f)

# Parameters
journal_threshold = 4
article_threshold = 10

# Logic
def get_journal_index(user_input):
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, journal_tfidf_matrix).flatten()
    indices = cosine_similarities.argsort()[::-1]
    top_recommendations = [i for i in indices if cosine_similarities[i] > 0][:min(journal_threshold, len(indices))]
    return top_recommendations

def get_article_recommendations(user_input):
    recommended_journals = get_journal_index(user_input)
    recommendations = []
    for journal_id in recommended_journals:
        user_tfidf = journal_main['article_vectorizer'][journal_id].transform([user_input])
        cosine_similarities = cosine_similarity(user_tfidf, journal_main['article_matrix'][journal_id]).flatten()
        indices = cosine_similarities.argsort()[::-1]
        top_recommendation_articles = [
            (cosine_similarities[i], i, journal_id)
            for i in indices if cosine_similarities[i] > 0
        ][:min(article_threshold, len(indices))]
        recommendations += top_recommendation_articles
    recommendations.sort(reverse=True)
    return recommendations

def get_links(user_input):
    check = validation(user_input)
    if check['validation'] == 'valid':
        recommendations = get_article_recommendations(check['sentence'])
        links = []
        for article in recommendations:
            similarity, article_id, journal_id = article
            link = {
                "title": journal_main['article_df'][journal_id].iloc[article_id, 0],
                "url": journal_main['article_df'][journal_id].iloc[article_id, 1],
                "article_id": int(article_id),
                "journal_id": int(journal_id)
            }
            links.append(link)
        return links
    return []

def validation(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    adjectives = [word for word, pos in tagged_words if pos.startswith('JJ')]
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]

    result = {}
    if not adjectives and not nouns:
        result['validation'] = 'invalid'
    else:
        combined_sentence = f"{' '.join(adjectives)} {' '.join(nouns)}".strip()
        result['validation'] = 'valid'
        result['sentence'] = combined_sentence

    return result

# Streamlit UI
st.set_page_config(page_title="Discover Research Articles", layout="centered")
st.markdown("""
    <style>
        .title { text-align: center; font-size: 36px; color: #2c3e50; font-weight: bold; margin-top: 30px; }
        .subtitle { text-align: center; font-size: 18px; color: #333; margin-bottom: 20px; font-style: italic; }
        .section { padding: 30px; margin-bottom: 40px; }
        .input-box { width: 100%; padding: 15px; font-size: 18px; border-radius: 8px; border: 1px solid #ddd; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .button { background-color: #3498db; color: white; padding: 15px 25px; font-size: 20px; border: none; border-radius: 8px; cursor: pointer; width: 100%; }
        .button:hover { background-color: #2980b9; }
        .article { padding: 15px; margin-bottom: 12px; border-radius: 8px; background-color: #ffffff; box-shadow: 0 3px 6px rgba(0,0,0,0.1); }
        .article-title { font-size: 20px; color: #3498db; font-weight: bold; text-decoration: none; }
        .article-meta { font-size: 12px; color: #888; margin-top: 5px; }
    </style>
    <div class="title">🔎 Discover Relevant Research Articles</div>
    <div class="subtitle">Enter a topic, keyword, or phrase to explore the latest articles in your field of interest!</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Find Research Articles")
    st.markdown("Type in a topic or keyword that you'd like to explore. Our system will fetch articles based on your input.")
    article_input = st.text_input("", placeholder="e.g., Quantum Computing, AI, Rocket Science ...", key="article_input", 
                                  help="Enter your research topic or keyword to find related articles.")
    
    if st.button("🔗 Find Articles", key="generate_links"):
        if article_input:
            validation_result = validation(article_input)
            
            if validation_result['validation'] == 'invalid':
                st.warning("Please try entering more descriptive terms, including nouns or adjectives.")
            else:
                result = get_links(validation_result['sentence'])
                
                if result:
                    st.markdown("### 🔗 Top Matching Articles", unsafe_allow_html=True)
                    for i, article in enumerate(result):
                        st.markdown(f"""
                            <div class="article">
                                <a class="article-title" href="{article['url']}" target="_blank">
                                    {i+1}. {article['title']}
                                </a>
                                <p class="article-meta">Article ID: {article['article_id']} | Journal ID: {article['journal_id']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No articles matched your query. Try being more specific with your input.")
        else:
            st.warning("Please enter a topic or keyword to get article recommendations.")
    
    st.markdown('</div>', unsafe_allow_html=True)
