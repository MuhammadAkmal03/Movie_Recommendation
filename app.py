import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from src.data.loader import load_models
from src.models.recommender import get_recommendations, search_by_keywords
from src.utils.helpers import ensure_poster_url
from src.utils.feedback import save_rating, get_feedback_stats
from src.utils.ab_testing import get_ab_test_summary

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Movie Recommender", 
    layout="centered",
    page_icon="🎬"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<h1 class='main-header'>🎬 Movie Recommendation Engine</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    with st.spinner("Loading models... (this may take a moment)"):
        models = load_models()
        
    # Ensure poster URLs exist
    models['org_dataset'] = ensure_poster_url(models['org_dataset'])
    
    # Sidebar info
    with st.sidebar:
        st.info("This system uses NLP-based search and hybrid recommendation algorithms.")
        st.markdown("### 📊 Stats")
        st.write(f"Movies: {len(models['movies_tag']):,}")
        
        # Feedback stats
        st.markdown("### 💬 Your Feedback")
        feedback_stats = get_feedback_stats()
        if feedback_stats["total_ratings"] > 0:
            st.write(f"Total Ratings: {feedback_stats['total_ratings']}")
            st.write(f"👍 Likes: {feedback_stats['likes']} ({feedback_stats['like_percentage']}%)")
            st.write(f"👎 Dislikes: {feedback_stats['dislikes']} ({feedback_stats['dislike_percentage']}%)")
        else:
            st.write("No ratings yet. Start rating movies!")
        
        st.markdown("### 💡 Search Tips")
        st.markdown("""
        **Try searching for:**
        - Actor names: "Tom Cruise"
        - Directors: "Christopher Nolan"
        - Genres: "action thriller"
        - Themes: "space exploration"
        - Combinations: "DiCaprio thriller"
        """)
        
    # Create tabs for different modes
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Movie-Based Recommendations",
        "🤖 AI Assistant", 
        "🔍 NLP Search",
        "📊 A/B Testing", 
        "📈 MLflow Metrics"
    ])
    
    # Tab 1: Movie-Based Recommendations
    with tab1:
        st.markdown("### Get Recommendations Based on a Movie")
        
        # Main UI
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Movie selection
            movies_df = pd.DataFrame(models['movies_tag'])
            movie_name = st.selectbox(
                "Select a Movie you like:", 
                movies_df['title'].values,
                index=None,
                placeholder="Type to search..."
            )

        with col2:
            # Method selection
            method = st.selectbox(
                "Recommendation Method:", 
                [
                    "Hybrid Recommendation",
                    "Content Based Filtering",
                    "Collaborative Filtering (KNN)",
                    "Collaborative Filtering (SVD)"
                ]
            )

        st.markdown("")
        
        if st.button("Get Recommendations", type="primary", key="recommend_btn"):
            if not movie_name:
                st.warning("⚠️ Please select a movie first!")
                return
                
            with st.spinner(f"Finding similar movies using {method}..."):
                recommendations = get_recommendations(movie_name, method, models)

            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations for **{movie_name.title()}**")
                st.markdown("---")
                
                # Display recommendations
                num_cols = min(len(recommendations), 5)
                cols = st.columns(num_cols)

                for idx, rec_data in enumerate(recommendations[:num_cols]):
                    # Handle both old format (title, poster_url) and new format (title, poster_url, explanations)
                    if len(rec_data) == 3:
                        title, poster_url, explanations = rec_data
                    else:
                        title, poster_url = rec_data
                        explanations = []
                    
                    with cols[idx]:
                        if poster_url:
                            st.image(poster_url, use_column_width=True)
                        else:
                            st.image("assets/noimage.png", use_column_width=True)
                        
                        # Fixed height title using HTML/CSS
                        display_title = title.title()
                        st.markdown(
                            f"""
                            <div style="height: 60px; display: flex; align-items: center; margin-bottom: 8px;">
                                <p style="margin: 0; font-weight: bold; font-size: 14px; line-height: 1.3;">
                                    {idx+1}. {display_title}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Show explanations
                        if explanations:
                            with st.expander("ℹ️ Why recommended?"):
                                for explanation in explanations[:3]:  # Show top 3 reasons
                                    st.caption(f"✓ {explanation}")
                        else:
                            # Add empty space to maintain alignment when no explanations
                            st.markdown('<div style="height: 48px;"></div>', unsafe_allow_html=True)
                        
                        # Rating buttons with callbacks
                        col_like, col_dislike = st.columns(2)
                        
                        like_key = f"like_rec_{idx}_{movie_name}"
                        dislike_key = f"dislike_rec_{idx}_{movie_name}"
                        
                        with col_like:
                            st.button(
                                "👍", 
                                key=like_key,
                                use_column_width=True,
                                on_click=save_rating,
                                args=(title, "like", movie_name, method)
                            )
                        with col_dislike:
                            st.button(
                                "👎", 
                                key=dislike_key,
                                use_column_width=True,
                                on_click=save_rating,
                                args=(title, "dislike", movie_name, method)
                            )
            else:
                if "Collaborative" in method:
                    st.error(f"😕 '{movie_name.title()}' not found.")
                    st.info("💡 Try using **Content Based Filtering** or **Hybrid Recommendation** instead, or select a different movie.")
                else:
                    st.error("😕 No recommendations found. Try another movie or method.")
    
    # Tab 2: AI Assistant
    with tab2:
        from src.agent.ui import render_agent_tab
        render_agent_tab(models)
    
    # Tab 3: NLP Search
    with tab3:
        st.markdown("### Search Movies by Keywords")
        st.markdown("Type anything: actor names, directors, genres, themes, or plot keywords!")
        
        # Search input
        search_query = st.text_input(
            "Enter your search:",
            placeholder="e.g., 'Leonardo DiCaprio thriller' or 'space adventure'",
            key="search_query"
        )
        
        # Search button directly below input
        if st.button("🔍 Search", type="primary", key="search_btn"):
            if not search_query or search_query.strip() == "":
                st.warning("⚠️ Please enter a search query!")
            else:
                with st.spinner(f"Searching for '{search_query}'..."):
                    results = search_by_keywords(search_query, models, n_results=5)
                
                if results:
                    st.success(f"Found {len(results)} movies matching '{search_query}'")
                    st.markdown("---")
                    
                    # Display results in single row (max 5)
                    cols = st.columns(len(results))
                    for idx, (title, poster_url, score) in enumerate(results):
                        with cols[idx]:
                            if poster_url:
                                st.image(poster_url, use_column_width=True)
                            else:
                                st.image("assets/noimage.png", use_column_width=True)
                            # Fixed height title using HTML/CSS
                            display_title = title.title()
                            if len(display_title) > 40:
                                display_title = display_title[:37] + "..."
                            st.markdown(
                                f"""
                                <div style="height: 60px; display: flex; align-items: center; margin-bottom: 8px;">
                                    <p style="margin: 0; font-weight: bold; font-size: 14px; line-height: 1.3;">
                                        {idx+1}. {display_title}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.caption(f"Relevance: {score:.0%}")
                            
                            # Rating buttons with callbacks
                            col_like, col_dislike = st.columns(2)
                            
                            like_key = f"like_search_{idx}_{search_query}"
                            dislike_key = f"dislike_search_{idx}_{search_query}"
                            
                            with col_like:
                                st.button(
                                    "👍", 
                                    key=like_key,
                                    use_column_width=True,
                                    on_click=save_rating,
                                    args=(title, "like", search_query, "NLP Search")
                                )
                            with col_dislike:
                                st.button(
                                    "👎", 
                                    key=dislike_key,
                                    use_column_width=True,
                                    on_click=save_rating,
                                    args=(title, "dislike", search_query, "NLP Search")
                                )
                else:
                    st.error(f"😕 No movies found for '{search_query}'. Try different keywords!")
    
    # Tab 4: A/B Testing Dashboard
    with tab4:

    

        st.markdown("### 📊 A/B Testing: Method Performance Comparison")
        st.markdown("Compare which recommendation method users like most based on feedback data.")
        
        # Get A/B test data
        ab_summary = get_ab_test_summary()
        
        if ab_summary["total_tests"] == 0:
            st.info("📝 No test data yet. Start rating recommendations to see A/B test results!")
        else:
            # Show winner
            st.success(f"🏆 **Winner:** {ab_summary['winner']['method']} with {ab_summary['winner']['like_rate']}% like rate")
            st.markdown("---")
            
            # Performance table
            st.markdown("### 📈 Method Performance")
            
            # Create comparison data
            methods_data = []
            for method_name, method_data in ab_summary["method_performance"].items():
                methods_data.append({
                    "Method": method_name,
                    "Total Ratings": method_data["total_ratings"],
                    "Likes": method_data["likes"],
                    "Dislikes": method_data["dislikes"],
                    "Like Rate": f"{method_data['like_rate']}%"
                })
            
            st.dataframe(pd.DataFrame(methods_data), use_column_width=True, hide_index=True)
            
            # Insights
            st.markdown("---")
            st.markdown("### 💡 Insights")
            
            winner = ab_summary["winner"]
            st.markdown(f"""
            - **Best Performer**: {winner['method']} with {winner['like_rate']}% like rate
            - **Total Tests**: {ab_summary['total_tests']} ratings collected
            - **Most Popular**: {max(ab_summary['method_performance'].items(), key=lambda x: x[1]['total_ratings'])[0]} ({max(m['total_ratings'] for m in ab_summary['method_performance'].values())} ratings)
            """)
            
            # Recommendations
            if winner['like_rate'] >= 80:
                st.success("🎯 Excellent performance! Users love this method.")
            elif winner['like_rate'] >= 60:
                st.info("👍 Good performance. Consider optimizing further.")
            else:
                st.warning("⚠️ Room for improvement. Analyze user feedback.")
    
    # Tab 5: MLflow Metrics
    with tab5:
        st.markdown("### 📈 MLflow Experiment Tracking")
        st.markdown("Track model training, recommendations, and A/B tests with MLflow.")
        
        # Instructions
        st.info("""
        All recommendation requests and user ratings are automatically logged to MLflow.
        """)
        
        # Manual logging button
        st.markdown("### 📊 Log Current A/B Test Results")
        
        if st.button("Log to MLflow", type="primary"):
            try:
                from src.utils.mlflow_logger import log_ab_test_metrics
                ab_summary = get_ab_test_summary()
                
                if ab_summary["total_tests"] > 0:
                    log_ab_test_metrics(ab_summary)
                    st.success("✅ A/B test results logged to MLflow!")
                else:
                    st.warning("No A/B test data to log yet.")
            except Exception as e:
                st.error(f"Failed to log: {str(e)}")
        
        # What's being tracked
        st.markdown("---")
        st.markdown("### 📋 What's Being Tracked")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Recommendations**")
            st.write("- Method used")
            st.write("- Query movie")
            st.write("- Response time")
            st.write("- Results count")
        with col2:
            st.markdown("**User Ratings**")
            st.write("- Movie rated")
            st.write("- Like/Dislike")
            st.write("- Method used")
            st.write("- Timestamp")
        with col3:
            st.markdown("**A/B Tests**")
            st.write("- Like rates per method")
            st.write("- Total ratings")
        
        # Quick stats
        st.markdown("---")
        st.markdown("### 📈 Current Session Stats")
        
        stats = get_feedback_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ratings", stats["total_ratings"])
        with col2:
            st.metric("Like Rate", f"{stats.get('like_percentage', 0)}%")
        with col3:
            st.metric("Experiments", "Auto-tracked")

if __name__ == "__main__":
    main()
