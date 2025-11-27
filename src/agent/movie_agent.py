"""
AI Agent for Movie Recommendations

This module implements a conversational AI agent that can:
- Understand natural language queries
- Use tools to search and recommend movies
- Maintain conversation context
"""

import json
import re
from typing import List, Dict, Any, Optional

class MovieAgent:
    """
    Conversational AI agent for movie recommendations
    """
    
    def __init__(self, api_key: str, models: Dict[str, Any]):
        """
        Initialize the agent
        
        Args:
            api_key: Google Gemini API key
            models: Dictionary containing loaded models and datasets
        """
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is not installed.")

        self.genai.configure(api_key=api_key)
        self.model = self.genai.GenerativeModel('gemini-2.5-flash')
        self.models = models
        self.conversation_history = []
        
        # Define available tools
        self.tools = {
            "search_movies": self._search_movies,
            "get_recommendations": self._get_recommendations,
            "get_movie_details": self._get_movie_details
        }
    
    def _search_movies(self, query: str, limit: int = 5) -> List[str]:
        """
        Tool: Search for movies by keywords
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of movie titles
        """
        from src.models.recommender import search_by_keywords
        
        # Use AI to optimize the query
        optimized_query = self._optimize_search_query(query)
        
        results = search_by_keywords(optimized_query, self.models, n_results=limit)
        if not results:
            return []
        
        # Return just titles
        return [title for title, _, _ in results]
    
    def _optimize_search_query(self, query: str) -> str:
        """
        Use AI to extract relevant keywords from natural language query
        
        Example:
            "I'd like to watch Nolan movies" -> "Nolan"
            "Show me some action thriller films" -> "action thriller"
        """
        try:
            prompt = f"""Extract only the relevant movie-related keywords from this query.
Remove filler words like: watch, show, find, movies, films, like, want, some, etc.
Keep: director names, actor names, genres, themes, movie titles.

Query: "{query}"

Return ONLY the cleaned keywords, nothing else. Examples:
- "I want to watch Nolan movies" -> "Nolan"
- "Show me action thriller films" -> "action thriller"
- "Find movies like Inception" -> "Inception"

Cleaned keywords:"""
            
            response = self.model.generate_content(prompt)
            cleaned = response.text.strip().strip('"').strip("'")
            
            # If AI returns empty or too short, use original
            if len(cleaned) < 2:
                return query
            
            return cleaned
        except:
            # If optimization fails, use original query
            return query
    
    def _get_recommendations(self, movie_title: str, method: str = "Hybrid Recommendation", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Tool: Get movie recommendations
        
        Args:
            movie_title: Title of the movie
            method: Recommendation method
            limit: Maximum number of results
            
        Returns:
            List of recommended movies with explanations
        """
        from src.models.recommender import get_recommendations
        
        results = get_recommendations(movie_title, method, self.models)
        if not results:
            return []
        
        # Format results with explanations
        recommendations = []
        for i, rec_data in enumerate(results[:limit]):
            if len(rec_data) == 3:
                title, _, explanations = rec_data
                recommendations.append({
                    "title": title,
                    "reasons": explanations[:3]  # Top 3 reasons
                })
            else:
                title, _ = rec_data
                recommendations.append({
                    "title": title,
                    "reasons": []
                })
        
        return recommendations
    
    def _get_movie_details(self, movie_title: str) -> Optional[Dict[str, Any]]:
        """
        Tool: Get movie details
        
        Args:
            movie_title: Title of the movie
            
        Returns:
            Dictionary with movie details
        """
        from src.models.recommender import normalize_title
        
        org_dataset = self.models['org_dataset']
        normalized = normalize_title(movie_title)
        
        match = org_dataset[org_dataset['title'] == normalized]
        if match.empty:
            return None
        
        movie = match.iloc[0]
        
        # Parse genres
        try:
            import ast
            genres = ast.literal_eval(movie['genres']) if isinstance(movie['genres'], str) else movie['genres']
            genre_names = [g.get('name', '') for g in genres if isinstance(g, dict)]
        except:
            genre_names = []
        
        return {
            "title": movie['title'],
            "genres": genre_names,
            "year": str(movie.get('release_date', ''))[:4] if movie.get('release_date') else 'Unknown',
            "overview": movie.get('overview', 'No description available'),
            "rating": float(movie.get('vote_average', 0))
        }
    
    def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name
        
        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        return None
    
    def chat(self, user_message: str) -> str:
        """
        Main chat function - processes user message and returns response
        
        Args:
            user_message: User's message
            
        Returns:
            Agent's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Create system prompt with tools and context
        system_prompt = self._create_system_prompt(user_message)
        
        try:
            # Get AI response
            response = self.model.generate_content(system_prompt)
            agent_response = response.text
            
            # Check if AI wants to use tools
            tool_calls = self._parse_tool_calls(agent_response)
            
            if tool_calls:
                # Execute tools and get results
                tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call['name'], **tool_call['args'])
                    tool_results.append({
                        "tool": tool_call['name'],
                        "result": result
                    })
                
                # Generate final response with tool results
                final_prompt = self._create_final_response_prompt(user_message, tool_results)
                final_response = self.model.generate_content(final_prompt)
                agent_response = final_response.text
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": agent_response
            })
            
            return agent_response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
            self.conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg
    
    def _create_system_prompt(self, user_message: str) -> str:
        """Create system prompt with tools and conversation context"""
        
        # Build conversation context
        context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.conversation_history[-6:]  # Last 3 turns
        ])
        
        prompt = f"""You are a helpful movie recommendation assistant with access to both a local movie database and your own extensive knowledge of cinema.

AVAILABLE TOOLS (Local Database):
1. search_movies(query: str, limit: int = 5)
   - Search for movies by keywords in the local database
   - Example: search_movies("nolan thriller")

2. get_recommendations(movie_title: str, method: str = "Hybrid Recommendation", limit: int = 5)
   - Get similar movies from the local database
   - Example: get_recommendations("Inception")

3. get_movie_details(movie_title: str)
   - Get detailed information about a movie from the local database
   - Example: get_movie_details("Inception")

HOW TO RESPOND:
1. **Prefer Local Database**: When possible, use the tools to search the local database first.
2. **Use Your Knowledge**: If the local database has no results, or if the user asks for general recommendations (e.g., "best movies of 2023", "Oscar winners", "trending movies"), you can recommend movies from your general knowledge.
3. **Be Transparent**: When recommending from your knowledge, mention it like: "Based on my knowledge, here are some great options..." or "These popular movies might interest you..."
4. **Mix Both**: You can combine both sources: "From our database, I found... Additionally, you might also enjoy these well-known films..."

TO USE A TOOL, respond with:
TOOL: tool_name(arg1="value1", arg2="value2")

You can use multiple tools:
TOOL: search_movies(query="sci-fi")
TOOL: get_movie_details(movie_title="Inception")

CONVERSATION HISTORY:
{context}

USER: {user_message}

Respond naturally and helpfully. Use tools when appropriate, but don't hesitate to use your own knowledge when it makes sense.
"""
        return prompt
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from AI response"""
        tool_calls = []
        
        # Look for TOOL: pattern
        pattern = r'TOOL:\s*(\w+)\((.*?)\)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for tool_name, args_str in matches:
            try:
                # Parse arguments
                args = {}
                # Simple parsing for key="value" pairs
                arg_pattern = r'(\w+)=["\'](.*?)["\']'
                arg_matches = re.findall(arg_pattern, args_str)
                for key, value in arg_matches:
                    args[key] = value
                
                tool_calls.append({
                    "name": tool_name,
                    "args": args
                })
            except:
                continue
        
        return tool_calls
    
    def _create_final_response_prompt(self, user_message: str, tool_results: List[Dict]) -> str:
        """Create prompt for final response with tool results"""
        
        results_text = "\n".join([
            f"Tool: {r['tool']}\nResult: {json.dumps(r['result'], indent=2)}"
            for r in tool_results
        ])
        
        prompt = f"""Based on the tool results below, provide a helpful response to the user.

USER QUESTION: {user_message}

TOOL RESULTS:
{results_text}

Provide a natural, conversational response. Format movie recommendations nicely with numbers and brief descriptions.
DO NOT include the TOOL: syntax in your response.
"""
        return prompt
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
