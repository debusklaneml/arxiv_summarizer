import urllib.parse
import urllib.request
import polars as pl
import xml.etree.ElementTree as ET
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# Custom CSS for background color
background_color_css = """
<style>
    /* Change the background color of the main content */
    .stApp {
        background-color: #3a3c40;
    }
</style>
"""

# Apply the custom CSS
st.markdown(background_color_css, unsafe_allow_html=True)

st.title("Arxiv Summarizer")

st.header("This tool allows you to search arXiv.")


def fetch_arxiv_results(search_term, start=0, max_results=10):
    """
    Fetch articles from arXiv based on a search term.

    Args:
        search_term (str): The term to search for in the arXiv database.
        start (int): The starting index of results (default is 0).
        max_results (int): The maximum number of results to fetch (default is 10).

    Returns:
        str: The XML response from the arXiv API as a decoded string.
    """
    # Base URL for the arXiv API
    base_url = "http://export.arxiv.org/api/query?"
    
    # Construct query parameters
    query_params = {
        "search_query": f"all:{search_term}",
        "start": start,
        "max_results": max_results
    }
    
    # Encode query parameters into URL format
    query_string = urllib.parse.urlencode(query_params)
    
    # Combine base URL and query string
    url = base_url + query_string
    
    try:
        # Open the URL and read the response
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        return data
    except Exception as e:
        return f"An error occurred: {e}"


testing = fetch_arxiv_results("confidence", max_results=5)
print(testing)

@st.cache_resource
def load_summarization_model():
    """
    Load and cache the summarization model to avoid reloading on every app run.
    
    Returns:
        Summarization pipeline: A pre-trained summarization model pipeline.
    """
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load the cached model
summarizer = load_summarization_model()

def summarize_text(text, max_length=140, min_length=30):
    """
    Summarize text using a pre-trained model.

    Args:
        text (str): The text to summarize.
        max_length (int): Maximum summary length.
        min_length (int): Minimum summary length.

    Returns:
        str: The summarized text.
    """
    if text and len(text) > min_length:
        summary = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
        return summary[0]['summary_text']
    return text

def parse_arxiv_to_polars_and_summarize(xml_data):
    """
    Parse arXiv XML data into a Polars DataFrame and summarize all articles into a single paragraph.

    Args:
        xml_data (str): The XML data as a string.

    Returns:
        tuple: A Polars DataFrame containing parsed arXiv data and a single summarized paragraph.
    """
    # Parse the XML string
    root = ET.fromstring(xml_data)
    
    # Define namespaces (arXiv uses Atom XML format)
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom'
    }
    
    # Extract entries and aggregate summaries
    entries = []
    all_summaries = ""
    
    for entry in root.findall('atom:entry', namespaces):
        # Extract relevant fields
        title = entry.find('atom:title', namespaces).text if entry.find('atom:title', namespaces) is not None else None
        summary = entry.find('atom:summary', namespaces).text if entry.find('atom:summary', namespaces) is not None else None
        authors = [author.find('atom:name', namespaces).text for author in entry.findall('atom:author', namespaces)]
        published = entry.find('atom:published', namespaces).text if entry.find('atom:published', namespaces) is not None else None
        link = entry.find('atom:link[@rel="alternate"]', namespaces).attrib['href'] if entry.find('atom:link[@rel="alternate"]', namespaces) is not None else None

        # Aggregate summaries
        if summary:
            all_summaries += " " + summary.strip()
        
        # Generate concise summary
        concise_summary = summarize_text(summary) if summary else None
        
        # Append to entries list
        entries.append({
            "title": title,
            "summary": concise_summary,
            "authors": ", ".join(authors),
            "published": published,
            "link": link
        })
    
    # Create a Polars DataFrame
    df = pl.DataFrame(entries).sort('published', descending=True)
    
    # Generate a single summarized paragraph from all summaries
    overall_summary = summarize_text(all_summaries) if all_summaries else "No summaries available."
    
    return df, overall_summary

# Add user input field for search
search_term = st.text_input("Search for articles on arXiv...", placeholder="Enter a topic, e.g., 'machine learning'")
search_count = st.number_input("How many articles should be downloaded?", min_value=1, max_value=100, value=10, step=1)

# Fetch and display results
if search_term:
    with st.spinner("Fetching results..."):
        xml_results = fetch_arxiv_results(search_term, max_results=int(search_count))
        
    if "An error occurred" in xml_results:
        st.error(xml_results)
    else:
        df, overall_summary = parse_arxiv_to_polars_and_summarize(xml_results)
        
        if not df.is_empty():
            st.subheader(f"Results for '{search_term}'")
            st.dataframe(df)
            
            st.subheader("Overall Summary")
            st.write(overall_summary)
        else:
            st.warning("No results found. Try a different search term.")