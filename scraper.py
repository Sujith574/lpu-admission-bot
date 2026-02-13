import requests
from bs4 import BeautifulSoup

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Remove excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)

def scrape_page(url):
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return None

    cleaned_text = clean_text(response.text)

    return {
        "url": url,
        "content": cleaned_text
    }
