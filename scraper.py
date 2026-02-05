import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

BASE_URL = "https://www.lpu.in"
SITEMAP = "https://www.lpu.in/sitemap.xml"

def get_urls():
    r = requests.get(SITEMAP, timeout=15)
    soup = BeautifulSoup(r.text, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

data = []

urls = get_urls()
print(f"Found {len(urls)} URLs")

for url in urls:
    try:
        r = requests.get(url, timeout=15)
        text = clean_text(r.text)

        if len(text) > 300:
            data.append({
                "url": url,
                "content": text
            })
    except:
        pass

with open("data/lpu_pages.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(data)} clean pages")