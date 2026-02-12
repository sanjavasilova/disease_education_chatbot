from bs4 import BeautifulSoup
import requests
import os
import csv
import pandas as pd
from pypdf import PdfReader

def extract_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(html, "html.parser")
    
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "iframe", "noscript"]):
        tag.decompose()
        
    for tag in soup.find_all(lambda t: t.get('id', '') and ('header' in t['id'].lower() or 'footer' in t['id'].lower() or 'nav' in t['id'].lower() or 'menu' in t['id'].lower())):
        tag.decompose()

    main_content = (
        soup.find("main") or 
        soup.find("article") or 
        soup.find("div", {"id": "main-content"}) or 
        soup.find("div", {"id": "content"}) or
        soup.find("div", {"class": "content-area"})
    )
    
    if main_content:
        for sub in main_content(["nav", "header", "footer", "aside"]):
            sub.decompose()
        return main_content.get_text(separator="\n", strip=True)
        
    return soup.get_text(separator="\n", strip=True)

def extract_pdf(path):
    reader = PdfReader(path)
    return "\n".join(p.extract_text() for p in reader.pages)

def extract_csv(path):
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def extract_parquet(path):
    df = pd.read_parquet(path)
    return df.to_dict('records')

def save_text(text, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
