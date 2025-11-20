from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
import os
import time

BASE_URL = "https://www.formaculture.com/"

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def extract_links(driver, current_url):
    """Extract internal links from the current page"""
    links = set()
    elements = driver.find_elements(By.TAG_NAME, "a")

    for el in elements:
        href = el.get_attribute("href")
        if href and href.startswith(BASE_URL):
            links.add(href.split("#")[0])  # remove #anchors

    return links


def scrape_page(url):
    """Scrape full rendered text from a page"""
    driver = get_driver()
    driver.get(url)
    time.sleep(4)

    text = driver.find_element(By.TAG_NAME, "body").text
    new_links = extract_links(driver, url)

    driver.quit()
    return text, new_links


def save_page(url, text):
    """Save page text with filename from URL"""
    os.makedirs("data", exist_ok=True)

    filename = url.replace(BASE_URL, "")
    filename = filename.strip("/") or "home"
    filename = filename.replace("/", "_")

    path = f"data/{filename}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# URL: {url}\n\n")
        f.write(text)


# ---------------------------
# 🔥 Recursive Crawl
# ---------------------------

visited = set()
to_visit = {BASE_URL}

while to_visit:
    url = to_visit.pop()

    if url in visited:
        continue

    print(f"\n🔍 Crawling: {url}")
    visited.add(url)

    try:
        text, found_links = scrape_page(url)
        save_page(url, text)

        # Add new links to crawling queue
        for link in found_links:
            if link not in visited:
                to_visit.add(link)

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")

print("\n🎉 DONE! All pages scraped.")
