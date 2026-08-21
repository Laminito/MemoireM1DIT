"""
Scraper pour Expat-Dakar : annonces de vente de biens immobiliers résidentiels
(appartements et maisons) sur tout le Sénégal.

Usage:
    python scraper.py

Sortie:
    data/raw/expat_dakar_listings_raw.csv
"""

import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.expat-dakar.com"
CATEGORIES = ["maisons-a-vendre", "appartements-a-vendre"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
REQUEST_DELAY_SECONDS = 1.5
MAX_PAGES_PER_CATEGORY = 200  # garde-fou ; l'arrêt réel se fait sur page vide
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "expat_dakar_listings_raw.csv"

FIELDNAMES = [
    "listing_id",
    "title",
    "category",
    "price_fcfa",
    "currency",
    "neighborhood",
    "city_region",
    "bedrooms",
    "surface_m2",
    "date_posted_raw",
    "url",
]


def fetch_page(category: str, page: int) -> str | None:
    url = f"{BASE_URL}/{category}"
    params = {"page": page} if page > 1 else {}
    response = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def extract_int_from_class(classes: list[str], prefix: str) -> int | None:
    for cls in classes:
        if cls.startswith(prefix):
            match = re.search(r"(\d+)$", cls)
            if match:
                return int(match.group(1))
    return None


def parse_listing_card(anchor) -> dict:
    data = anchor.attrs

    location_div = anchor.select_one(".listing-card__header__location")
    neighborhood, city_region = None, None
    if location_div:
        location_text = " ".join(location_div.get_text(separator="|", strip=True).split("|"))
        parts = [p.strip() for p in location_div.get_text(separator="\n", strip=True).split("\n") if p.strip()]
        if len(parts) >= 2:
            neighborhood, city_region = parts[0].rstrip(","), parts[1]
        elif len(parts) == 1:
            neighborhood = parts[0].rstrip(",")

    bedrooms_tag = anchor.select_one(".listing-card__header__tags__item--no-of-bedrooms")
    surface_tag = anchor.select_one(".listing-card__header__tags__item--square-metres")
    bedrooms = extract_int_from_class(bedrooms_tag.get("class", []), "listing-card__header__tags__item--no-of-bedrooms_") if bedrooms_tag else None
    surface = extract_int_from_class(surface_tag.get("class", []), "listing-card__header__tags__item--square-metres_") if surface_tag else None

    date_div = anchor.select_one(".listing-card__header__date")
    date_posted_raw = date_div.get_text(strip=True) if date_div else None

    return {
        "listing_id": data.get("data-t-listing_id"),
        "title": data.get("data-t-listing_title"),
        "category": data.get("data-t-listing_category_title"),
        "price_fcfa": data.get("data-t-listing_price"),
        "currency": data.get("data-t-listing_currency"),
        "neighborhood": neighborhood,
        "city_region": city_region,
        "bedrooms": bedrooms,
        "surface_m2": surface,
        "date_posted_raw": date_posted_raw,
        "url": data.get("href"),
    }


def scrape_category(category: str) -> list[dict]:
    listings = []
    seen_ids = set()

    for page in range(1, MAX_PAGES_PER_CATEGORY + 1):
        print(f"[{category}] page {page}...")
        html = fetch_page(category, page)
        if html is None:
            print(f"[{category}] page {page} : 404 (fin de pagination), arrêt.")
            break
        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select("a.listing-card__inner")

        if not anchors:
            print(f"[{category}] page {page} vide, arrêt.")
            break

        new_on_page = 0
        for anchor in anchors:
            listing_id = anchor.get("data-t-listing_id")
            if not listing_id or listing_id in seen_ids:
                continue
            seen_ids.add(listing_id)
            listings.append(parse_listing_card(anchor))
            new_on_page += 1

        if new_on_page == 0:
            print(f"[{category}] page {page} : plus aucune nouvelle annonce, arrêt.")
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    return listings


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_listings = []

    for category in CATEGORIES:
        category_listings = scrape_category(category)
        all_listings.extend(category_listings)

        # Sauvegarde incrémentale après chaque catégorie pour ne rien perdre en cas d'échec réseau.
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_listings)
        print(f"[{category}] {len(category_listings)} annonces (total cumulé: {len(all_listings)})")

    print(f"\nTotal: {len(all_listings)} annonces sauvegardées dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
