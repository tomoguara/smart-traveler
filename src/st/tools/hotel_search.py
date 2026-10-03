# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Hotel search tool using SerpAPI (via google-serp package)."""

import json
from typing import Optional
from langchain_core.tools import tool

# Cap on hotels returned to the agent (keeps the LLM context small)
MAX_HOTEL_OPTIONS = 8

# Tool sort options -> SerpAPI google_hotels `sort_by` codes ("recommended" = Google's default order)
SORT_BY_CODES = {
    "recommended": None,
    "price_low": 3,
    "rating": 8,
    "most_reviewed": 13,
}


def _summarize_hotel(prop: dict) -> dict:
    """Reduce one SerpAPI property to the fields the agent needs."""
    return {
        "name": prop.get("name"),
        "type": prop.get("type"),
        "hotel_class": prop.get("extracted_hotel_class"),
        "overall_rating": prop.get("overall_rating"),
        "reviews": prop.get("reviews"),
        "location_rating": prop.get("location_rating"),
        "rate_per_night_usd": (prop.get("rate_per_night") or {}).get("extracted_lowest"),
        "total_rate_usd": (prop.get("total_rate") or {}).get("extracted_lowest"),
        "check_in_time": prop.get("check_in_time"),
        "check_out_time": prop.get("check_out_time"),
        "amenities": prop.get("amenities", [])[:8],
        "nearby": [place.get("name") for place in prop.get("nearby_places", [])][:3],
        "description": prop.get("description"),
        "link": prop.get("link"),
    }


def summarize_hotel_results(result: dict, max_options: int = MAX_HOTEL_OPTIONS) -> dict:
    """Turn a raw SerpAPI google_hotels response into a compact summary.

    Properties without a price for the requested dates are left out and counted.

    Args:
        result: Raw SerpAPI response as a dict
        max_options: Maximum number of properties to keep

    Returns:
        Compact dict with the top priced properties
    """
    if "error" in result:
        return {"error": result["error"]}

    properties = result.get("properties", [])
    priced = [p for p in properties if (p.get("rate_per_night") or {}).get("extracted_lowest") is not None]
    if not priced:
        return {"message": "No hotels with prices found for these dates"}

    return {
        "hotels": [_summarize_hotel(p) for p in priced[:max_options]],
        "unpriced_properties_omitted": len(properties) - len(priced),
    }


@tool(
    "search_hotels",
    description="Search for hotels using SerpAPI Google Hotels engine",
    return_direct=True
)
def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
    sort_by: str = "recommended",
) -> str:
    """Search for hotels in a location.

    Args:
        location: Location (e.g., 'New York', 'Paris')
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        adults: Number of adults
        children: Number of children
        rooms: Number of rooms
        sort_by: Sort options: 'recommended', 'price_low', 'rating', 'most_reviewed'

    Returns:
        Compact JSON summary of the top priced hotels
    """
    import os
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    try:
        from serpapi import Client
    except ImportError:
        raise ImportError(
            "Please install serpapi with `pip install serpapi`"
        )

    # Build parameters for SerpAPI
    params = {
        "engine": "google_hotels",
        "q": location,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "rooms": rooms,
        "hl": "en",
        "currency": "USD",
    }

    if children:
        params["children"] = children

    # SerpAPI uses numeric `sort_by` codes; "recommended" keeps Google's default order
    if SORT_BY_CODES.get(sort_by) is not None:
        params["sort_by"] = SORT_BY_CODES[sort_by]

    try:
        client = Client(api_key=api_key)
        result = client.search(params)
        return json.dumps(summarize_hotel_results(result.as_dict()), ensure_ascii=False)
    except Exception as e:
        return f"Error searching hotels: {str(e)}"
