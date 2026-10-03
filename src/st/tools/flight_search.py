# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Flight search tool using SerpAPI (via google-serp package)."""

import json
import re
from typing import Optional
from langchain_core.tools import tool

# Cap on flight options returned to the agent (keeps the LLM context small)
MAX_FLIGHT_OPTIONS = 8

# Cap on return-flight options listed for the top outbound option of a round trip
MAX_RETURN_OPTIONS = 5


# One or more comma-separated 3-letter IATA codes (e.g. "JFK" or "JFK,EWR,LGA"),
# or a Google location id ("/m/..." or "/g/...")
_AIRPORT_ID = re.compile(r"[A-Z]{3}(,[A-Z]{3})*|/[mg]/\S+")


def _normalize_airport(code: str) -> Optional[str]:
    """Return the SerpAPI airport id for `code`, or None if it is not an airport code."""
    code = code.strip()
    if not code.startswith("/"):
        code = code.upper().replace(" ", "")
    return code if _AIRPORT_ID.fullmatch(code) else None


def _airport_time(airport: dict) -> str:
    return f"{airport.get('id')} {airport.get('time')}"


def _summarize_flight(option: dict) -> dict:
    """Reduce one SerpAPI flight option to the fields the agent needs."""
    return {
        "price_usd": option.get("price"),
        "total_duration_min": option.get("total_duration"),
        "stops": len(option.get("layovers", [])),
        "legs": [
            {
                "airline": leg.get("airline"),
                "flight_number": leg.get("flight_number"),
                "departure": _airport_time(leg.get("departure_airport", {})),
                "arrival": _airport_time(leg.get("arrival_airport", {})),
                "duration_min": leg.get("duration"),
                "travel_class": leg.get("travel_class"),
                "overnight": bool(leg.get("overnight")),
            }
            for leg in option.get("flights", [])
        ],
        "layovers": [
            {
                "airport": layover.get("id"),
                "duration_min": layover.get("duration"),
                "overnight": bool(layover.get("overnight")),
            }
            for layover in option.get("layovers", [])
        ],
        "carbon_vs_typical_pct": (option.get("carbon_emissions") or {}).get("difference_percent"),
    }


def summarize_flight_results(result: dict, max_options: int = MAX_FLIGHT_OPTIONS) -> dict:
    """Turn a raw SerpAPI google_flights response into a compact summary.

    Args:
        result: Raw SerpAPI response as a dict
        max_options: Maximum number of flight options to keep

    Returns:
        Compact dict with price insights and the top flight options
    """
    if "error" in result:
        return {"error": result["error"]}

    best = [_summarize_flight(f) for f in result.get("best_flights", [])][:max_options]
    other = [_summarize_flight(f) for f in result.get("other_flights", [])][:max_options - len(best)]
    if not best and not other:
        return {"message": "No flights found"}

    insights = result.get("price_insights", {})
    summary = {
        "price_insights": {
            "lowest_price_usd": insights.get("lowest_price"),
            "price_level": insights.get("price_level"),
            "typical_price_range_usd": insights.get("typical_price_range"),
        },
        "best_flights": best,
        "other_flights": other,
    }
    if result.get("search_parameters", {}).get("return_date"):
        summary["note"] = (
            "Round trip: each outbound option's price_usd is the lowest round-trip total "
            "starting with that outbound journey. return_flights_for_top_outbound lists "
            "return journeys for the first outbound option; their price_usd is the full "
            "round-trip total for that outbound + return pair."
        )
    return summary


def _top_outbound(result: dict) -> Optional[dict]:
    """First outbound option, in the same order the summary lists them."""
    options = result.get("best_flights", []) + result.get("other_flights", [])
    return options[0] if options else None


def fetch_return_flights(client, params: dict, outbound: dict, max_options: int = MAX_RETURN_OPTIONS) -> dict:
    """Look up return journeys for one outbound option of a round trip.

    SerpAPI lists only outbound journeys for a round trip; a follow-up search with the
    option's departure_token returns the matching return journeys (1 extra API credit).

    Args:
        client: serpapi.Client
        params: Parameters of the original round-trip search
        outbound: Raw outbound option containing a departure_token
        max_options: Maximum number of return options to keep

    Returns:
        Compact dict with the outbound flight numbers and their return options
    """
    outbound_flights = [leg.get("flight_number") for leg in outbound.get("flights", [])]
    try:
        result = client.search({**params, "departure_token": outbound["departure_token"]}).as_dict()
    except Exception as e:
        return {"outbound_flights": outbound_flights, "error": f"Could not fetch return flights: {str(e)}"}

    if "error" in result:
        return {"outbound_flights": outbound_flights, "error": result["error"]}

    options = result.get("best_flights", []) + result.get("other_flights", [])
    return {
        "outbound_flights": outbound_flights,
        "return_options": [_summarize_flight(o) for o in options[:max_options]],
    }


@tool(
    "search_flights",
    description="Search for flights between airports using SerpAPI Google Flights engine",
    return_direct=True
)
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    stops: Optional[int] = None,
) -> str:
    """Search for flights between airports.

    Args:
        departure_airport: Departure IATA airport code (e.g., 'JFK'), or comma-separated
            codes for a multi-airport city (e.g., 'JFK,EWR,LGA'); not a city name
        arrival_airport: Arrival IATA airport code (e.g., 'LIS'), same format
        outbound_date: Departure date (YYYY-MM-DD)
        return_date: Return date (YYYY-MM-DD); omit for a one-way trip
        adults: Number of adults
        children: Number of children
        stops: Number of stops (0=Any, 1=Nonstop, 2=1 stop or fewer, 3=2 stops or fewer)

    Returns:
        Compact JSON summary of the top flight options (plus return options for the top
        outbound option of a round trip)
    """
    import os
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    # Reject city names before spending a SerpAPI credit, so the agent can retry with codes
    departure_id = _normalize_airport(departure_airport)
    arrival_id = _normalize_airport(arrival_airport)
    invalid = [a for a, n in ((departure_airport, departure_id), (arrival_airport, arrival_id)) if n is None]
    if invalid:
        return json.dumps({"error": (
            f"Invalid airport code(s): {', '.join(repr(a) for a in invalid)}. Use 3-letter IATA "
            "airport codes, e.g. 'JFK' or 'LIS'; for a city with several airports pass them "
            "comma-separated, e.g. 'JFK,EWR,LGA'."
        )})

    try:
        from serpapi import Client
    except ImportError:
        raise ImportError(
            "Please install serpapi with `pip install serpapi`"
        )

    # Build parameters for SerpAPI
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": 1 if return_date else 2,  # 1=round trip (needs return_date), 2=one way
        "adults": adults,
        "hl": "en",
        "currency": "USD",
    }

    if return_date:
        params["return_date"] = return_date

    if children:
        params["children"] = children

    # SerpAPI takes the numeric code directly (0=any, 1=nonstop, 2=<=1 stop, 3=<=2 stops)
    if stops in (0, 1, 2, 3):
        params["stops"] = stops

    try:
        client = Client(api_key=api_key)
        result = client.search(params).as_dict()
        summary = summarize_flight_results(result)

        # Round trip: add the return journeys for the top outbound option
        top = _top_outbound(result)
        if return_date and "error" not in result and top and top.get("departure_token"):
            summary["return_flights_for_top_outbound"] = fetch_return_flights(client, params, top)

        return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        return f"Error searching flights: {str(e)}"
