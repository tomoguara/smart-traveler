"""Unit tests for the flight, hotel and web search tools (no network calls)."""

import json

import langchain_tavily
import pytest
import serpapi
from langchain_core.messages import AIMessage, ToolMessage

from st.agents import create_search_agent
from st.tools import search_flights, search_hotels, web_search

from conftest import StubLLM


def _flight_option(price, airline="TAP Air Portugal", layover=None):
    legs = [{
        "departure_airport": {"name": "John F. Kennedy International Airport", "id": "JFK", "time": "2026-11-10 22:05"},
        "arrival_airport": {"name": "Humberto Delgado Airport", "id": "LIS", "time": "2026-11-11 10:00"},
        "duration": 415, "airplane": "Airbus A330", "airline": airline, "airline_logo": "https://logo.png",
        "travel_class": "Economy", "flight_number": "TP 210", "legroom": "31 in",
        "extensions": ["Wi-Fi for a fee", "Carbon emissions estimate: 331 kg"], "overnight": True,
    }]
    return {
        "flights": legs,
        "layovers": [layover] if layover else [],
        "total_duration": 415,
        "carbon_emissions": {"this_flight": 331000, "typical_for_this_route": 340000, "difference_percent": -3},
        "price": price,
        "type": "Round trip",
        "airline_logo": "https://logo.png",
        "departure_token": "x" * 500,
    }


FLIGHTS_RESPONSE = {
    "search_metadata": {"id": "abc", "raw_html_file": "https://serpapi.com/raw.html"},
    "search_parameters": {"engine": "google_flights", "return_date": "2026-11-14"},
    "best_flights": [_flight_option(695), _flight_option(720, layover={"duration": 220, "name": "Madrid", "id": "MAD"})],
    "other_flights": [_flight_option(800 + i) for i in range(10)],
    "price_insights": {"lowest_price": 695, "price_level": "high", "typical_price_range": [420, 690],
                       "price_history": [[1785643200, 604]] * 60},
    "airports": [{"departure": [], "arrival": []}],
}


def _property(name, nightly=None):
    prop = {
        "type": "hotel", "name": name, "description": "Boutique hotel in Baixa.",
        "link": "https://hotel.example", "property_token": "t" * 100,
        "gps_coordinates": {"latitude": 38.7, "longitude": -9.1},
        "check_in_time": "3:00 PM", "check_out_time": "12:00 PM",
        "nearby_places": [{"name": "Rossio", "transportations": []}, {"name": "Santa Justa Lift"},
                          {"name": "Chiado"}, {"name": "Alfama"}],
        "hotel_class": "4-star hotel", "extracted_hotel_class": 4,
        "images": [{"thumbnail": "https://img", "original_image": "https://img"}] * 20,
        "overall_rating": 4.8, "reviews": 1278, "location_rating": 4.9,
        "reviews_breakdown": [{"name": "Service", "positive": 100}] * 10,
        "amenities": [f"Amenity {i}" for i in range(15)],
    }
    if nightly is not None:
        prop["rate_per_night"] = {"lowest": f"${nightly}", "extracted_lowest": nightly}
        prop["total_rate"] = {"lowest": f"${nightly * 4}", "extracted_lowest": nightly * 4}
    return prop


HOTELS_RESPONSE = {
    "search_metadata": {"id": "abc"},
    "search_parameters": {"engine": "google_hotels"},
    "ads": [{"name": "Ad hotel"}] * 5,
    "properties": [_property("Sold Out Palace")] + [_property(f"Hotel {i}", 100 + i) for i in range(12)],
}


class FakeResults:
    def __init__(self, data):
        self.data = data

    def as_dict(self):
        return self.data


@pytest.fixture
def fake_serpapi(monkeypatch):
    """Replace serpapi.Client; returns the list of params each search received.

    Searches get the given responses in order (the last one repeats).
    """
    calls = []

    def install(*responses):
        class FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key

            def search(self, params):
                calls.append(params)
                return FakeResults(responses[min(len(calls), len(responses)) - 1])

        monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
        monkeypatch.setattr(serpapi, "Client", FakeClient)
        return calls

    return install


# --- search_flights ---------------------------------------------------------

RETURN_RESPONSE = {
    "search_metadata": {"id": "def"},
    "other_flights": [{**_flight_option(695 + i), "booking_token": "b" * 500} for i in range(8)],
}


def test_flights_round_trip_params_and_compact_summary(fake_serpapi):
    calls = fake_serpapi(FLIGHTS_RESPONSE, RETURN_RESPONSE)

    out = search_flights.invoke({"departure_airport": "jfk", "arrival_airport": "lis",
                                 "outbound_date": "2026-11-10", "return_date": "2026-11-14", "stops": 1})

    params = calls[0]
    assert params["type"] == 1
    assert params["return_date"] == "2026-11-14"
    assert params["stops"] == 1  # numeric SerpAPI code, not "nonstop"
    assert (params["departure_id"], params["arrival_id"]) == ("JFK", "LIS")

    summary = json.loads(out)
    assert summary["price_insights"] == {"lowest_price_usd": 695, "price_level": "high",
                                         "typical_price_range_usd": [420, 690]}
    assert len(summary["best_flights"]) + len(summary["other_flights"]) == 8
    second = summary["best_flights"][1]
    assert second["price_usd"] == 720
    assert second["stops"] == 1
    assert second["layovers"] == [{"airport": "MAD", "duration_min": 220, "overnight": False}]
    assert second["legs"][0]["departure"] == "JFK 2026-11-10 22:05"
    assert "Round trip" in summary["note"]
    # Bulky raw fields never reach the agent
    assert "departure_token" not in out and "booking_token" not in out
    assert "price_history" not in out and "\x1b[" not in out
    assert len(out) < len(json.dumps(FLIGHTS_RESPONSE)) / 2


def test_round_trip_fetches_return_flights_for_top_outbound(fake_serpapi):
    calls = fake_serpapi(FLIGHTS_RESPONSE, RETURN_RESPONSE)

    out = search_flights.invoke({"departure_airport": "JFK", "arrival_airport": "LIS",
                                 "outbound_date": "2026-11-10", "return_date": "2026-11-14"})

    assert len(calls) == 2  # exactly one extra lookup
    follow_up = calls[1]
    assert follow_up["departure_token"] == FLIGHTS_RESPONSE["best_flights"][0]["departure_token"]
    assert {k: v for k, v in follow_up.items() if k != "departure_token"} == calls[0]

    returns = json.loads(out)["return_flights_for_top_outbound"]
    assert returns["outbound_flights"] == ["TP 210"]
    assert len(returns["return_options"]) == 5
    assert returns["return_options"][0]["price_usd"] == 695


def test_round_trip_return_lookup_error_keeps_outbound(fake_serpapi):
    fake_serpapi(FLIGHTS_RESPONSE, {"error": "Invalid departure_token"})

    out = search_flights.invoke({"departure_airport": "JFK", "arrival_airport": "LIS",
                                 "outbound_date": "2026-11-10", "return_date": "2026-11-14"})

    summary = json.loads(out)
    assert summary["return_flights_for_top_outbound"] == {"outbound_flights": ["TP 210"],
                                                          "error": "Invalid departure_token"}
    assert len(summary["best_flights"]) == 2


def test_flights_one_way_uses_type_2(fake_serpapi):
    calls = fake_serpapi({**FLIGHTS_RESPONSE, "search_parameters": {}})

    out = search_flights.invoke({"departure_airport": "JFK", "arrival_airport": "LIS", "outbound_date": "2026-11-10"})

    assert len(calls) == 1  # no return-flight lookup for one-way trips
    assert calls[0]["type"] == 2
    assert "return_date" not in calls[0]
    assert "stops" not in calls[0]
    assert "note" not in json.loads(out)


def test_flights_api_error_and_no_results(fake_serpapi):
    fake_serpapi({"error": "Google Flights hasn't returned any results for this query."})
    out = search_flights.invoke({"departure_airport": "JFK", "arrival_airport": "XXX", "outbound_date": "2026-11-10"})
    assert json.loads(out) == {"error": "Google Flights hasn't returned any results for this query."}


def test_flights_no_options(fake_serpapi):
    fake_serpapi({"search_parameters": {}, "best_flights": [], "other_flights": []})
    out = search_flights.invoke({"departure_airport": "JFK", "arrival_airport": "LIS", "outbound_date": "2026-11-10"})
    assert json.loads(out) == {"message": "No flights found"}


def test_flights_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
        search_flights.invoke({"departure_airport": "JFK", "arrival_airport": "LIS", "outbound_date": "2026-11-10"})


# --- search_hotels ----------------------------------------------------------

@pytest.mark.parametrize("sort_by, code", [("price_low", 3), ("rating", 8), ("most_reviewed", 13)])
def test_hotels_sort_by_codes(fake_serpapi, sort_by, code):
    calls = fake_serpapi(HOTELS_RESPONSE)

    search_hotels.invoke({"location": "Lisbon", "check_in_date": "2026-11-10",
                          "check_out_date": "2026-11-14", "sort_by": sort_by})

    assert calls[0]["sort_by"] == code
    assert "sort" not in calls[0]


def test_hotels_recommended_keeps_default_order(fake_serpapi):
    calls = fake_serpapi(HOTELS_RESPONSE)
    search_hotels.invoke({"location": "Lisbon", "check_in_date": "2026-11-10", "check_out_date": "2026-11-14"})
    assert "sort_by" not in calls[0]


def test_hotels_compact_summary_skips_unpriced(fake_serpapi):
    fake_serpapi(HOTELS_RESPONSE)

    out = search_hotels.invoke({"location": "Lisbon", "check_in_date": "2026-11-10", "check_out_date": "2026-11-14"})

    summary = json.loads(out)
    assert len(summary["hotels"]) == 8
    assert summary["unpriced_properties_omitted"] == 1
    first = summary["hotels"][0]
    assert first["name"] == "Hotel 0"
    assert (first["rate_per_night_usd"], first["total_rate_usd"]) == (100, 400)
    assert first["hotel_class"] == 4
    assert len(first["amenities"]) == 8
    assert first["nearby"] == ["Rossio", "Santa Justa Lift", "Chiado"]
    assert "images" not in out and "property_token" not in out and "Ad hotel" not in out
    assert len(out) < len(json.dumps(HOTELS_RESPONSE)) / 2


def test_hotels_none_priced(fake_serpapi):
    fake_serpapi({"properties": [_property("Sold Out Palace")]})
    out = search_hotels.invoke({"location": "Lisbon", "check_in_date": "2026-11-10", "check_out_date": "2026-11-14"})
    assert json.loads(out) == {"message": "No hotels with prices found for these dates"}


# --- web_search -------------------------------------------------------------

def test_web_search_requests_snippets_and_compacts(monkeypatch):
    captured = {}

    class FakeTavilySearch:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def invoke(self, args):
            captured["args"] = args
            return {"query": args["query"], "answer": "Spring and autumn.", "images": [],
                    "results": [{"title": "Lisbon guide", "url": "https://x", "content": "y" * 5000,
                                 "score": 0.9, "raw_content": "z" * 100000}]}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(langchain_tavily, "TavilySearch", FakeTavilySearch)

    out = web_search.invoke({"query": "best time to visit Lisbon", "topic": "news", "time_range": "month"})

    assert captured["init"]["include_raw_content"] is False
    assert captured["args"] == {"query": "best time to visit Lisbon", "search_depth": "basic",
                                "topic": "news", "time_range": "month"}
    summary = json.loads(out)
    assert summary["answer"] == "Spring and autumn."
    assert summary["results"][0]["content"] == "y" * 1500
    assert "raw_content" not in out


# --- search agent tool node -------------------------------------------------

def test_search_tool_node_answers_unknown_tool_calls():
    _, search_tool_node = create_search_agent(StubLLM())
    call = AIMessage(content="", tool_calls=[{"name": "book_flight", "args": {}, "id": "call-3"}])

    [message] = search_tool_node({"search_messages": [call]})["search_messages"]

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-3"
    assert message.content == "Error: unknown tool book_flight"


@pytest.mark.parametrize("departure, expected", [("jfk", "JFK"), ("JFK, EWR,LGA", "JFK,EWR,LGA"), ("/m/02_286", "/m/02_286")])
def test_flights_accepts_airport_codes(fake_serpapi, departure, expected):
    calls = fake_serpapi({"search_parameters": {}, "best_flights": [_flight_option(500)]})
    search_flights.invoke({"departure_airport": departure, "arrival_airport": "LIS", "outbound_date": "2026-10-09"})
    assert calls[0]["departure_id"] == expected


def test_flights_rejects_city_names_without_calling_serpapi(fake_serpapi):
    calls = fake_serpapi(FLIGHTS_RESPONSE)

    out = search_flights.invoke({"departure_airport": "New York", "arrival_airport": "Lisbon",
                                 "outbound_date": "2026-10-09"})

    assert calls == []
    error = json.loads(out)["error"]
    assert "'New York'" in error and "'Lisbon'" in error and "JFK,EWR,LGA" in error
