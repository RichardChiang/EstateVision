import re
from collections import OrderedDict
from typing import List, Tuple

from src.scraping import GoogleMapsScraper, MapType

location = Tuple[float, float]

DEFAULT_MAX_CRAWL_DEPTH = 4
DEFAULT_MAX_REQUESTS = 10
DEFAULT_DELTA = 0.001533  # delta in coordinates
DEFAULT_PRECISION = 6  # number of coordinate decimals to keep


def get_crawl_locations(
    locations=List[location],
    max_requests=DEFAULT_MAX_REQUESTS,
    max_crawl_depth=DEFAULT_MAX_CRAWL_DEPTH,
    jump_distance=DEFAULT_DELTA,
    precision=DEFAULT_PRECISION,
) -> List[location]:
    """
    Crawls locations starting from a list of locations outwards in a grid.

    A naive implementation not accounting for shape distortion of Earth's surface and shape.
    """
    visited = OrderedDict()
    stack = [(round(lat, precision), round(lon, precision)) for lat, lon in locations]

    depth = 0
    requests_made = 0

    while stack and requests_made < max_requests and depth < max_crawl_depth:
        new_stack = []

        for location in stack:
            if location not in visited:
                visited[location] = True
                requests_made += 1

                lat, lon = location

                # crawl new locations from origin
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_lat = round(lat + dx * jump_distance, precision)
                    new_lon = round(lon + dy * jump_distance, precision)
                    new_stack.append((new_lat, new_lon))

        stack = new_stack
        depth += 1

    return list(visited.keys())[:max_requests]


def scrape_image_from_locations(
    locations: List[location],
    scraper: GoogleMapsScraper,
):
    """
    Scrapes street and satellite images from a list of locations using a GoogleMapsScraper.
    """
    output_files = []

    for location in locations:
        lat, lon = location
        # scrape both map types at location
        for map_type in [MapType.STREET, MapType.SATELLITE]:
            filename = scraper.scrape_map_image(
                map_type=map_type,
                lat=lat,
                lon=lon,
            )
            if filename:
                output_files.append(filename)


def get_existing_locations(
    scraper: GoogleMapsScraper,
    filenames: List[str],
) -> List[Tuple[float, float]]:
    """
    Returns a list of existing (lat, long) tuples from previously scraped images."""
    existing_locations = set()

    for filename in filenames:
        match = re.search(scraper._filename_parser_regex(), filename)
        if match:
            existing_locations.add(
                (
                    float(match.group(1)),
                    float(match.group(2)),
                )
            )

    return list(existing_locations)
