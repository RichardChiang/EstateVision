import pytest

from src.crawling import (
    GoogleMapsScraper,
    MapType,
    get_crawl_locations,
    get_existing_locations,
    scrape_image_from_locations,
)


class TestGetCrawlLocations:
    def test_single_location(self):
        locations = [(10.12345, 20.54321)]
        expected_result = [(10.12345, 20.54321)]
        max_requests = 1

        result = get_crawl_locations(locations, max_requests)
        assert result == expected_result

    def test_multiple_locations(self):
        locations = [(10.12345, 20.54321), (30.98765, 40.12345)]
        max_crawl_depth = 1
        expected_result = [(10.12345, 20.54321), (30.98765, 40.12345)]

        result = get_crawl_locations(locations, max_crawl_depth=max_crawl_depth)
        assert result == expected_result

    def test_empty_locations(self):
        assert get_crawl_locations(locations=[]) == []

    def test_max_requests_reached(self):
        locations = [(10.12345, 20.54321), (30.98765, 40.12345)]
        max_requests = 5

        result = get_crawl_locations(
            locations,
            max_requests=max_requests,
            max_crawl_depth=10,
        )
        assert len(result) == max_requests

    def test_maintains_precision(self):
        def is_precise(num, precision):
            tolerance = 0.001
            return abs(num - round(num, precision)) <= tolerance

        locations = [(10.12345, 20.54999), (30.9, 40.18901)]
        precision = 3

        result = get_crawl_locations(locations, precision=precision)

        for lat, lon in result:
            assert is_precise(lat, precision) and is_precise(lon, precision)

    @pytest.mark.parametrize(
        ("max_crawl_depth", "expected_result"),
        [(1, 1), (2, 5), (3, 13)],
    )
    def test_max_crawl_depth_reached(self, max_crawl_depth, expected_result):
        locations = [(10.12345, 20.54321)]

        result = get_crawl_locations(
            locations,
            max_requests=1000,
            max_crawl_depth=max_crawl_depth,
        )
        assert len(result) == expected_result

    def test_crawl_locations_returns_unique_locations(self):
        locations = [(10.12345, 20.54321), (10.12345, 20.54321)]

        result = get_crawl_locations(locations)
        assert len(set(result)) == len(result)

    def test_crawl_locations_return_sorted_by_crawl_time(self):
        def distance(coord1, coord2):
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

        start_lat, start_lon = (10.12345, 20.54321)
        max_requests = 13
        max_crawl_depth = 3

        result = get_crawl_locations(
            locations=[(start_lat, start_lon)],
            max_requests=max_requests,
            max_crawl_depth=max_crawl_depth,
        )
        assert distance(result[0], (start_lat, start_lon)) == 0
        assert distance(result[0], result[1]) < distance(result[0], result[-1])


class FakeScraper:
    def __init__(self):
        self.scrape_requests = []

    def scrape_map_image(self, map_type, lat, lon, invalid=False):
        self.scrape_requests.append((map_type, lat, lon))
        return None if invalid else "image.png"

    def _filename_parser_regex(self):
        return GoogleMapsScraper._filename_parser_regex()


@pytest.fixture
def scraper():
    return FakeScraper()


class TestScrapeImageFromLocations:
    def test_scrape_image_gets_street_and_satellite_maps(self, scraper):
        locations = [(10.12345, 20.54321)]
        scrape_image_from_locations(locations, scraper)
        assert scraper.scrape_requests == [
            (MapType.STREET, 10.12345, 20.54321),
            (MapType.SATELLITE, 10.12345, 20.54321),
        ]

    def test_scrape_image_from_multiple_locations(self, scraper):
        locations = [(10.12345, 20.54321), (30.98765, 40.12345)]
        scrape_image_from_locations(locations, scraper)
        assert len(scraper.scrape_requests) == len(locations) * 2

    def test_scrape_image_from_empty_locations(self, scraper):
        locations = []
        scrape_image_from_locations(locations, scraper)
        assert scraper.scrape_requests == []


class TestGetExistingLocations:
    def test_parses_positive_and_negative_coords(self, scraper):
        filenames = [
            "street_10.12345_20.54321.png",
            "satellite_10.12345_20.54321.png",
            "street_-30.98765_40.12345.png",
            "street_29.98765_-33.98211.png",
        ]

        assert sorted(get_existing_locations(scraper, filenames)) == [
            (-30.98765, 40.12345),
            (10.12345, 20.54321),
            (29.98765, -33.98211),
        ]

    def test_returns_unique_locations(self, scraper):
        filenames = [
            "street_10.12345_20.54321.png",
            "satellite_10.12345_20.54321.png",
            "street_-30.98765_40.12345.png",
            "satellite_-30.98765_40.12345.png",
        ]

        assert sorted(get_existing_locations(scraper, filenames)) == [
            (-30.98765, 40.12345),
            (10.12345, 20.54321),
        ]

    def test_skips_improperly_formatted_filenames(self, scraper):
        filenames = [
            "street_10_20.png",
            "satellite_10_20_30.png",
            "street_-30_40.0.png",
        ]

        assert get_existing_locations(scraper, filenames) == []
