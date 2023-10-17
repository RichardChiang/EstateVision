import re
from dataclasses import dataclass

import pytest

from src.scraping import GoogleMapsScraper, MapType


class FakeFileSystem:
    def __init__(self):
        self.files = {}

    def save_to_path(self, save_dir: str, filename: str, payload):
        filepath = f"{save_dir}/{filename}"
        self.files[filepath] = payload
        return filepath


@dataclass
class FakeResponse:
    status_code: int
    content: bytes


class FakeRequests:
    def __init__(self, responses=None):
        self.prev_requests = []
        self.prev_params = []
        self.responses = responses or []

    def set_responses(self, responses):
        self.responses = responses

    def get(self, url, params):
        self.prev_requests.append(url)
        self.prev_params.append(params)
        return self.responses.pop() if self.responses else None


@pytest.fixture
def filesystem():
    return FakeFileSystem()


@pytest.fixture
def requests():
    return FakeRequests()


def test_scraper_raises_without_valid_key():
    with pytest.raises(ValueError):
        _ = GoogleMapsScraper(None, None, None, FakeFileSystem())


def test_scraper_generate_filename():
    scraper = GoogleMapsScraper("API_KEY", None, None, None)
    filename = scraper._generate_filename(MapType.STREET, 41, -12)
    assert filename == "street_41_-12.png"
    filename = scraper._generate_filename(MapType.SATELLITE, 0, -12.3)
    assert filename == "satellite_0_-12.3.png"


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (41.01, -12.23),
        (-30.12345, 40.223),
        (35.1, -180.00),
    ],
)
def test_filename_generated_parseable_by_own_regex(lat, lon):
    scraper = GoogleMapsScraper("API_KEY", None, None, None)
    filename = scraper._generate_filename(MapType.STREET, lat, lon)
    regex = scraper._filename_parser_regex()
    assert [float(match) for match in re.search(regex, filename).groups()] == [lat, lon]


def test_scraper_makes_proper_get_request(requests, filesystem):
    requests.set_responses(
        [
            FakeResponse(status_code=200, content=b""),
            FakeResponse(status_code=200, content=b""),
        ]
    )
    scraper = GoogleMapsScraper("API_KEY", "data", requests, filesystem)
    scraper.scrape_map_image(MapType.STREET, 41, -12)
    scraper.scrape_map_image(MapType.SATELLITE, 0, -12.3)
    assert len(requests.prev_requests) == 2
    assert requests.prev_params == [
        scraper._create_params(MapType.STREET, 41, -12),
        scraper._create_params(MapType.SATELLITE, 0, -12.3),
    ]


def test_scraper_handles_errored_request(requests, filesystem):
    requests.set_responses([FakeResponse(status_code=404, content=b"")])
    scraper = GoogleMapsScraper("API_KEY", "data", requests, filesystem)
    filename = scraper.scrape_map_image(MapType.STREET, 41, -12)
    assert filename is None


def test_scraper_handles_successful_request(requests, filesystem):
    requests.set_responses([FakeResponse(status_code=200, content=b"")])
    scraper = GoogleMapsScraper("API_KEY", "data", requests, filesystem)
    filename = scraper.scrape_map_image(MapType.STREET, 41, -12)
    print(filename)
    assert filename is not None
