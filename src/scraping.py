import enum
import os
from typing import Optional


class MapType(enum.Enum):
    STREET = "street"
    SATELLITE = "satellite"


class FileSystem:
    def save_to_path(self, save_dir: str, filename: str, payload) -> Optional[str]:
        """
        Saves a request payload to a file in the specified directory.

        Makes the directory if it doesn't exist.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, filename)
            with open(path, "wb") as f:
                f.write(payload)
            return path
        except OSError:
            print(f"Unable to create directory: {save_dir}")
            return None


class GoogleMapsScraper:
    """
    A class for scraping static map images from Google Maps API.

    This class allows you to retrieve static map images from Google Maps at specific coordinates and save them to a specified directory.
    """

    def __init__(self, api_key: str, save_dir: str, requests, filesystem):
        if api_key is None:
            raise ValueError("Google Maps API key is missing.")

        self.api_key = api_key
        self.save_dir = save_dir
        self.requests = requests
        self.filesystem = filesystem

    def _generate_filename(self, map_type: MapType, lat: float, lon: float):
        """
        Generate a unique filename for the map image.
        """
        return f"{map_type.value}_{lat}_{lon}.png"

    def _create_params(self, map_type: MapType, lat: float, lon: float):
        """
        Creates the default parameters for the Google Maps API request.
        """
        return {
            "center": f"{lat},{lon}",
            "zoom": 19,
            "size": "1280x1280",
            "scale": 2,
            "style": "feature:road|element:all|visibility:off",  # Hide roads
            "maptype": map_type.value,
            "key": self.api_key,
        }

    def get_map_image(
        self,
        map_type: MapType,
        lat: float,
        lon: float,
    ) -> Optional[str]:
        """
        Scrape a static map from Google Maps at the given coordinates, and save it to the scraper's save directory.

        Returns:
            The filename of the saved map image, or None if an error occurred.
        """
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = self._create_params(map_type, lat, lon)
        response = self.requests.get(url, params=params)

        if response.status_code == 200:
            filename = self._generate_filename(map_type, lat, lon)
            return self.filesystem.save_to_path(
                self.save_dir,
                filename,
                response.content,
            )
        else:
            print(
                f"Map Image request at ({lat}, {lon}) errored with status code: {response.status_code}"
            )
            return None
