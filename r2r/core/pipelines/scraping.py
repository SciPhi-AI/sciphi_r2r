from typing import Iterator, Optional

from ..abstractions.document import DocumentPage
from ..utils import generate_run_id
from ..utils.logging import LoggingDatabaseConnection
from .pipeline import Pipeline


class ScraperPipeline(Pipeline):
    def __init__(
        self,
        logging_connection: Optional[LoggingDatabaseConnection] = None,
        *args,
        **kwargs,
    ):
        super().__init__(logging_connection=logging_connection, **kwargs)

    def initialize_pipeline(self, *args, **kwargs) -> None:
        self.pipeline_run_info = {
            "run_id": generate_run_id(),
            "type": "scraper",
        }

    def scrape_url(self, url: str) -> Iterator[DocumentPage]:
        """
        Scrape the given URL and return the raw data.
        """
        pass

    def run(
        self, document_id: str, url: str, **kwargs
    ) -> Iterator[DocumentPage]:
        """
        Run the scraping method for the given URL.
        Yields the processed BasicDocument objects.
        """
        self.initialize_pipeline()

        if not url:
            raise ValueError("No URL provided to scrape.")

        yield from self.scrape_url(url)

    def run_stream(
        self,
        query: str,
        context: str,
        completion: str,
        run_id: Optional[str],
        **kwargs,
    ):
        raise NotImplementedError(
            "Streaming mode not supported for `ScraperPipeline`."
        )
