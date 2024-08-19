from typing import List, Optional

from fastapi import Depends, Form

from r2r.main.api.routes.base_router import BaseRouter
from r2r.main.engine import R2REngine


class RestructureRouter(BaseRouter):
    def __init__(self, engine: R2REngine):
        super().__init__(engine)
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/enrich_graph")
        @self.base_endpoint
        async def enrich_graph(
            auth_user=(
                Depends(self.engine.providers.auth.auth_wrapper)
                if self.engine.providers.auth
                else None
            ),
        ):
            """
            Perform graph enrichment, e.g. GraphRAG, over the ingested documents.

            Returns:
                Dict[str, Any]: Results of the graph enrichment process.
            """
            return await self.engine.aenrich_graph()
