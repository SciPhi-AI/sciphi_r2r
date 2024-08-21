from typing import Any, Dict

import click
from cli.command_group import cli
from cli.utils.param_types import JSON
from cli.utils.timer import timer


@cli.command()
@click.pass_obj
def enrich_graph(client):
    """
    Perform graph enrichment over the entire graph.
    """
    with timer():
        response = client.restructure()

    click.echo(response)
