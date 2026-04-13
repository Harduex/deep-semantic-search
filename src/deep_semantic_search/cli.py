"""Command-line interface for deep-semantic-search."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import click

from .config import CLIP_MODEL_DEFAULT, IMAGE_EXTENSIONS, TEXT_MODEL_DEFAULT


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _is_image_path(query: str) -> bool:
    return Path(query).suffix.lower() in IMAGE_EXTENSIONS and Path(query).exists()


def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        click.echo("No results found.")
        return
    widths = {col: max(len(col), *(len(str(r.get(col, ""))) for r in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    click.echo(header)
    click.echo("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        click.echo("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def _output(rows: list[dict], columns: list[str], fmt: str) -> None:
    if fmt == "json":
        click.echo(json.dumps(rows, indent=2))
    elif fmt == "csv":
        if not rows:
            return
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    else:
        _print_table(rows, columns)


@click.group()
@click.version_option(package_name="deep-semantic-search")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """dss — Deep Semantic Search CLI.

    Semantic search, clustering, and RAG for text and image data.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    _setup_logging(verbose, quiet)


# ---------------------------------------------------------------------------
# image-search
# ---------------------------------------------------------------------------
@cli.command("image-search")
@click.option("--folder", "-f", required=True, multiple=True, help="Folder(s) containing images.")
@click.option("--query", "-q", required=True, help="Text query or path to an image file.")
@click.option("--top", "-n", default=10, show_default=True, help="Number of results.")
@click.option("--model", default=CLIP_MODEL_DEFAULT, show_default=True, help="CLIP model name.")
@click.option("--reindex", is_flag=True, help="Force re-indexing.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def image_search(folder: tuple[str, ...], query: str, top: int, model: str, reindex: bool, fmt: str) -> None:
    """Search images by text or by image similarity."""
    from .image_indexer import ImageIndexer
    from .image_loader import LoadImageData
    from .image_searcher import ImageSearcher

    loader = LoadImageData()
    images = loader.from_folder(list(folder))
    if not images:
        click.echo("No images found in the specified folder(s).", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(images)} images. Indexing...")
    indexer = ImageIndexer(images, model_name=model)
    indexer.run_index(reindex=reindex)

    searcher = ImageSearcher(indexer)
    if _is_image_path(query):
        click.echo(f"Searching by image: {query}")
        results = searcher.search_by_image(query, n=top)
    else:
        click.echo(f"Searching by text: {query}")
        results = searcher.search_by_text(query, n=top)

    rows = [{"rank": i + 1, "path": path, "score": f"{score:.4f}"} for i, (path, score) in enumerate(results.items())]
    _output(rows, ["rank", "path", "score"], fmt)


# ---------------------------------------------------------------------------
# text-search
# ---------------------------------------------------------------------------
@cli.command("text-search")
@click.option("--folder", "-f", required=True, help="Folder containing text/html files.")
@click.argument("query")
@click.option("--top", "-n", default=10, show_default=True, help="Number of results.")
@click.option("--model", default=TEXT_MODEL_DEFAULT, show_default=True, help="Sentence Transformer model.")
@click.option("--reindex", is_flag=True, help="Force re-embedding.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def text_search(folder: str, query: str, top: int, model: str, reindex: bool, fmt: str) -> None:
    """Search text documents by semantic similarity."""
    from .text_embedder import TextEmbedder
    from .text_loader import LoadTextData
    from .text_searcher import TextSearch

    loader = LoadTextData()
    corpus = loader.from_folder(folder)
    if not corpus:
        click.echo("No text files found in the specified folder.", err=True)
        raise SystemExit(1)

    click.echo(f"Loaded {len(corpus)} documents. Embedding...")
    embedder = TextEmbedder(model_name=model)
    embedder.embed(corpus, reindex=reindex)

    searcher = TextSearch(embedder)
    results = searcher.find_similar(query, top_n=top)

    rows = [
        {"rank": i + 1, "score": f"{r['score']:.4f}", "path": r["path"], "text": r["text"][:120]}
        for i, r in enumerate(results)
    ]
    _output(rows, ["rank", "score", "path", "text"], fmt)


# ---------------------------------------------------------------------------
# image-cluster
# ---------------------------------------------------------------------------
@cli.command("image-cluster")
@click.option("--folder", "-f", required=True, multiple=True, help="Folder(s) containing images.")
@click.option("--clusters", "-k", required=True, type=int, help="Number of clusters.")
@click.option("--caption", is_flag=True, help="Use BLIP captioning for topic labels.")
@click.option("--save-dir", type=click.Path(), default=None, help="Save clustered images to this directory.")
@click.option("--model", default=CLIP_MODEL_DEFAULT, show_default=True, help="CLIP model name.")
@click.option("--reindex", is_flag=True, help="Force re-indexing.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def image_cluster(
    folder: tuple[str, ...], clusters: int, caption: bool, save_dir: str | None, model: str, reindex: bool, fmt: str
) -> None:
    """Cluster images using KMeans on CLIP embeddings."""
    from .image_clusterer import ImageClusterer
    from .image_indexer import ImageIndexer
    from .image_loader import LoadImageData

    loader = LoadImageData()
    images = loader.from_folder(list(folder))
    if not images:
        click.echo("No images found in the specified folder(s).", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(images)} images. Indexing...")
    indexer = ImageIndexer(images, model_name=model)
    indexer.run_index(reindex=reindex)

    captioner = None
    if caption:
        from .image_captioner import ImageCaptioner

        captioner = ImageCaptioner()

    clusterer = ImageClusterer(indexer)
    click.echo(f"Clustering into {clusters} groups...")
    result_df = clusterer.cluster(n_clusters=clusters, captioner=captioner)

    rows = [
        {"cluster": int(row["cluster"]), "topic": row.get("topic", ""), "path": row["images_paths"]}
        for _, row in result_df.iterrows()
    ]
    _output(rows, ["cluster", "topic", "path"], fmt)

    if save_dir:
        clusterer.save_clusters(save_dir)
        click.echo(f"Clustered images saved to {save_dir}")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--folder", "-f", required=True, help="Folder containing text/html files.")
@click.argument("question")
@click.option("--model", default=None, help="Ollama model name (default: env OLLAMA_LLM_MODEL or gemma4:e4b).")
@click.option("--chunk-size", default=1500, show_default=True, help="Text chunk size for splitting.")
@click.option("--chunk-overlap", default=100, show_default=True, help="Overlap between chunks.")
def ask(folder: str, question: str, model: str | None, chunk_size: int, chunk_overlap: int) -> None:
    """Ask a question over text documents using RAG."""
    from .rag import ask_question
    from .text_loader import LoadTextData

    loader = LoadTextData()
    corpus = loader.from_folder(folder)
    if not corpus:
        click.echo("No text files found in the specified folder.", err=True)
        raise SystemExit(1)

    click.echo(f"Loaded {len(corpus)} documents. Running RAG...")
    answer = ask_question(
        text_data=list(corpus.values()),
        question=question,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model,
    )
    click.echo(f"\n{answer}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
