"""Command-line interface for deep-semantic-search."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import click

from .config import BGE_M3_MODEL_DEFAULT, IMAGE_EXTENSIONS, SIGLIP_MODEL_DEFAULT


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
@click.option("--model", default=SIGLIP_MODEL_DEFAULT, show_default=True, help="SigLIP model name.")
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

    rows = [{"rank": r["rank"], "path": r["path"], "score": f"{r['score']:.4f}"} for r in results]
    _output(rows, ["rank", "path", "score"], fmt)


# ---------------------------------------------------------------------------
# text-search
# ---------------------------------------------------------------------------
@cli.command("text-search")
@click.option("--folder", "-f", required=True, help="Folder containing text/html files.")
@click.argument("query")
@click.option("--top", "-n", default=10, show_default=True, help="Number of results.")
@click.option("--model", default=BGE_M3_MODEL_DEFAULT, show_default=True, help="BGE-M3 model name.")
@click.option("--reindex", is_flag=True, help="Force re-embedding.")
@click.option("--rerank", is_flag=True, help="Rerank results with cross-encoder.")
@click.option("--hybrid/--no-hybrid", default=True, show_default=True, help="Use hybrid dense+sparse search.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def text_search(
    folder: str, query: str, top: int, model: str, reindex: bool, rerank: bool, hybrid: bool, fmt: str
) -> None:
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
    results = searcher.find_similar(query, top_n=top, rerank=rerank, hybrid=hybrid)

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
@click.option("--clusters", "-k", default=None, type=int, help="Number of clusters (omit for HDBSCAN auto).")
@click.option("--min-cluster-size", default=5, show_default=True, help="Minimum cluster size for HDBSCAN.")
@click.option("--caption", is_flag=True, help="Use Florence-2 captioning for topic labels.")
@click.option("--save-dir", type=click.Path(), default=None, help="Save clustered images to this directory.")
@click.option("--model", default=SIGLIP_MODEL_DEFAULT, show_default=True, help="SigLIP model name.")
@click.option("--reindex", is_flag=True, help="Force re-indexing.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def image_cluster(
    folder: tuple[str, ...],
    clusters: int | None,
    min_cluster_size: int,
    caption: bool,
    save_dir: str | None,
    model: str,
    reindex: bool,
    fmt: str,
) -> None:
    """Cluster images using KMeans or HDBSCAN on SigLIP embeddings."""
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
    method = f"KMeans(k={clusters})" if clusters else "HDBSCAN"
    click.echo(f"Clustering with {method}...")
    result_df = clusterer.cluster(
        n_clusters=clusters, captioner=captioner, min_cluster_size=min_cluster_size
    )

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
@click.option("--chunk-size", default=1500, show_default=True, help="Text chunk size.")
@click.option("--rerank", is_flag=True, help="Rerank retrieved chunks with cross-encoder.")
@click.option(
    "--semantic-chunking/--no-semantic-chunking", default=True, show_default=True,
    help="Use semantic chunking.",
)
def ask(
    folder: str,
    question: str,
    model: str | None,
    chunk_size: int,
    rerank: bool,
    semantic_chunking: bool,
) -> None:
    """Ask a question over text documents using RAG."""
    from .rag import RAG
    from .text_loader import LoadTextData

    loader = LoadTextData()
    corpus = loader.from_folder(folder)
    if not corpus:
        click.echo("No text files found in the specified folder.", err=True)
        raise SystemExit(1)

    click.echo(f"Loaded {len(corpus)} documents. Running RAG...")
    rag = RAG(model_name=model, rerank=rerank)
    answer = rag.ask(
        text_data=list(corpus.values()),
        question=question,
        chunk_size=chunk_size,
        semantic_chunking=semantic_chunking,
    )
    click.echo(f"\n{answer}")


# ---------------------------------------------------------------------------
# unified-search
# ---------------------------------------------------------------------------
@cli.command("unified-search")
@click.option("--image-folder", default=None, help="Folder containing images.")
@click.option("--text-folder", default=None, help="Folder containing text files.")
@click.option("--query", "-q", required=True, help="Text query.")
@click.option("--top", "-n", default=10, show_default=True, help="Number of results.")
@click.option(
    "--filter", "modality_filter", type=click.Choice(["all", "image", "text"]),
    default="all", show_default=True,
)
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def unified_search(
    image_folder: str | None,
    text_folder: str | None,
    query: str,
    top: int,
    modality_filter: str,
    fmt: str,
) -> None:
    """Search across images and text in a unified embedding space."""
    from .unified_search import UnifiedIndexer, UnifiedSearcher

    if not image_folder and not text_folder:
        click.echo("Provide at least --image-folder or --text-folder.", err=True)
        raise SystemExit(1)

    indexer = UnifiedIndexer()

    if image_folder:
        from .image_loader import LoadImageData

        loader = LoadImageData()
        images = loader.from_folder([image_folder])
        if images:
            click.echo(f"Adding {len(images)} images...")
            indexer.add_images(images)

    if text_folder:
        from .text_loader import LoadTextData

        loader = LoadTextData()
        corpus = loader.from_folder(text_folder)
        if corpus:
            click.echo(f"Adding {len(corpus)} text documents...")
            indexer.add_texts(list(corpus.values()), labels=list(corpus.keys()))

    if not indexer._entries:
        click.echo("No data found to index.", err=True)
        raise SystemExit(1)

    indexer.build_index()
    searcher = UnifiedSearcher(indexer)

    filt = None if modality_filter == "all" else modality_filter
    results = searcher.search(query, n=top, modality_filter=filt)

    rows = [
        {"rank": r["rank"], "type": r["type"], "source": r["source"], "score": f"{r['score']:.4f}"}
        for r in results
    ]
    _output(rows, ["rank", "type", "source", "score"], fmt)


# ---------------------------------------------------------------------------
# find-duplicates
# ---------------------------------------------------------------------------
@cli.command("find-duplicates")
@click.option("--folder", "-f", required=True, multiple=True, help="Folder(s) containing images.")
@click.option("--threshold", "-t", default=0.95, show_default=True, help="Similarity threshold.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", show_default=True)
def find_duplicates(folder: tuple[str, ...], threshold: float, fmt: str) -> None:
    """Find near-duplicate images in the given folder(s)."""
    from .image_indexer import ImageIndexer
    from .image_loader import LoadImageData
    from .image_searcher import ImageSearcher

    loader = LoadImageData()
    images = loader.from_folder(list(folder))
    if not images:
        click.echo("No images found.", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(images)} images. Indexing...")
    indexer = ImageIndexer(images)
    indexer.run_index()

    searcher = ImageSearcher(indexer)
    duplicates = searcher.find_duplicates(threshold=threshold)

    if not duplicates:
        click.echo("No duplicates found.")
        return

    rows = [
        {"path1": p1, "path2": p2, "similarity": f"{sim:.4f}"}
        for p1, p2, sim in duplicates
    ]
    _output(rows, ["path1", "path2", "similarity"], fmt)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
