import gradio as gr
from deep_semantic_search import (
    LoadTextData,
    TextEmbedder,
    TextSearch,
    LoadImageData,
    ImageSearch,
)
import os
from werkzeug.utils import secure_filename

# Global variable to store the search engine setup
image_search_setup = None


def index(folder_path, image_count=None, reindex=True):
    global image_search_setup

    try:
        image_count = int(image_count) if image_count else None

        # Index the images
        load_data = LoadImageData()
        image_list = load_data.from_folder([folder_path], shuffle=True)
        image_search_setup = ImageSearch(image_list, image_count=image_count)
        image_search_setup.run_index(reindex)

        # Index the text data
        corpus_list = LoadTextData().from_folder(folder_path)
        TextEmbedder().embed(corpus_list, reindex)

        return "Images and texts indexed successfully"
    except Exception as e:
        return str(e)


def search_similar_images(image, number_of_images=5):
    global image_search_setup

    try:
        number_of_images = int(number_of_images) if number_of_images else 5

        filename = secure_filename(image.name)
        image.save(os.path.join("./data", filename))

        query_image_path = os.path.join("./data", filename)
        similar_images = image_search_setup.get_similar_images(
            query_image_path, number_of_images
        )

        return similar_images
    except Exception as e:
        return str(e)


def search_from_text(text, number_of_images=5):
    global image_search_setup

    try:
        number_of_images = int(number_of_images) if number_of_images else 5

        # Search for similar images to the text query
        similar_images = image_search_setup.get_similar_images_to_text(
            text, number_of_images
        )

        # Search for similar texts to the text query
        TextEmbedder().load_embedding()
        similar_texts = TextSearch().find_similar(text)

        return {
            "images": similar_images,
            "texts": similar_texts,
        }
    except Exception as e:
        return str(e)


def cluster_images(n_clusters):
    global image_search_setup

    try:
        n_clusters = int(n_clusters)

        # Cluster the images
        image_search_setup.cluster_images(n_clusters)
        image_search_setup.save_clustered_images("./data/clusters")

        # Caption the first 15 images in each cluster
        for i in range(n_clusters):
            cluster_images = image_search_setup.get_clustered_images(i)
            captioned_images = image_search_setup.caption_images(
                cluster_images[:15])

            # Get the best topic for the first 15 images in each cluster
            best_topics = image_search_setup.get_best_topics(
                captioned_images["caption"].to_list()
            )
            new_folder_name = f"./data/clusters/{i}_{best_topics[0]}"
            os.rename(f"./data/clusters/{i}", new_folder_name)

        return "Images clustered successfully"
    except Exception as e:
        return str(e)


def get_cluster_images(cluster_id):
    global image_search_setup

    try:
        cluster_id = int(cluster_id)

        img_list = image_search_setup.get_clustered_images(cluster_id)

        return img_list
    except Exception as e:
        return str(e)


# Gradio UI
indexing_interface = gr.Interface(
    fn=index,
    inputs=[
        gr.Textbox(label="Folder Path"),
        gr.Textbox(label="Image Count (Optional)"),
        gr.Checkbox(label="Reindex"),
    ],
    outputs="text",
    title="Indexing Interface",
    description="Index images and texts for searching",
)

search_interface = gr.Interface(
    fn=search_similar_images,
    inputs=[
        gr.Image(label="Query Image"),
        gr.Textbox(label="Number of Images"),
    ],
    outputs="image",
    title="Search Similar Images Interface",
    description="Search for similar images to the query image",
)

search_text_interface = gr.Interface(
    fn=search_from_text,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Textbox(label="Number of Images"),
    ],
    outputs=["image", "text"],
    title="Search from Text Interface",
    description="Search for similar images and texts to the query text",
)

clustering_interface = gr.Interface(
    fn=cluster_images,
    inputs=gr.Textbox(label="Number of Clusters"),
    outputs="text",
    title="Clustering Interface",
    description="Cluster images for better organization",
)

get_cluster_images_interface = gr.Interface(
    fn=get_cluster_images,
    inputs=gr.Textbox(label="Cluster ID"),
    outputs="image",
    title="Get Cluster Images Interface",
    description="Get images from a specific cluster",
)

if __name__ == "__main__":
    indexing_interface.launch()
    search_interface.launch()
    search_text_interface.launch()
    clustering_interface.launch()
    get_cluster_images_interface.launch()
