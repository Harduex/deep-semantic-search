import gradio as gr
from deep_semantic_search import (
    LoadTextData,
    TextEmbedder,
    TextSearch,
    LoadImageData,
    ImageSearch,
)
import os

from deep_semantic_search.rag import ask_question

DEFAULT_SEARCH_FOLDER_PATH = os.getenv("DEFAULT_SEARCH_FOLDER_PATH")

image_search_setup = None


def index(folder_path, files_count=None, reindex=True):
    global image_search_setup

    try:
        files_count = int(files_count) if files_count else None

        # Index the images
        load_data = LoadImageData()
        image_list = load_data.from_folder([folder_path], shuffle=True)
        image_search_setup = ImageSearch(image_list, image_count=files_count)
        image_search_setup.run_index(reindex)

        # Index the text data
        corpus_list = LoadTextData().from_folder(folder_path, corpus_count=files_count)
        TextEmbedder().embed(corpus_list, reindex)

        return "Indexing Successful!"
    except Exception as e:
        return str(e)


def search_from_image(image_file_path, number_of_images=5):
    global image_search_setup

    try:
        number_of_images = int(number_of_images) if number_of_images else 5

        # Search for similar images to the query image
        similar_images = image_search_setup.get_similar_images(
            image_file_path, number_of_images
        )

        # Search for similar texts to the query image
        query_image_caption = image_search_setup.caption_images([image_file_path])
        TextEmbedder().load_embedding()
        similar_texts = TextSearch().find_similar(query_image_caption["caption"][0])

        return similar_images, similar_texts
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

        return similar_images, similar_texts
    except Exception as e:
        return str(e)


def cluster_images(n_clusters, save=False):
    global image_search_setup

    try:
        n_clusters = int(n_clusters)
        image_data = image_search_setup.cluster_images(n_clusters)

        if save:
            image_search_setup.save_clustered_images("./data/clusters")

        return image_data.drop(columns=["features"])
    except Exception as e:
        return str(e)


def get_cluster_images(cluster_id):
    global image_search_setup

    try:
        cluster_id = int(cluster_id)
        img_paths = image_search_setup.get_clustered_images(cluster_id)
        return img_paths

    except Exception as e:
        return str(e)


def chat_with_your_data(question):
    similar_images, similar_texts = search_from_text(question, 1000)
    text_results = [result["text"] for result in similar_texts]

    # Uncomment the following code to caption the images and include them in the context
    # image_paths = {key: value for key, value in similar_images.items()}
    # image_captions = image_search_setup.caption_images(image_paths, "")
    # image_results = list(image_captions["caption"])
    # context = text_results + image_results
    
    context = text_results
    answer = ask_question(context, question)

    return answer

# Gradio UI
indexing_interface = gr.Interface(
    fn=index,
    inputs=[
        gr.Textbox(label="Folder Path", value=DEFAULT_SEARCH_FOLDER_PATH),
        gr.Textbox(label="Files Count (Leave blank for all)"),
        gr.Checkbox(label="Reindex"),
    ],
    outputs="text",
    title="Indexing Interface",
    description="Index images and texts for searching",
)

image_search_interface = gr.Interface(
    fn=search_from_image,
    inputs=[
        gr.Image(label="Query Image", type="filepath"),
        gr.Textbox(label="Number of Results", value="5"),
    ],
    outputs=[
        # gr.Image(label="Similar Images"),
        gr.Textbox(label="Similar Images"),
        gr.Textbox(label="Similar Texts"),
    ],
    title="Search from Image Interface",
    description="Search for similar images and texts from an image query",
)

text_search_interface = gr.Interface(
    fn=search_from_text,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Textbox(label="Number of Results", value="5"),
    ],
    outputs=[
        gr.Textbox(label="Similar Images"),
        gr.Textbox(label="Similar Texts"),
    ],
    title="Search from Text Interface",
    description="Search for similar images and texts from a text query",
)

clustering_interface = gr.Interface(
    fn=cluster_images,
    inputs=[
        gr.Textbox(label="Number of Clusters"),
        gr.Checkbox(label="Save Clusters"),
    ],
    outputs=[
        gr.Textbox(label="Image Clusters"),
    ],
    title="Clustering Interface",
    description="Cluster images for better organization",
)

get_cluster_images_interface = gr.Interface(
    fn=get_cluster_images,
    inputs=gr.Textbox(label="Cluster ID"),
    outputs=[
        # gr.Gallery(label="Cluster Images", ),
        gr.Textbox(label="Cluster Images"),
    ],
    title="Get Cluster Images Interface",
    description="Get images from a specific cluster",
)

chat_with_your_data_interface = gr.Interface(
    fn=chat_with_your_data,
    inputs=[
        gr.Textbox(label="Question"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Chat with your data",
    description="Ask a question and get an answer based on your data",
)

main_interface = gr.TabbedInterface(
    [
        indexing_interface,
        image_search_interface,
        text_search_interface,
        clustering_interface,
        get_cluster_images_interface,
        chat_with_your_data_interface,
    ],
    [
        "Indexing",
        "Search From Image",
        "Search From Text",
        "Images Clustering",
        "Get Cluster Images",
        "Chat With Your Data",
    ],
)

if __name__ == "__main__":
    main_interface.launch()
