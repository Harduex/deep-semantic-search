import requests

BASE_URL = "http://localhost:5000"
FOLDER_PATH = "/home/simeon/mnt/Files/Projects/Programming/IT Step/AI_2023/experiments/datasets/caltech-101/101_ObjectCategories"
IMAGE_PATH = "/home/simeon/mnt/Files/Projects/Programming/IT Step/AI_2023/experiments/datasets/caltech-101/101_ObjectCategories/strawberry/image_0022.jpg"
IMAGE_COUNT = 1000
NUMBER_OF_IMAGES = 5
TEXT = "cat"
N_CLUSTERS = 5
CLUSTER_ID = 0


def test_index_images():
    response = requests.post(
        f"{BASE_URL}/index",
        json={"folder_path": FOLDER_PATH, "image_count": IMAGE_COUNT},
    )
    assert response.status_code == 200
    assert "Images indexed successfully" in response.text


def test_search_similar_images():
    with open(IMAGE_PATH, "rb") as img:
        response = requests.post(
            f"{BASE_URL}/search",
            files={"image": img},
            data={"number_of_images": NUMBER_OF_IMAGES},
        )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_search_similar_images_to_text():
    response = requests.post(
        f"{BASE_URL}/search-text",
        json={"text": TEXT, "number_of_images": NUMBER_OF_IMAGES},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_cluster_images():
    response = requests.post(f"{BASE_URL}/cluster", json={"n_clusters": N_CLUSTERS})
    assert response.status_code == 200
    assert "Images clustered successfully" in response.text


def test_get_cluster_images():
    response = requests.post(
        f"{BASE_URL}/get-cluster-images", json={"cluster_id": CLUSTER_ID}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "images" in response.json()
