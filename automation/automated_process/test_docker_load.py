import subprocess
import time
import requests
import os
def start_user_docker(image_name, port=5001):
    try:
        print("Starting the Docker container...\n")
        container_id = subprocess.check_output([
            "docker", "run", "-d", "-p", f"{port}:5001", image_name
        ]).strip().decode("utf-8")
        # current_dir = os.getcwd()
        # print("current dir is: ", os.getcwd())

        # container_id = subprocess.check_output([
        #     "docker", "run", "-d", "-p", f"{port}:5001", "-v", f"{current_dir}:/app", image_name
        # ]).strip().decode("utf-8")
        print(f"Started Docker container with ID: {container_id}")
        health_check_url = f"http://localhost:{port}/health-check"
        for _ in range(20):  
            try:
                response = requests.get(health_check_url, timeout=2)
                if response.status_code == 200:
                    print("Flask app is ready.")
                    return f"http://localhost:{port}", container_id
            except requests.exceptions.RequestException:
                print("Waiting for Flask app to be ready...")
                time.sleep(10)
        raise Exception("Flask app failed to start within the expected time.")
    except Exception as e:
        raise Exception(f"Failed to start Docker container: {e}")
    

def stop_user_docker(container_id):
    try:
        print(f"Stopping Docker container with ID: {container_id}")
        subprocess.check_call(["docker", "stop", container_id])
        print("Stopped Docker container.")
    except Exception as e:
        print(f"Error stopping Docker container: {e}")


def test_docker_start(image_name):
    try:
        docker_url, container_id = start_user_docker(image_name)
        print(f"Docker container started. Testing Flask app at {docker_url}")
        test_query = {"query": "Test query for Flask app"}
        url = f"{docker_url}/run-algo"
        response = requests.post(url, json=test_query)
        print("Response status code:", response.status_code)
        print("Response content:", response.json())
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if 'container_id' in locals():
            stop_user_docker(container_id)
            
if __name__ == "__main__":
    # test_docker_start("base-adpush-algo")
    test_docker_start("rag_gpt4o")