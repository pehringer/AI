# Prerequisites:
- ```make```
- ```docker```

# Steps:
1) Startup docker ```sudo systemctl start docker```
2) Download ollama docker image (only needs to be done once) ```make download_ollama```
3) Run a AI model in a docker container:
  - 1.3 GB ```make start_llama3_2_1b```
  - 2.0 GB ```make start_llama3_2_3b```
  - 3.8 GB ```start_codellama_7b```
  - 7.4 GB ```start_codellama_13b```
  - 19 GB ```start_codellama_34b```
4) Stop and cleanup docker container ```make stop```
5) Stop docker ```sudo systemctl stop docker```

# Useful Docker Commands
### Docker Engine Commands

Start Docker engine:
```
sudo systemctl start docker
```

Stop Docker engine:
```
sudo systemctl stop docker
sudo systemctl stop docker.socket
```

### Docker Image Commands

Build Docker image:
```
sudo docker build -t [IMAGE_NAME] [DOCKERFILE_PATH]
```

List Docker images:
```
sudo docker image ls
```

Delete Docker image:
```
sudo docker image rm [IMAGE_NAME]
```

### Docker Container Commands

Run Docker container:
```
sudo docker run [IMAGE_NAME]
```

List Docker containers:
```
sudo docker ps -a
```

Display Docker container logs:
```
sudo docker logs [CONTAINER_NAME]
```

Stop Docker Container:
```
sudo docker stop [CONTAINER_NAME]
```

Delete Docker container (only if stopped):
```
sudo docker rm [CONTAINER_NAME]
```
