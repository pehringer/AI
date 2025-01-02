download_ollama:
	sudo docker pull ollama/ollama:latest

start_llama3_2_1b:
	sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
	sudo docker exec -it ollama ollama run llama3.2:1b

start_llama3_2_3b:
	sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
	sudo docker exec -it ollama ollama run llama3.2:3b

start_codellama_7b:
	sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
	sudo docker exec -it ollama ollama run codellama:7b

start_codellama_13b:
	sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
	sudo docker exec -it ollama ollama run codellama:13b

start_codellama_34b:
	sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
	sudo docker exec -it ollama ollama run codellama:34b

stop:
	sudo docker stop ollama
	sudo docker rm ollama
