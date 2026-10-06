# Docker commands practice

This project provides a FastAPI app, a Dockerfile, dependency requirements, and
the three written answers in `ANSWERS.md`.

Open Docker Desktop and wait for its engine to start. Use Linux containers for
the Python image. You do not need a local Python installation or a virtual
environment for these steps; Python and the dependencies run inside Docker.

## 1. Verify Docker

Open PowerShell:

```powershell
docker version
docker run hello-world
```

`docker version` should show both Client and Server information. The next
command should display the hello-world success message and then exit.

## 2. Run Python interactively

From PowerShell:

```powershell
docker run -it python:3.11-slim bash
```

Your prompt now belongs to Bash inside the Linux container. Run these commands
there, one at a time:

```bash
python --version
pip install requests
python -c "import requests; print(requests.__version__)"
echo 'Created inside this container' > /tmp/practice.txt
cat /tmp/practice.txt
exit
```

The Python version should begin with `3.11`. Installing and importing `requests`
should succeed. `exit` returns you to PowerShell and stops this container.

To check that stopping preserved its file, find this Python container's ID:

```powershell
docker ps -a
```

Replace `PASTE_PYTHON_CONTAINER_ID_HERE` below with that ID, then run:

```powershell
$pythonContainerId = "PASTE_PYTHON_CONTAINER_ID_HERE"
docker start -ai $pythonContainerId
```

Inside the restarted container:

```bash
cat /tmp/practice.txt
python -c "import requests; print(requests.__version__)"
exit
```

You should see the file content and the installed package again. Back in
PowerShell, remove only that stopped practice container:

```powershell
docker rm $pythonContainerId
```

## 3. Build and run the FastAPI app

Extract the ZIP. Open its `docker-practice` folder in File Explorer, right-click
inside it, and choose to open a terminal. Use PowerShell in the folder containing
`app.py`, `Dockerfile`, and `requirements.txt`.

Build the image. The final dot tells Docker to use the current folder:

```powershell
docker build -t my-first-api .
```

Run the container in the foreground, as in the assignment:

```powershell
docker run -p 8000:8000 --name my-first-api-container my-first-api
```

Leave this PowerShell window open. In your browser, visit:

- [API root](http://localhost:8000)
- [Swagger UI](http://localhost:8000/docs)

The root should return:

```json
{"message":"Hello from Docker!"}
```

In Swagger UI, expand `GET /`, click **Try it out**, and then **Execute**.
Expect status 200 and the same JSON.

## 4. Inspect, stop, remove, and run again

Open a second PowerShell window while the API is running. Run each command:

```powershell
docker ps
docker ps -a
docker images
docker logs my-first-api-container
docker inspect my-first-api-container
Invoke-RestMethod -Uri "http://localhost:8000/"
```

The first command should list the API container as running. `docker ps -a` also
includes stopped containers, and `docker images` should include `my-first-api`,
`python`, and `hello-world`.

Practice stopping and removing this API container. A container name works in
these commands just like the ID from `docker ps`:

```powershell
docker stop my-first-api-container
docker ps
docker ps -a
docker rm my-first-api-container
docker ps -a
```

After stopping, it disappears from `docker ps` but remains in `docker ps -a`.
After removal, it disappears from both. The image remains available.

Start a new container in the background so the final deliverable is running:

```powershell
docker run -d --name my-first-api-container -p 8000:8000 my-first-api
docker ps
```

Once the application has started, verify it again:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/"
```

The `-d` flag lets the app keep running while PowerShell returns to its prompt.

## 5. Try the same image twice

Running this twice creates two separate containers; each exits after printing:

```powershell
docker run hello-world
docker run hello-world
docker ps -a
```

To run a second API instance alongside the final container, use a different
name and a different host port:

```powershell
docker run -d --name my-second-api-container -p 8001:8000 my-first-api
```

Visit [the second API](http://localhost:8001) after it starts. Then clean up only
the second instance, leaving your first API container running:

```powershell
docker stop my-second-api-container
docker rm my-second-api-container
```

## Common fixes

- **Docker is not recognized:** make sure Docker Desktop is installed, then open
  a new PowerShell window.
- **Cannot connect to the Docker daemon:** start Docker Desktop and wait for its
  engine to become ready.
- **Port 8000 is already in use:** stop the app using it, or run this container
  with `-p 8001:8000` and visit `http://localhost:8001`.
- **The container name is already in use:** run `docker ps -a` and identify the
  earlier practice container. If it is the same app, you can use
  `docker start my-first-api-container` to restart it instead of creating another.
- **Docker cannot find the Dockerfile:** open PowerShell in the extracted folder
  containing the file. Its filename must be `Dockerfile`, without `.txt`.

## Submission checklist

- [ ] The hello-world command succeeded.
- [ ] Python 3.11 ran inside an interactive container.
- [ ] `requests` installed and imported successfully.
- [ ] The image built successfully.
- [ ] The API returned the expected JSON at `http://localhost:8000`.
- [ ] Running, stopped, and removed containers were checked.
- [ ] The final API container is running and visible in `docker ps`.
- [ ] `ANSWERS.md` is included with the submission.

These are expected results to verify on your own computer. Docker was not
available in the preparation workspace, so the image build and container run
have not been tested here.

The application passed local HTTP checks for `/`, `/openapi.json`, and `/docs`
using Python 3.12.14 outside Docker. Each endpoint returned status 200, the root
returned the expected JSON, and the OpenAPI schema included `GET /`.

## References

- [Docker run](https://docs.docker.com/reference/cli/docker/container/run/)
- [FastAPI in Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Uvicorn server command](https://fastapi.tiangolo.com/deployment/manually/)
