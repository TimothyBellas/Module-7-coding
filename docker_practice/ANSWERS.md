# Written answers

## 1. What happens when you docker run the same image twice?

Each `docker run` attempts to create and start a new container from the image.
Successful runs produce separate containers with different IDs and their own
writable filesystems. The image is reused; it is not changed by either container.
For example, running `docker run hello-world` twice produces two stopped
containers that appear in `docker ps -a`.

If both runs use the same explicit container name, the second run fails because
the name is already taken. If both try to publish the same host port while the
first is running, the second cannot start with that port. Use a different name
and an available host port for a second API container.

## 2. What happens to files created inside a container when it stops?

Files written to the container's ordinary writable filesystem remain when the
container stops. Restarting that same container with `docker start` preserves
them. A new container created from the original image does not inherit them.

Removing the container with `docker rm` deletes that writable layer. With
`--rm`, removal happens automatically when the container exits. Data stored
separately in a named volume or bind mount can survive container removal.

The `requests` package installed during the interactive Python exercise follows
the same rule: it belongs to that container, not the original Python image.

## 3. How is the -p flag used to map ports?

The format is `-p HOST_PORT:CONTAINER_PORT`: the left number is the port on your
computer, and the right number is the port where the app listens inside Docker.

- `-p 8000:8000` sends traffic from `http://localhost:8000` to container port 8000.
- `-p 8080:8000` sends traffic from `http://localhost:8080` to container port 8000.

The Dockerfile's `EXPOSE 8000` documents the port; the `-p` flag publishes it.

## References

- [Docker run](https://docs.docker.com/reference/cli/docker/container/run/)
- [Container storage layers](https://docs.docker.com/engine/storage/drivers/)
- [Publishing ports](https://docs.docker.com/get-started/docker-concepts/running-containers/publishing-ports/)
