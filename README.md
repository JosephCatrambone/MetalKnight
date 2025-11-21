Port of an old anti-spam tool for Imgur.  In the middle of a migration + rewrite.

Quickstart (Docker):

1. `docker pull docker.io/josephcatrambone/metalknight:v0`
2. `run -p 8080:80  metalknight:v0`
3. Open http://localhost:8080/docs

Quickstart (Developers):

For a self-hosted container: 

 1. Copy (docker doesn't handle symlinks) the final versions of the models from the root model folders into the resoruces folder.
 2. `cd py`
 3. `docker build -t metalknight:latest .`
 4. If you're me: `docker tag metalknight:latest josephcatrambone/metalknight:latest`

Overview:

- models : Training, scripts, and data for building models.
- py : Primary root of the hosted application.
  - metal_knight : Source root of the application and all the glue logic for invoking models.
    - models : Python wrappers for models and an abstract base class to make inference consistent.
  - tests
  - resources : Model weights that get copied into the container image at the start of a build.  Changing these will lead to a full rebuild.
  - requirements.txt/dev
  - Dockerfile
- rust : Early efforts for a self-contained executable, similar to the above, but without Docker.  Goal is lower overhead and faster spinup on lambdas.
