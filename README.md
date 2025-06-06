# MLOps Engineering Project

Machine Learning engineering project for models deployment

## Model Envrionment Setup and Development

### Local Development Setup

Python dependencies are managed by [pip-tools](https://pypi.org/project/pip-tools/). You need to create conda or venv for your env first.

pip-compile generates a requirements.txt file using the latest versions that fulfil the dependencies you specify in the supported files.

If pip-compile finds an existing requirements.txt file that fulfils the dependencies then no changes will be made, even if updates are available.

To force pip-compile to update all packages in an existing requirements.txt, run pip-compile --upgrade.


1. Prepapre Python env 

   Create conda environment from env file:

        conda env create -f environment.yml

    Alternatively, create create a virtual envrionment with 'venv'

        python3 -m venv env
        

2. Activate the environment:

        conda activate mlops

        source env/bin/activate

pip-compile requirements.in

4. Install model in editable mode for active development

        pip install -e .

5. Install as a package

        pip install

6. If a new package is added to the requirements.txt file:
   

        pip install --upgrade -r requirements.txt

7. Removing installed virtual environment

    For conda:

        conda remove --name mlops --all

    For pip env - delete associated directory

8. Updating with new dependenciespip-compile --upgrade

        pip-compile --upgrade
        
        pip install --ignore-installed -r requirements.txt

        Or

        pip install --upgrade --force-reinstall -r requirements.txt


### Local Deployment

#### Build an image
docker build . -t yevdeveloper/ml-ops:latest

#### Start a container with a given name
docker run --name=ml-ops --rm -p 5000:5000 -d yevdeveloper/ml-ops:latest

docker run --name=ml-ops -p 5000:5000 -d yevdeveloper/ml-ops:latest

#### Check stdout
docker logs ml-ops
docker logs -f ml-ops

#### Login running docker container
docker exec -it ml-ops  bash

#### Testing
curl -g http://localhost:5000/predict     --data-urlencode 'json={"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}'

#### To delete all images

docker rmi $(docker images -a)

#### To delete containers which are in exited state

docker rm $(docker ps -a -f status=exited -q)

#### To delete containers which are in created state

docker rm $(docker ps -a -f status=created -q)


### Troubleshooting

docker history yevdeveloper/ml-ops

### OpenShift integration

Build from repository
oc new-app --name=ml-ops https://github.com/yev-dev/ml-ops.git --strategy=docker
oc new-app --name=ml-ops git@github.com:yev-dev/ml-ops.git --context-dir=worker
oc new-app --name=ml-ops git@github.com:yev-dev/ml-ops.git#branch

Build from local copy


curl -g http://ml-ops-ml-pipeline.192.168.99.101.nip.io/predict --data-urlencode 'json={"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}'

#### Useful commands:
oc get endpoints
oc get route
oc get is -> Streams repo



### Troubleshooting

#### A docker container is not starting 


```
# Lists all created containers but potentually not run containers
docker container ls --all 

docker run --rm -it --name MYCONTAINER yevdeveloper/ml-ops:latest bash

```