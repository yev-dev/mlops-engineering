# ml-ops
Machine Learning DevOps project

#### Build an image
docker build . -t yevdeveloper/ml-ops:latest

#### Start a container with a given name
docker run --name=ml-ops --rm -p 5000:5000 -d yevdeveloper/ml-ops:latest

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

