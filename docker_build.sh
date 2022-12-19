docker buildx build --platform linux/amd64 -t fcollman/skelkeys:$1 . 

docker run -it emjoyce/skelkeys2:1.0.5 /bin/bash -c "mamba run -n skeleton_keys python task_worker.py https://sqs.us-west-2.amazonaws.com/629034007606/EmilySkeletons 180"

