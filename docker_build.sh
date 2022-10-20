docker buildx build --no-cache --platform linux/amd64 -t fcollman/skelkeys:$1 . --build-arg GITHUB_TOKEN=$GITHUB_TOKEN
