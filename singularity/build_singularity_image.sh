source vars.sh
DOCKER_IMAGE=rna_fish_3d-production
SINGULARITY_NAME=rna_fish_3d
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v `pwd`:/output \
    --privileged \
    -t \
    --rm \
    singularityware/docker2singularity \
    $DOCKER_IMAGE \
    $SINGULARITY_NAME
