docker build  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg UNAME="arman" -t armaneshaghi/monai:latest .
