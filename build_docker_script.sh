docker build  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg UNAME="user" -t mspinpoint/mindglide:may2024 .
