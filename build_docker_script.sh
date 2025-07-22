docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg UNAME="user" -t mspinpoint/mindglide:super2_2024 .
