
docker build -t imageutils_test -f Dockerfile --no-cache .

docker run -it -v $(pwd):/workspace imageutils_test python3 test.py
