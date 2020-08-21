import urllib.request

print('Just start me then unzip dataset ;)')

url = 'https://leon.bottou.org/_media/projects/infimnist.tar.gz'
urllib.request.urlretrieve(url, 'infmnist/infimnist.tar.gz')
