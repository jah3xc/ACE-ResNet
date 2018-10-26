# ACENet
Using ACE to perform classification of Hyperspectral imagery

### Dependencies
This project uses `pipenv` to manage dependencies. To use, install pipenv, and then:
```
pipenv install --skip-lock
```

### Usage
To run ACENet: 
```
python3 ACENet [data_file] [ground_truth] 
```
where `data_file` and `ground_truth` are hyperspectral `.mat` files. 