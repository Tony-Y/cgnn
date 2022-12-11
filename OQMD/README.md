**Note that the official OQMD v1.2 dataset for CGNN is available at [this link](https://doi.org/10.5281/zenodo.7118055). If you want to skip the dataset construction below, please place all the contents of the archive (oqmd-v1_2-for-cgnn.7z) in this folder (cgnn/OQMD).**

**The Docker image for the qmpy environment [`tonyy999/qmpy-v1.2`](https://hub.docker.com/repository/docker/tonyy999/qmpy-v1.2) is available to skip configuring the qmpy environment.**

# The Open Quantum Materials Database (OQMD)
If you want to use the OQMD as a dataset for training CGNN models, you need a conda environment with the [qmpy](https://github.com/wolverton-research-group/qmpy) installed. [pymatgen](http://pymatgen.org) is also necessary for the use of the data format of the Materials Project. Because the OQMD uses MySQL 5.7, first install it.

If your Linux OS is Ubuntu 16.04 or 18.04, you can install MySQL 5.7 as follows:

```
sudo apt install mysql-server mysql-client libmysqlclient-dev
```

Configure the MySQL server.
Here is an example of the MySQL server configuration file `my.cnf`:

```
[mysqld]
server_id=1
innodb_buffer_pool_size=1G
innodb_log_file_size=1G
innodb_flush_method=O_DIRECT
innodb_io_capacity=4000
innodb_io_capacity_max=8000
# For importing data
skip_innodb_doublewrite
```

After configuring, restart the MySQL server. If you use Ubuntu, execute the following command:
```
sudo systemctl restart mysql
```

Enter a MySQL session as root.

```
mysql -u root -p
```

Then, create a new user and grant that user permissions.

```
mysql> CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
mysql> GRANT ALL PRIVILEGES ON *.* TO 'username'@'localhost';
```

`username` may be your username. `password` should be a strong password to secure your MySQL account.

You can get the OQMD v1.2 from [the OQMD site](http://oqmd.org). After downloading it, first create the database in a MySQL session.

```
mysql> CREATE DATABASE oqmd_v1_2;
```

Then, restore the downloaded data to the empty database.

```
> zcat qmdb__v1_2__062018.sql.gz | mysql -u $USER -p oqmd_v1_2
[Enter your MySQL password]
```

where `$USER` is your username. The database finally grows approximate 80 GB.
Edit the MySQL server configuration file to remove `skip_innodb_doublewrite` once finished.

Next, create the environment `qmpy` as follows:

```
conda create --name qmpy
conda install -n qmpy scikit-learn matplotlib python=2.7
conda activate  qmpy
pip install pymatgen==2018.12.12 monty==1.0.3
pip install qmpy==1.2.0 ase==3.17
pip install pydash tqdm joblib
```
This step requires `gcc`. So, you have to install `gcc` in your computer if not yet.

You must edit the [Django](https://www.djangoproject.com) setting file in the database directory of `qmpy`.

```
python -c "from qmpy import db; print db.__path__"
```

This shows the database directory like

```
['.../qmpy/lib/python2.7/site-packages/qmpy/db']
```

Edit `settings.py` in the directory `.../qmpy/db` according to your OQMD database. It is preferable to change `DEBUG=True` to `DEBUG=False` in order to resolve the memory issue of [Django](https://docs.djangoproject.com). Add your database to `DATABASES`.

```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'oqmd_v1_2',
        'USER': 'username',
        'PASSWORD': 'password',
        'HOST': '127.0.0.1',
        'PORT': ''
    }
}
```

`username` and `password` must be the username and password of your MySQL account.
The empty `PORT` makes Django use the default port number of MySQL `3306`. If the MySQL server uses a different port, set `PORT` to it. Remove unwanted hosts from `ALLOWED_HOSTS` (*i.e.*, `ALLOWED_HOSTS = []`).

To create dataset files used for `cgnn.py`, run `oqmd_data.py`, `mp_graph.py`, and `oqmd.py` sequentially in the directory `cgnn/OQMD` as follows:

```
conda activate qmpy
mkdir data
python ../tools/oqmd_data.py
python ../tools/mp_graph.py
```

`oqmd_data.py` takes a few hours to retrieve 561k OQMD entries. `mp_graph.py` takes a few hours to create all the crystal graphs for the OQMD dataset when using 8 CPUs concurrently. The directory `data` finally becomes 294 MB in total size.

```
python oqmd.py data
```

This script will create the dataset files:

```
config.json
graph_data.npz
split.json
targets.csv
```

The official dataset contains some incorrect space groups. So, your generaged `targets.csv` differs from the official one. Errata are available at [this link](https://github.com/Tony-Y/oqmd-v1.2-dataset-for-cgnn/blob/main/errata_spacegroup.csv).

(c) 2019 Takenori Yamamoto
