# The Open Quantum Materials Database (OQMD)
If you want to use the OQMD as a dataset for training CGNN models, you need a conda environment with the [qmpy](https://github.com/wolverton-research-group/qmpy) installed. [pymatgen](http://pymatgen.org) is also necessary for the use of the data format of the Materials Project. Because the OQMD uses MySQL, first install it.

If your Linux OS is Ubuntu, you can install MySQL as follows:

```
sudo apt install mysql-server mysql-client libmysqlclient-dev
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
zcat qmdb__v1_2__062018.sql.gz | mysql -u $USER -p $PASSWORD oqmd_v1_2
```

where `$USER` is your username and `$PASSWORD` is your MySQL password. The database finally grows approximate 80 GB, but the intermediate stages need larger storage (*ca.* 200 GB).

Next, create the environment `qmpy` as follows:

```
conda create --name qmpy
conda install -n qmpy -c matsci pymatgen scikit-learn python=2.7
source activate  qmpy
pip install qmpy==1.2.0
pip install pydash tqdm joblib
```

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
source activate qmpy
mkdir data
python ../tools/oqmd_data.py
python ../tools/mp_graph.py
```

`oqmd_data.py` takes a few hours to retrieve 561k OQMD entries. You will get the failure to determine the space group for the entry id `1018092`, but may ignore it because no space groups are used in the further processing. `mp_graph.py` takes a few hours to create all the crystal graphs for the OQMD dataset when using 8 CPUs concurrently. The directory `data` finally becomes 294 MB in total size.

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

(c) 2019 Takenori Yamamoto
