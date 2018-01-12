FROM jicscicomp/bioformats

COPY requirements.txt .
RUN pip install -r requirements.txt
ADD scripts.tar.gz .
RUN pip install git+https://github.com/jic-dtool/smarttools#subdirectory=smarttools/smarttoolbase
